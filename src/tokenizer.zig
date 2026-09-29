//! Zeno — Tokenizer vocabulary representation and candidate validation.
//!
//! Zero-copy vocabulary extraction from GGUF metadata, special token handling,
//! and single-token candidate validation matching specs/tokenizer.md.

const std = @import("std");
const gguf = @import("gguf.zig");
const root = @import("root.zig");

pub const TokenId = u32;

pub const TokenResolutionError = error{
    MultiTokenCandidateNotAllowed,
    TokenNotFound,
};

pub const TokenizerError = error{
    MissingTokensMetadata,
    MalformedTokensMetadata,
    MultiTokenCandidateNotAllowed,
    DuplicateCandidateToken,
    TokenNotFound,
};

pub const MergeInfo = struct {
    rank: u32,
    merged_id: TokenId,
};

pub const ControlToken = struct {
    text: []const u8,
    id: TokenId,
};

pub const Tokenizer = struct {
    /// Zero-copy slices pointing directly into GGUF mmap data for all 262,144 tokens.
    vocab: []const []const u8,
    /// Fast string-to-token lookup table.
    token_to_id: std.StringHashMapUnmanaged(TokenId),
    /// Fast BPE merge table mapping (left_id << 32 | right_id) -> MergeInfo.
    merges: std.AutoHashMapUnmanaged(u64, MergeInfo),
    /// Control and special tokens, sorted by text length descending.
    control_tokens: []const ControlToken,

    // Special token IDs from specs/tokenizer.md
    pad_id: TokenId = root.SpecialTokens.pad,
    eos_id: TokenId = root.SpecialTokens.eos,
    bos_id: TokenId = root.SpecialTokens.bos,
    unk_id: TokenId = root.SpecialTokens.unk,
    mask_id: TokenId = root.SpecialTokens.mask,

    /// Initializes tokenizer vocabulary, control tokens, and BPE merges from a parsed GGUF file.
    pub fn initFromGguf(allocator: std.mem.Allocator, gfile: gguf.GgufFile) !Tokenizer {
        const tokens_entry = gfile.findMetadata("tokenizer.ggml.tokens") orelse {
            return error.MissingTokensMetadata;
        };

        if (tokens_entry.value_type != .array) {
            return error.MalformedTokensMetadata;
        }

        // Parse array header: subtype (u32) + length (u64)
        const data = gfile.data;
        if (tokens_entry.offset + 12 > data.len) return error.MalformedTokensMetadata;

        const sub_type = std.mem.readInt(u32, data[tokens_entry.offset..][0..4], .little);
        if (sub_type != @backingInt(gguf.MetadataValueType.string)) {
            return error.MalformedTokensMetadata;
        }

        const count: usize = @intCast(std.mem.readInt(u64, data[tokens_entry.offset + 4 ..][0..8], .little));
        if (count != root.ModelConfig.vocab_size) {
            return error.MalformedTokensMetadata;
        }

        // Allocate vocabulary slice
        const vocab = try allocator.alloc([]const u8, count);
        errdefer allocator.free(vocab);

        // Zero-copy slice extraction
        var cursor = tokens_entry.offset + 12;
        var token_map: std.StringHashMapUnmanaged(TokenId) = .{};
        errdefer token_map.deinit(allocator);

        // Pre-allocate map capacity for common vocabulary lookups
        try token_map.ensureTotalCapacity(allocator, @intCast(count));

        for (0..count) |i| {
            if (cursor + 8 > data.len) return error.MalformedTokensMetadata;
            const len: usize = @intCast(std.mem.readInt(u64, data[cursor..][0..8], .little));
            cursor += 8;

            if (cursor + len > data.len) return error.MalformedTokensMetadata;
            const str = data[cursor .. cursor + len];
            cursor += len;

            vocab[i] = str;
            // Insert into fast lookup map (first occurrence wins)
            if (!token_map.contains(str)) {
                token_map.putAssumeCapacity(str, @intCast(i));
            }
        }

        // Parse control tokens from tokenizer.ggml.token_type if present
        var control_tokens_list: std.ArrayList(ControlToken) = .empty;
        errdefer control_tokens_list.deinit(allocator);

        if (gfile.findMetadata("tokenizer.ggml.token_type")) |tt_entry| {
            if (tt_entry.value_type == .array and tt_entry.offset + 12 <= data.len) {
                const tt_count: usize = @intCast(std.mem.readInt(u64, data[tt_entry.offset + 4 ..][0..8], .little));
                const tt_start = tt_entry.offset + 12;
                if (tt_start + tt_count * 4 <= data.len) {
                    for (0..@min(tt_count, count)) |i| {
                        const tp = std.mem.readInt(i32, data[tt_start + i * 4 ..][0..4], .little);
                        // Types 3 (control) and 4 (user-defined special tokens)
                        if (tp == 3 or tp == 4) {
                            try control_tokens_list.append(allocator, .{
                                .text = vocab[i],
                                .id = @intCast(i),
                            });
                        }
                    }
                }
            }
        }

        // Sort control tokens by text length descending so longer tokens match first greedily
        std.mem.sort(ControlToken, control_tokens_list.items, {}, struct {
            fn lessThan(_: void, a: ControlToken, b: ControlToken) bool {
                return a.text.len > b.text.len;
            }
        }.lessThan);

        const control_tokens = try control_tokens_list.toOwnedSlice(allocator);
        errdefer allocator.free(control_tokens);

        // Parse merges table from tokenizer.ggml.merges
        var merges_map: std.AutoHashMapUnmanaged(u64, MergeInfo) = .{};
        errdefer merges_map.deinit(allocator);

        if (gfile.findMetadata("tokenizer.ggml.merges")) |merges_entry| {
            if (merges_entry.value_type == .array and merges_entry.offset + 12 <= data.len) {
                const merges_count: usize = @intCast(std.mem.readInt(u64, data[merges_entry.offset + 4 ..][0..8], .little));
                try merges_map.ensureTotalCapacity(allocator, @intCast(merges_count));

                var m_cursor = merges_entry.offset + 12;
                for (0..merges_count) |i| {
                    if (m_cursor + 8 > data.len) return error.MalformedTokensMetadata;
                    const len: usize = @intCast(std.mem.readInt(u64, data[m_cursor..][0..8], .little));
                    m_cursor += 8;

                    if (m_cursor + len > data.len) return error.MalformedTokensMetadata;
                    const m = data[m_cursor .. m_cursor + len];
                    m_cursor += len;

                    if (std.mem.indexOfScalar(u8, m, ' ')) |space_idx| {
                        const left = m[0..space_idx];
                        const right = m[space_idx + 1 ..];
                        if (token_map.get(left)) |lid| {
                            if (token_map.get(right)) |rid| {
                                var concat_buf: [128]u8 = undefined;
                                if (left.len + right.len <= concat_buf.len) {
                                    @memcpy(concat_buf[0..left.len], left);
                                    @memcpy(concat_buf[left.len .. left.len + right.len], right);
                                    const combined = concat_buf[0 .. left.len + right.len];
                                    if (token_map.get(combined)) |mid| {
                                        const key = (@as(u64, lid) << 32) | @as(u64, rid);
                                        merges_map.putAssumeCapacity(key, .{
                                            .rank = @intCast(i),
                                            .merged_id = mid,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return Tokenizer{
            .vocab = vocab,
            .token_to_id = token_map,
            .merges = merges_map,
            .control_tokens = control_tokens,
        };
    }

    /// Releases allocated vocabulary memory, merges, control tokens, and lookup hash map.
    pub fn deinit(self: *Tokenizer, allocator: std.mem.Allocator) void {
        allocator.free(self.vocab);
        self.token_to_id.deinit(allocator);
        self.merges.deinit(allocator);
        allocator.free(self.control_tokens);
    }

    /// Finds a token ID by exact byte slice match.
    pub fn findTokenId(self: Tokenizer, token_str: []const u8) ?TokenId {
        return self.token_to_id.get(token_str);
    }

    /// Decodes a token ID to its string representation.
    pub fn decode(self: Tokenizer, token_id: TokenId) ?[]const u8 {
        if (token_id >= self.vocab.len) return null;
        return self.vocab[token_id];
    }

    /// Encodes a non-control text slice using SentencePiece space normalization and BPE merges.
    fn encodePiece(self: Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]TokenId {
        if (text.len == 0) return &.{};

        // 1. Replace ASCII space with SentencePiece meta-space '\xe2\x96\x81'
        var space_count: usize = 0;
        for (text) |b| {
            if (b == ' ') space_count += 1;
        }
        const norm_len = text.len + space_count * 2;
        const norm = try allocator.alloc(u8, norm_len);
        defer allocator.free(norm);

        var out_idx: usize = 0;
        for (text) |b| {
            if (b == ' ') {
                norm[out_idx] = 0xe2;
                norm[out_idx + 1] = 0x96;
                norm[out_idx + 2] = 0x81;
                out_idx += 3;
            } else {
                norm[out_idx] = b;
                out_idx += 1;
            }
        }

        // 2. Initial tokenization into character chunks or byte fallback tokens
        var tokens: std.ArrayList(TokenId) = .empty;
        errdefer tokens.deinit(allocator);

        var i: usize = 0;
        while (i < norm.len) {
            var matched = false;
            const max_len = @min(4, norm.len - i);
            var l = max_len;
            while (l > 0) : (l -= 1) {
                const chunk = norm[i .. i + l];
                if (self.token_to_id.get(chunk)) |id| {
                    try tokens.append(allocator, id);
                    i += l;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                // Byte fallback: tokens 238..493 correspond to <0x00>..<0xFF>
                const byte_val: u32 = norm[i];
                try tokens.append(allocator, 238 + byte_val);
                i += 1;
            }
        }

        // 3. Iterative BPE merge loop: greedily merge lowest-rank adjacent pair
        while (tokens.items.len >= 2) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_idx: ?usize = null;
            var best_merged: TokenId = 0;

            var j: usize = 0;
            while (j < tokens.items.len - 1) : (j += 1) {
                const key = (@as(u64, tokens.items[j]) << 32) | @as(u64, tokens.items[j + 1]);
                if (self.merges.get(key)) |minfo| {
                    if (minfo.rank < best_rank) {
                        best_rank = minfo.rank;
                        best_idx = j;
                        best_merged = minfo.merged_id;
                    }
                }
            }

            if (best_idx) |idx| {
                tokens.items[idx] = best_merged;
                _ = tokens.orderedRemove(idx + 1);
            } else {
                break;
            }
        }

        return try tokens.toOwnedSlice(allocator);
    }

    /// Encodes a full input text string into token IDs.
    /// Recognizes control tokens greedily and applies BPE to all intervening text.
    pub fn encode(self: Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]TokenId {
        if (text.len == 0) return &.{};

        var result: std.ArrayList(TokenId) = .empty;
        errdefer result.deinit(allocator);

        var i: usize = 0;
        while (i < text.len) {
            // 1. Check if text[i..] matches any control token
            var matched_ctrl: ?ControlToken = null;
            for (self.control_tokens) |ct| {
                if (std.mem.startsWith(u8, text[i..], ct.text)) {
                    matched_ctrl = ct;
                    break;
                }
            }

            if (matched_ctrl) |ct| {
                try result.append(allocator, ct.id);
                i += ct.text.len;
                continue;
            }

            // 2. Find nearest control token start in text[i..]
            var next_ctrl_pos = text.len;
            for (self.control_tokens) |ct| {
                if (std.mem.indexOf(u8, text[i..], ct.text)) |pos| {
                    const abs_pos = i + pos;
                    if (abs_pos < next_ctrl_pos) {
                        next_ctrl_pos = abs_pos;
                    }
                }
            }

            const piece = text[i..next_ctrl_pos];
            const piece_tokens = try self.encodePiece(allocator, piece);
            defer allocator.free(piece_tokens);
            try result.appendSlice(allocator, piece_tokens);
            i = next_ctrl_pos;
        }

        return try result.toOwnedSlice(allocator);
    }

    /// Validates and resolves a candidate string into a single token ID.
    /// Checks both exact token match and leading-space prefixed match (' ' = U+2581 = \xe2\x96\x81).
    pub fn resolveCandidateToken(self: Tokenizer, text: []const u8) TokenResolutionError!TokenId {
        if (text.len == 0) {
            return error.TokenNotFound;
        }

        // 1. Try exact match
        if (self.findTokenId(text)) |id| {
            return id;
        }

        // 2. Try with leading space ' ' (\xe2\x96\x81)
        var buf: [128]u8 = undefined;
        const gemma_space = "\xe2\x96\x81";
        if (gemma_space.len + text.len <= buf.len) {
            @memcpy(buf[0..gemma_space.len], gemma_space);
            @memcpy(buf[gemma_space.len .. gemma_space.len + text.len], text);
            const spaced_token = buf[0 .. gemma_space.len + text.len];
            if (self.findTokenId(spaced_token)) |id| {
                return id;
            }
        }

        // If string contains whitespace, it is demonstrably multi-token
        if (std.mem.indexOfScalar(u8, text, ' ') != null) {
            return error.MultiTokenCandidateNotAllowed;
        }

        return error.TokenNotFound;
    }
};

test "Tokenizer: verify special control tokens and candidate resolution on local model" {
    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    // Skip test if model file is not present locally
    var model = root.DiffusionGemmaModel.loadFromFile(allocator, model_path) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer model.deinit();

    var tokenizer = try Tokenizer.initFromGguf(allocator, model.gguf_file);
    defer tokenizer.deinit(allocator);

    // Verify first 5 control tokens match specs/tokenizer.md
    try std.testing.expectEqualStrings("<pad>", tokenizer.decode(root.SpecialTokens.pad).?);
    try std.testing.expectEqualStrings("<eos>", tokenizer.decode(root.SpecialTokens.eos).?);
    try std.testing.expectEqualStrings("<bos>", tokenizer.decode(root.SpecialTokens.bos).?);
    try std.testing.expectEqualStrings("<unk>", tokenizer.decode(root.SpecialTokens.unk).?);
    try std.testing.expectEqualStrings("<mask>", tokenizer.decode(root.SpecialTokens.mask).?);

    // Verify candidate resolution for single tokens
    const id_yes = try tokenizer.resolveCandidateToken("Yes");
    try std.testing.expect(id_yes == 10784 or id_yes == 8438);

    const id_no = try tokenizer.resolveCandidateToken("No");
    try std.testing.expect(id_no == 3771 or id_no == 2301);

    const id_a = try tokenizer.resolveCandidateToken("A");
    try std.testing.expect(id_a == 236776);

    // Verify multi-token candidate rejection vs token not found
    try std.testing.expectError(error.MultiTokenCandidateNotAllowed, tokenizer.resolveCandidateToken("A Very Long Nonexistent String That Cannot Be One Token"));
    try std.testing.expectError(error.TokenNotFound, tokenizer.resolveCandidateToken("nonexistenttokenxyz123"));
}

test "Tokenizer: verify encode() handles control tokens and BPE merges" {
    const allocator = std.testing.allocator;
    const model_path = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf";

    var model = root.DiffusionGemmaModel.loadFromFile(allocator, model_path) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer model.deinit();

    var tokenizer = try Tokenizer.initFromGguf(allocator, model.gguf_file);
    defer tokenizer.deinit(allocator);

    // Test encoding single words
    const tokens_yes = try tokenizer.encode(allocator, "Yes");
    defer allocator.free(tokens_yes);
    try std.testing.expectEqual(@as(usize, 1), tokens_yes.len);
    try std.testing.expectEqual(@as(TokenId, 10784), tokens_yes[0]);

    const tokens_no = try tokenizer.encode(allocator, "No");
    defer allocator.free(tokens_no);
    try std.testing.expectEqual(@as(usize, 1), tokens_no.len);
    try std.testing.expectEqual(@as(TokenId, 3771), tokens_no[0]);

    // Test encoding text with control tokens
    const prompt = "<|turn>user\nIs the database operational?<turn|>\n<|turn>model\nAnswer: ";
    const tokens_prompt = try tokenizer.encode(allocator, prompt);
    defer allocator.free(tokens_prompt);

    // First token must be <|turn> (ID 105)
    try std.testing.expect(tokens_prompt.len > 10);
    try std.testing.expectEqual(@as(TokenId, 105), tokens_prompt[0]);
    // Must contain <turn|> (ID 106)
    var found_turn_close = false;
    for (tokens_prompt) |tid| {
        if (tid == 106) found_turn_close = true;
    }
    try std.testing.expect(found_turn_close);
}

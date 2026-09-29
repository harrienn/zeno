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

pub const Tokenizer = struct {
    /// Zero-copy slices pointing directly into GGUF mmap data for all 262,144 tokens.
    vocab: []const []const u8,
    /// Fast string-to-token lookup table.
    token_to_id: std.StringHashMapUnmanaged(TokenId),

    // Special token IDs from specs/tokenizer.md
    pad_id: TokenId = root.SpecialTokens.pad,
    eos_id: TokenId = root.SpecialTokens.eos,
    bos_id: TokenId = root.SpecialTokens.bos,
    unk_id: TokenId = root.SpecialTokens.unk,
    mask_id: TokenId = root.SpecialTokens.mask,

    /// Initializes tokenizer vocabulary from a parsed GGUF file.
    pub fn initFromGguf(allocator: std.mem.Allocator, gfile: gguf.GgufFile) !Tokenizer {
        const entry = gfile.findMetadata("tokenizer.ggml.tokens") orelse {
            return error.MissingTokensMetadata;
        };

        if (entry.value_type != .array) {
            return error.MalformedTokensMetadata;
        }

        // Parse array header: subtype (u32) + length (u64)
        const data = gfile.data;
        if (entry.offset + 12 > data.len) return error.MalformedTokensMetadata;

        const sub_type = std.mem.readInt(u32, data[entry.offset..][0..4], .little);
        if (sub_type != @intFromEnum(gguf.MetadataValueType.string)) {
            return error.MalformedTokensMetadata;
        }

        const count: usize = @intCast(std.mem.readInt(u64, data[entry.offset + 4 ..][0..8], .little));
        if (count != root.ModelConfig.vocab_size) {
            return error.MalformedTokensMetadata;
        }

        // Allocate vocabulary slice
        const vocab = try allocator.alloc([]const u8, count);
        errdefer allocator.free(vocab);

        // Zero-copy slice extraction
        var cursor = entry.offset + 12;
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

        return Tokenizer{
            .vocab = vocab,
            .token_to_id = token_map,
        };
    }

    /// Releases allocated vocabulary memory and lookup hash map.
    pub fn deinit(self: *Tokenizer, allocator: std.mem.Allocator) void {
        allocator.free(self.vocab);
        self.token_to_id.deinit(allocator);
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

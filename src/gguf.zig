//! Zeno — GGUF v3 binary format parser.
//!
//! Provides zero-copy parsing of container headers, metadata key-value pairs,
//! and tensor descriptors from memory-mapped files.

const std = @import("std");
const quant = @import("quant.zig");

pub const GGUF_MAGIC = [4]u8{ 'G', 'G', 'U', 'F' };

pub const MetadataValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool_val = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

/// Descriptor for a tensor within a GGUF file.
pub const TensorInfo = struct {
    name: []const u8,
    n_dimensions: u32,
    dimensions: [4]u64,
    ggml_type: u32,
    offset: u64, // relative to data_offset
};

/// Entry in the metadata key-value table.
pub const MetadataEntry = struct {
    key: []const u8,
    value_type: MetadataValueType,
    offset: usize, // offset in file data where value payload begins
};

pub const GgufError = error{
    InvalidMagic,
    UnsupportedVersion,
    UnexpectedEof,
    MalformedString,
    MalformedArray,
    TensorNotFound,
    UnsupportedGgmlType,
};

/// Computes the total byte size required for a tensor payload using overflow-safe arithmetic.
pub fn getTensorByteSize(tensor: TensorInfo) !usize {
    var total_elements: usize = 1;
    for (0..tensor.n_dimensions) |d| {
        const dim_usize = std.math.cast(usize, tensor.dimensions[d]) orelse return error.UnexpectedEof;
        total_elements = std.math.mul(usize, total_elements, dim_usize) catch return error.UnexpectedEof;
    }
    return switch (tensor.ggml_type) {
        0 => std.math.mul(usize, total_elements, @sizeOf(f32)) catch return error.UnexpectedEof,
        1 => std.math.mul(usize, total_elements, @sizeOf(f16)) catch return error.UnexpectedEof,
        6 => { // Q5_0
            if (total_elements % quant.QK5_0 != 0) return error.UnexpectedEof;
            const blocks = total_elements / quant.QK5_0;
            return std.math.mul(usize, blocks, @sizeOf(quant.BlockQ5_0)) catch return error.UnexpectedEof;
        },
        8 => { // Q8_0
            if (total_elements % quant.QK8_0 != 0) return error.UnexpectedEof;
            const blocks = total_elements / quant.QK8_0;
            return std.math.mul(usize, blocks, @sizeOf(quant.BlockQ8_0)) catch return error.UnexpectedEof;
        },
        12 => { // Q4_K
            if (total_elements % quant.QK_K != 0) return error.UnexpectedEof;
            const blocks = total_elements / quant.QK_K;
            return std.math.mul(usize, blocks, @sizeOf(quant.BlockQ4_K)) catch return error.UnexpectedEof;
        },
        14 => { // Q6_K
            if (total_elements % quant.QK_K != 0) return error.UnexpectedEof;
            const blocks = total_elements / quant.QK_K;
            return std.math.mul(usize, blocks, @sizeOf(quant.BlockQ6_K)) catch return error.UnexpectedEof;
        },
        else => error.UnsupportedGgmlType,
    };
}

pub const GgufFile = struct {
    data: []const u8,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    alignment: usize,
    data_offset: usize,
    metadata: []MetadataEntry,
    tensors: []TensorInfo,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *GgufFile) void {
        if (self.metadata.len > 0) {
            self.allocator.free(self.metadata);
            self.metadata = &.{};
        }
        if (self.tensors.len > 0) {
            self.allocator.free(self.tensors);
            self.tensors = &.{};
        }
    }

    /// Finds a metadata entry by exact key name.
    pub fn findMetadata(self: GgufFile, key: []const u8) ?MetadataEntry {
        for (self.metadata) |entry| {
            if (std.mem.eql(u8, entry.key, key)) {
                return entry;
            }
        }
        return null;
    }

    /// Reads a uint32 metadata value.
    pub fn getMetadataU32(self: GgufFile, key: []const u8) ?u32 {
        const entry = self.findMetadata(key) orelse return null;
        if (entry.value_type != .uint32) return null;
        if (entry.offset + 4 > self.data.len) return null;
        return std.mem.readInt(u32, self.data[entry.offset..][0..4], .little);
    }

    /// Reads a float32 metadata value.
    pub fn getMetadataF32(self: GgufFile, key: []const u8) ?f32 {
        const entry = self.findMetadata(key) orelse return null;
        if (entry.value_type != .float32) return null;
        if (entry.offset + 4 > self.data.len) return null;
        const bits = std.mem.readInt(u32, self.data[entry.offset..][0..4], .little);
        return @bitCast(bits);
    }

    /// Reads a boolean metadata value.
    pub fn getMetadataBool(self: GgufFile, key: []const u8) ?bool {
        const entry = self.findMetadata(key) orelse return null;
        if (entry.value_type != .bool_val) return null;
        if (entry.offset + 1 > self.data.len) return null;
        return self.data[entry.offset] != 0;
    }

    /// Reads a string metadata value as a zero-copy slice.
    pub fn getMetadataString(self: GgufFile, key: []const u8) ?[]const u8 {
        const entry = self.findMetadata(key) orelse return null;
        if (entry.value_type != .string) return null;
        if (entry.offset + 8 > self.data.len) return null;
        const str_len: usize = @intCast(std.mem.readInt(u64, self.data[entry.offset..][0..8], .little));
        const str_start = entry.offset + 8;
        if (str_start + str_len > self.data.len) return null;
        return self.data[str_start .. str_start + str_len];
    }

    /// Finds a tensor descriptor by exact tensor name.
    pub fn findTensor(self: GgufFile, name: []const u8) ?TensorInfo {
        for (self.tensors) |t| {
            if (std.mem.eql(u8, t.name, name)) {
                return t;
            }
        }
        return null;
    }

    /// Returns the raw byte slice for a given tensor.
    pub fn getTensorSlice(self: GgufFile, tensor: TensorInfo) []const u8 {
        const offset_usize = std.math.cast(usize, tensor.offset) orelse return &[_]u8{};
        const start = std.math.add(usize, self.data_offset, offset_usize) catch return &[_]u8{};
        if (start >= self.data.len) return &[_]u8{};
        return self.data[start..];
    }
};

/// Helper to read a GGUF string at a cursor offset, advancing cursor.
fn readGgufString(data: []const u8, cursor: *usize) ![]const u8 {
    if (cursor.* + 8 > data.len) return error.UnexpectedEof;
    const len: usize = @intCast(std.mem.readInt(u64, data[cursor.*..][0..8], .little));
    cursor.* += 8;
    if (cursor.* + len > data.len) return error.UnexpectedEof;
    const s = data[cursor.* .. cursor.* + len];
    cursor.* += len;
    return s;
}

/// Skips over an array payload in data, advancing cursor.
fn skipArray(data: []const u8, cursor: *usize) !void {
    if (cursor.* + 12 > data.len) return error.UnexpectedEof;
    const sub_type_int = std.mem.readInt(u32, data[cursor.*..][0..4], .little);
    const sub_type: MetadataValueType = std.enums.fromInt(MetadataValueType, sub_type_int) orelse return error.MalformedArray;
    cursor.* += 4;
    const arr_len: usize = @intCast(std.mem.readInt(u64, data[cursor.*..][0..8], .little));
    cursor.* += 8;

    switch (sub_type) {
        .uint8, .int8, .bool_val => {
            if (cursor.* + arr_len > data.len) return error.UnexpectedEof;
            cursor.* += arr_len;
        },
        .uint16, .int16 => {
            const bytes = std.math.mul(usize, arr_len, 2) catch return error.UnexpectedEof;
            if (cursor.* + bytes > data.len) return error.UnexpectedEof;
            cursor.* += bytes;
        },
        .uint32, .int32, .float32 => {
            const bytes = std.math.mul(usize, arr_len, 4) catch return error.UnexpectedEof;
            if (cursor.* + bytes > data.len) return error.UnexpectedEof;
            cursor.* += bytes;
        },
        .uint64, .int64, .float64 => {
            const bytes = std.math.mul(usize, arr_len, 8) catch return error.UnexpectedEof;
            if (cursor.* + bytes > data.len) return error.UnexpectedEof;
            cursor.* += bytes;
        },
        .string => {
            for (0..arr_len) |_| {
                _ = try readGgufString(data, cursor);
            }
        },
        .array => {
            for (0..arr_len) |_| {
                try skipArray(data, cursor);
            }
        },
    }
}

/// Parses a GGUF file from memory.
pub fn parseGguf(allocator: std.mem.Allocator, data: []const u8) !GgufFile {
    if (data.len < 24) return error.UnexpectedEof;

    // Check magic
    if (!std.mem.eql(u8, data[0..4], &GGUF_MAGIC)) {
        return error.InvalidMagic;
    }

    const version = std.mem.readInt(u32, data[4..8], .little);
    if (version != 2 and version != 3) {
        return error.UnsupportedVersion;
    }

    const tensor_count = std.mem.readInt(u64, data[8..16], .little);
    const metadata_kv_count = std.mem.readInt(u64, data[16..24], .little);

    var cursor: usize = 24;

    // Allocate metadata table
    const metadata = try allocator.alloc(MetadataEntry, @intCast(metadata_kv_count));
    errdefer allocator.free(metadata);

    var default_alignment: usize = 32;

    // Parse metadata key-values
    for (0..@intCast(metadata_kv_count)) |i| {
        const key = try readGgufString(data, &cursor);
        if (cursor + 4 > data.len) return error.UnexpectedEof;
        const vtype_raw = std.mem.readInt(u32, data[cursor..][0..4], .little);
        cursor += 4;
        const vtype: MetadataValueType = std.enums.fromInt(MetadataValueType, vtype_raw) orelse return error.MalformedString;

        const val_start = cursor;
        metadata[i] = .{
            .key = key,
            .value_type = vtype,
            .offset = val_start,
        };

        // Advance cursor past value
        switch (vtype) {
            .uint8, .int8, .bool_val => cursor += 1,
            .uint16, .int16 => cursor += 2,
            .uint32, .int32, .float32 => cursor += 4,
            .uint64, .int64, .float64 => cursor += 8,
            .string => _ = try readGgufString(data, &cursor),
            .array => try skipArray(data, &cursor),
        }

        // Check if this metadata defines alignment
        if (std.mem.eql(u8, key, "general.alignment") and vtype == .uint32) {
            default_alignment = @intCast(std.mem.readInt(u32, data[val_start..][0..4], .little));
        }
    }

    // Allocate tensor table
    const tensors = try allocator.alloc(TensorInfo, @intCast(tensor_count));
    errdefer allocator.free(tensors);

    // Parse tensor info table
    for (0..@intCast(tensor_count)) |i| {
        const name = try readGgufString(data, &cursor);
        if (cursor + 4 > data.len) return error.UnexpectedEof;
        const n_dims = std.mem.readInt(u32, data[cursor..][0..4], .little);
        cursor += 4;

        if (n_dims > 4) return error.MalformedArray;
        var dims: [4]u64 = [_]u64{ 1, 1, 1, 1 };
        for (0..n_dims) |d| {
            if (cursor + 8 > data.len) return error.UnexpectedEof;
            dims[d] = std.mem.readInt(u64, data[cursor..][0..8], .little);
            cursor += 8;
        }

        if (cursor + 12 > data.len) return error.UnexpectedEof;
        const ggml_type = std.mem.readInt(u32, data[cursor..][0..4], .little);
        cursor += 4;
        const offset = std.mem.readInt(u64, data[cursor..][0..8], .little);
        cursor += 8;

        tensors[i] = .{
            .name = name,
            .n_dimensions = n_dims,
            .dimensions = dims,
            .ggml_type = ggml_type,
            .offset = offset,
        };
    }

    // Compute data offset aligned to general.alignment
    const data_offset = ((cursor + default_alignment - 1) / default_alignment) * default_alignment;
    if (tensor_count > 0 and data_offset > data.len) return error.UnexpectedEof;

    for (tensors) |t| {
        const offset_usize = std.math.cast(usize, t.offset) orelse return error.UnexpectedEof;
        const start = std.math.add(usize, data_offset, offset_usize) catch return error.UnexpectedEof;
        const tensor_bytes = try getTensorByteSize(t);
        const end = std.math.add(usize, start, tensor_bytes) catch return error.UnexpectedEof;
        if (end > data.len) return error.UnexpectedEof;
    }

    return GgufFile{
        .data = data,
        .version = version,
        .tensor_count = tensor_count,
        .metadata_kv_count = metadata_kv_count,
        .alignment = default_alignment,
        .data_offset = data_offset,
        .metadata = metadata,
        .tensors = tensors,
        .allocator = allocator,
    };
}

test "parseGguf: synthetic buffer validation" {
    var buf: [256]u8 = undefined;
    var cursor: usize = 0;

    // Magic
    @memcpy(buf[cursor .. cursor + 4], "GGUF");
    cursor += 4;

    // Version (3)
    std.mem.writeInt(u32, buf[cursor..][0..4], 3, .little);
    cursor += 4;

    // Tensor count (0)
    std.mem.writeInt(u64, buf[cursor..][0..8], 0, .little);
    cursor += 8;

    // Metadata count (1)
    std.mem.writeInt(u64, buf[cursor..][0..8], 1, .little);
    cursor += 8;

    // Key: "test_key" (len 8)
    std.mem.writeInt(u64, buf[cursor..][0..8], 8, .little);
    cursor += 8;
    @memcpy(buf[cursor .. cursor + 8], "test_key");
    cursor += 8;

    // Type: uint32 (4)
    std.mem.writeInt(u32, buf[cursor..][0..4], 4, .little);
    cursor += 4;

    // Value: 1234
    std.mem.writeInt(u32, buf[cursor..][0..4], 1234, .little);
    cursor += 4;

    var gguf = try parseGguf(std.testing.allocator, buf[0..cursor]);
    defer gguf.deinit();

    try std.testing.expectEqual(@as(u32, 3), gguf.version);
    try std.testing.expectEqual(@as(u64, 0), gguf.tensor_count);
    try std.testing.expectEqual(@as(u64, 1), gguf.metadata_kv_count);

    const val = gguf.getMetadataU32("test_key");
    try std.testing.expect(val != null);
    try std.testing.expectEqual(@as(u32, 1234), val.?);
}

test "parseGguf: truncated buffer and tensor out-of-bounds returns error.UnexpectedEof" {
    var buf: [256]u8 = undefined;
    var cursor: usize = 0;

    @memcpy(buf[cursor .. cursor + 4], "GGUF");
    cursor += 4;
    std.mem.writeInt(u32, buf[cursor..][0..4], 3, .little);
    cursor += 4;
    // Tensor count (1)
    std.mem.writeInt(u64, buf[cursor..][0..8], 1, .little);
    cursor += 8;
    // Metadata count (0)
    std.mem.writeInt(u64, buf[cursor..][0..8], 0, .little);
    cursor += 8;

    // Tensor name: "t0" (len 2)
    std.mem.writeInt(u64, buf[cursor..][0..8], 2, .little);
    cursor += 8;
    @memcpy(buf[cursor .. cursor + 2], "t0");
    cursor += 2;
    // n_dims: 1
    std.mem.writeInt(u32, buf[cursor..][0..4], 1, .little);
    cursor += 4;
    // dim 0: 16
    std.mem.writeInt(u64, buf[cursor..][0..8], 16, .little);
    cursor += 8;
    // ggml_type: 0 (F32)
    std.mem.writeInt(u32, buf[cursor..][0..4], 0, .little);
    cursor += 4;
    // offset: 100000 (far beyond buf)
    std.mem.writeInt(u64, buf[cursor..][0..8], 100000, .little);
    cursor += 8;

    try std.testing.expectError(error.UnexpectedEof, parseGguf(std.testing.allocator, buf[0..cursor]));
}

test "parseGguf: valid tensor start with truncated payload returns error.UnexpectedEof" {
    var buf: [256]u8 = undefined;
    var cursor: usize = 0;

    @memcpy(buf[cursor .. cursor + 4], "GGUF");
    cursor += 4;
    std.mem.writeInt(u32, buf[cursor..][0..4], 3, .little);
    cursor += 4;
    // Tensor count (1)
    std.mem.writeInt(u64, buf[cursor..][0..8], 1, .little);
    cursor += 8;
    // Metadata count (0)
    std.mem.writeInt(u64, buf[cursor..][0..8], 0, .little);
    cursor += 8;

    // Tensor name: "t0" (len 2)
    std.mem.writeInt(u64, buf[cursor..][0..8], 2, .little);
    cursor += 8;
    @memcpy(buf[cursor .. cursor + 2], "t0");
    cursor += 2;
    // n_dims: 1
    std.mem.writeInt(u32, buf[cursor..][0..4], 1, .little);
    cursor += 4;
    // dim 0: 16 (requires 16 * 4 = 64 bytes for F32)
    std.mem.writeInt(u64, buf[cursor..][0..8], 16, .little);
    cursor += 8;
    // ggml_type: 0 (F32)
    std.mem.writeInt(u32, buf[cursor..][0..4], 0, .little);
    cursor += 4;
    // offset: 0 (starts exactly at data_offset inside buf)
    std.mem.writeInt(u64, buf[cursor..][0..8], 0, .little);
    cursor += 8;

    // Pad only 8 bytes after alignment boundary, so start <= buf.len, but start + 64 > buf.len
    const alignment: usize = 32;
    const data_offset = ((cursor + alignment - 1) / alignment) * alignment;
    const truncated_total = data_offset + 8;
    @memset(buf[0..truncated_total], 0);

    // Re-fill header
    cursor = 0;
    @memcpy(buf[cursor .. cursor + 4], "GGUF");
    cursor += 4;
    std.mem.writeInt(u32, buf[cursor..][0..4], 3, .little);
    cursor += 4;
    std.mem.writeInt(u64, buf[cursor..][0..8], 1, .little);
    cursor += 8;
    std.mem.writeInt(u64, buf[cursor..][0..8], 0, .little);
    cursor += 8;
    std.mem.writeInt(u64, buf[cursor..][0..8], 2, .little);
    cursor += 8;
    @memcpy(buf[cursor .. cursor + 2], "t0");
    cursor += 2;
    std.mem.writeInt(u32, buf[cursor..][0..4], 1, .little);
    cursor += 4;
    std.mem.writeInt(u64, buf[cursor..][0..8], 16, .little);
    cursor += 8;
    std.mem.writeInt(u32, buf[cursor..][0..4], 0, .little);
    cursor += 4;
    std.mem.writeInt(u64, buf[cursor..][0..8], 0, .little);
    cursor += 8;

    try std.testing.expectError(error.UnexpectedEof, parseGguf(std.testing.allocator, buf[0..truncated_total]));
}

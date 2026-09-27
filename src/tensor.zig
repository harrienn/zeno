//! Zeno — CPU Tensor representation and memory management.
//!
//! Provides explicit row-major 4D tensors, non-allocating slicing, and allocator tracking.

const std = @import("std");

/// Data types supported across the runtime.
pub const DType = enum {
    f32,
    f16,
    q4_k,
    q6_k,
    q8_0,
};

/// Contiguous row-major CPU float tensor of rank up to 4.
pub const Tensor = struct {
    /// Dimension lengths [d0, d1, d2, d3]. Unused leading dimensions are set to 1.
    shape: [4]usize,
    /// Strides per dimension in elements.
    strides: [4]usize,
    /// Contiguous element buffer.
    data: []f32,
    /// Optional owning allocator. If null, data is an external or transient slice.
    allocator: ?std.mem.Allocator = null,

    /// Computes total number of elements in the tensor.
    pub fn numElements(shape: [4]usize) usize {
        return shape[0] * shape[1] * shape[2] * shape[3];
    }

    /// Computes row-major strides for a given 4D shape.
    pub fn computeStrides(shape: [4]usize) [4]usize {
        return [4]usize{
            shape[1] * shape[2] * shape[3],
            shape[2] * shape[3],
            shape[3],
            1,
        };
    }

    /// Allocates and initializes a new 4D tensor.
    pub fn init(allocator: std.mem.Allocator, shape: [4]usize) !Tensor {
        const total = numElements(shape);
        const data = try allocator.alloc(f32, total);
        @memset(data, 0.0);
        return Tensor{
            .shape = shape,
            .strides = computeStrides(shape),
            .data = data,
            .allocator = allocator,
        };
    }

    /// Allocates and initializes a 2D tensor [rows, cols].
    pub fn init2D(allocator: std.mem.Allocator, rows: usize, cols: usize) !Tensor {
        return init(allocator, [4]usize{ 1, 1, rows, cols });
    }

    /// Allocates and initializes a 1D tensor [len].
    pub fn init1D(allocator: std.mem.Allocator, len: usize) !Tensor {
        return init(allocator, [4]usize{ 1, 1, 1, len });
    }

    /// Creates a non-owning tensor view wrapping an existing slice.
    pub fn fromSlice(data: []f32, shape: [4]usize) Tensor {
        std.debug.assert(data.len == numElements(shape));
        return Tensor{
            .shape = shape,
            .strides = computeStrides(shape),
            .data = data,
            .allocator = null,
        };
    }

    /// Releases allocated memory if an allocator was provided.
    pub fn deinit(self: *Tensor) void {
        if (self.allocator) |alloc| {
            alloc.free(self.data);
            self.data = &.{};
        }
    }

    /// Returns a slice to a specific row in a 2D tensor [1, 1, rows, cols] or [rows, cols].
    pub fn rowSlice(self: Tensor, row: usize) []f32 {
        const cols = self.shape[3];
        const rows = self.shape[2];
        std.debug.assert(row < rows);
        const start = row * cols;
        return self.data[start .. start + cols];
    }

    /// Sets all elements to a scalar value.
    pub fn fill(self: *Tensor, val: f32) void {
        @memset(self.data, val);
    }

    /// Copies data from a source slice of equal length.
    pub fn copyFrom(self: *Tensor, src: []const f32) void {
        std.debug.assert(self.data.len == src.len);
        @memcpy(self.data, src);
    }

    /// Element access for 1D tensors.
    pub inline fn at1D(self: Tensor, idx: usize) f32 {
        return self.data[idx];
    }

    /// Element mutation for 1D tensors.
    pub inline fn set1D(self: *Tensor, idx: usize, val: f32) void {
        self.data[idx] = val;
    }

    /// Element access for 2D tensors [rows, cols].
    pub inline fn at2D(self: Tensor, r: usize, c: usize) f32 {
        return self.data[r * self.shape[3] + c];
    }

    /// Element mutation for 2D tensors [rows, cols].
    pub inline fn set2D(self: *Tensor, r: usize, c: usize, val: f32) void {
        self.data[r * self.shape[3] + c] = val;
    }
};

test "Tensor: init, access, and deinit" {
    const allocator = std.testing.allocator;
    var t = try Tensor.init2D(allocator, 4, 8);
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 32), t.data.len);
    try std.testing.expectEqual(@as(f32, 0.0), t.at2D(0, 0));

    t.set2D(2, 5, 42.0);
    try std.testing.expectEqual(@as(f32, 42.0), t.at2D(2, 5));

    const row2 = t.rowSlice(2);
    try std.testing.expectEqual(@as(usize, 8), row2.len);
    try std.testing.expectEqual(@as(f32, 42.0), row2[5]);
}

test "Tensor: non-owning view fromSlice" {
    var buffer: [16]f32 = undefined;
    @memset(&buffer, 1.5);

    const t = Tensor.fromSlice(&buffer, [4]usize{ 1, 1, 4, 4 });
    try std.testing.expectEqual(@as(usize, 16), t.data.len);
    try std.testing.expectEqual(@as(f32, 1.5), t.at2D(1, 1));
}

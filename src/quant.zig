//! Zeno — GGUF Quantization block formats and reference CPU dequantization.
//!
//! Faithful implementation of GGML / llama.cpp block structures (Q4_K, Q6_K, Q8_0).

const std = @import("std");

/// Number of values in a K-quant super-block.
pub const QK_K: usize = 256;

/// Number of values in a Q8_0 block.
pub const QK8_0: usize = 32;

/// Q4_K block format: 256 4-bit values per block (144 bytes).
pub const BlockQ4_K = extern struct {
    d: f16,
    dmin: f16,
    scales: [12]u8,
    qs: [QK_K / 2]u8,
};

/// Q6_K block format: 256 6-bit values per block (210 bytes).
pub const BlockQ6_K = extern struct {
    ql: [QK_K / 2]u8,
    qh: [QK_K / 4]u8,
    scales: [QK_K / 16]i8,
    d: f16,
};

/// Q8_0 block format: 32 8-bit values per block (34 bytes).
pub const BlockQ8_0 = extern struct {
    d: f16,
    qs: [QK8_0]i8,
};

// Compile-time byte size assertions matching GGML binary layout.
comptime {
    std.debug.assert(@sizeOf(BlockQ4_K) == 144);
    std.debug.assert(@sizeOf(BlockQ6_K) == 210);
    std.debug.assert(@sizeOf(BlockQ8_0) == 34);
}

/// Helper extracting scale and min factors from Q4_K 12-byte packed scale array.
pub inline fn getScaleMinK4(j: usize, q: *const [12]u8, sc: *u8, m: *u8) void {
    if (j < 4) {
        sc.* = q[j] & 63;
        m.* = q[j + 4] & 63;
    } else {
        sc.* = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m.* = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

/// Dequantizes a row of Q4_K blocks into an f32 destination buffer.
pub fn dequantizeRowQ4_K(blocks: []const BlockQ4_K, dst: []f32) void {
    std.debug.assert(dst.len >= blocks.len * QK_K);

    var out_idx: usize = 0;
    for (blocks) |block| {
        const d: f32 = @floatCast(block.d);
        const min: f32 = @floatCast(block.dmin);

        var is: usize = 0;
        var q_idx: usize = 0;
        var j: usize = 0;
        while (j < QK_K) : (j += 64) {
            var sc: u8 = undefined;
            var m: u8 = undefined;

            getScaleMinK4(is + 0, &block.scales, &sc, &m);
            const d1 = d * @as(f32, @floatFromInt(sc));
            const m1 = min * @as(f32, @floatFromInt(m));

            getScaleMinK4(is + 1, &block.scales, &sc, &m);
            const d2 = d * @as(f32, @floatFromInt(sc));
            const m2 = min * @as(f32, @floatFromInt(m));

            const q_slice = block.qs[q_idx .. q_idx + 32];
            for (0..32) |l| {
                dst[out_idx + l] = d1 * @as(f32, @floatFromInt(q_slice[l] & 0x0F)) - m1;
            }
            for (0..32) |l| {
                dst[out_idx + 32 + l] = d2 * @as(f32, @floatFromInt(q_slice[l] >> 4)) - m2;
            }

            out_idx += 64;
            q_idx += 32;
            is += 2;
        }
    }
}

/// Dequantizes a row of Q6_K blocks into an f32 destination buffer.
pub fn dequantizeRowQ6_K(blocks: []const BlockQ6_K, dst: []f32) void {
    std.debug.assert(dst.len >= blocks.len * QK_K);

    for (blocks, 0..) |block, i| {
        const d: f32 = @floatCast(block.d);
        var ql_idx: usize = 0;
        var qh_idx: usize = 0;
        var sc_idx: usize = 0;
        var out_idx: usize = i * QK_K;

        var n: usize = 0;
        while (n < QK_K) : (n += 128) {
            const ql = block.ql[ql_idx .. ql_idx + 64];
            const qh = block.qh[qh_idx .. qh_idx + 32];
            const sc = block.scales[sc_idx .. sc_idx + 8];

            for (0..32) |l| {
                const is = l / 16;
                const q1: i32 = @as(i32, @intCast((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4))) - 32;
                const q2: i32 = @as(i32, @intCast((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4))) - 32;
                const q3: i32 = @as(i32, @intCast((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4))) - 32;
                const q4: i32 = @as(i32, @intCast((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4))) - 32;

                dst[out_idx + l + 0] = d * @as(f32, @floatFromInt(sc[is + 0])) * @as(f32, @floatFromInt(q1));
                dst[out_idx + l + 32] = d * @as(f32, @floatFromInt(sc[is + 2])) * @as(f32, @floatFromInt(q2));
                dst[out_idx + l + 64] = d * @as(f32, @floatFromInt(sc[is + 4])) * @as(f32, @floatFromInt(q3));
                dst[out_idx + l + 96] = d * @as(f32, @floatFromInt(sc[is + 6])) * @as(f32, @floatFromInt(q4));
            }

            out_idx += 128;
            ql_idx += 64;
            qh_idx += 32;
            sc_idx += 8;
        }
    }
}

/// Dequantizes a row of Q8_0 blocks into an f32 destination buffer.
pub fn dequantizeRowQ8_0(blocks: []const BlockQ8_0, dst: []f32) void {
    std.debug.assert(dst.len >= blocks.len * QK8_0);

    for (blocks, 0..) |block, i| {
        const d: f32 = @floatCast(block.d);
        const out_start = i * QK8_0;
        for (0..QK8_0) |j| {
            dst[out_start + j] = @as(f32, @floatFromInt(block.qs[j])) * d;
        }
    }
}

/// Computes dot product between a row of Q6_K blocks and an f32 vector x of equal element length.
pub fn dotRowQ6_K(blocks: []const BlockQ6_K, x: []const f32) f32 {
    const total_elements = blocks.len * QK_K;
    std.debug.assert(x.len == total_elements);

    var sum: f32 = 0.0;
    for (blocks, 0..) |block, i| {
        const d: f32 = @floatCast(block.d);
        const x_block = x[i * QK_K .. (i + 1) * QK_K];

        var ql_idx: usize = 0;
        var qh_idx: usize = 0;
        var sc_idx: usize = 0;
        var x_idx: usize = 0;

        var n: usize = 0;
        while (n < QK_K) : (n += 128) {
            const ql = block.ql[ql_idx .. ql_idx + 64];
            const qh = block.qh[qh_idx .. qh_idx + 32];
            const sc = block.scales[sc_idx .. sc_idx + 8];

            for (0..32) |l| {
                const is = l / 16;
                const q1: i32 = @as(i32, @intCast((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4))) - 32;
                const q2: i32 = @as(i32, @intCast((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4))) - 32;
                const q3: i32 = @as(i32, @intCast((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4))) - 32;
                const q4: i32 = @as(i32, @intCast((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4))) - 32;

                const v1 = d * @as(f32, @floatFromInt(sc[is + 0])) * @as(f32, @floatFromInt(q1));
                const v2 = d * @as(f32, @floatFromInt(sc[is + 2])) * @as(f32, @floatFromInt(q2));
                const v3 = d * @as(f32, @floatFromInt(sc[is + 4])) * @as(f32, @floatFromInt(q3));
                const v4 = d * @as(f32, @floatFromInt(sc[is + 6])) * @as(f32, @floatFromInt(q4));

                sum += v1 * x_block[x_idx + l + 0];
                sum += v2 * x_block[x_idx + l + 32];
                sum += v3 * x_block[x_idx + l + 64];
                sum += v4 * x_block[x_idx + l + 96];
            }

            x_idx += 128;
            ql_idx += 64;
            qh_idx += 32;
            sc_idx += 8;
        }
    }
    return sum;
}

/// Computes dot product between a row of Q4_K blocks and an f32 vector x of equal element length.
pub fn dotRowQ4_K(blocks: []const BlockQ4_K, x: []const f32) f32 {
    const total_elements = blocks.len * QK_K;
    std.debug.assert(x.len == total_elements);

    var sum: f32 = 0.0;
    var x_idx: usize = 0;
    for (blocks) |block| {
        const d: f32 = @floatCast(block.d);
        const min: f32 = @floatCast(block.dmin);

        var is: usize = 0;
        var q_idx: usize = 0;
        var j: usize = 0;
        while (j < QK_K) : (j += 64) {
            var sc: u8 = undefined;
            var m: u8 = undefined;

            getScaleMinK4(is + 0, &block.scales, &sc, &m);
            const d1 = d * @as(f32, @floatFromInt(sc));
            const m1 = min * @as(f32, @floatFromInt(m));

            getScaleMinK4(is + 1, &block.scales, &sc, &m);
            const d2 = d * @as(f32, @floatFromInt(sc));
            const m2 = min * @as(f32, @floatFromInt(m));

            const q_slice = block.qs[q_idx .. q_idx + 32];
            for (0..32) |l| {
                const val1 = d1 * @as(f32, @floatFromInt(q_slice[l] & 0x0F)) - m1;
                sum += val1 * x[x_idx + l];
            }
            for (0..32) |l| {
                const val2 = d2 * @as(f32, @floatFromInt(q_slice[l] >> 4)) - m2;
                sum += val2 * x[x_idx + 32 + l];
            }

            x_idx += 64;
            q_idx += 32;
            is += 2;
        }
    }
    return sum;
}

/// Computes dot product between a row of Q8_0 blocks and an f32 vector x of equal element length.
pub fn dotRowQ8_0(blocks: []const BlockQ8_0, x: []const f32) f32 {
    const total_elements = blocks.len * QK8_0;
    std.debug.assert(x.len == total_elements);

    var sum: f32 = 0.0;
    for (blocks, 0..) |block, i| {
        const d: f32 = @floatCast(block.d);
        const x_start = i * QK8_0;
        for (0..QK8_0) |j| {
            sum += (@as(f32, @floatFromInt(block.qs[j])) * d) * x[x_start + j];
        }
    }
    return sum;
}

test "Block sizes match GGML specifications" {
    try std.testing.expectEqual(@as(usize, 144), @sizeOf(BlockQ4_K));
    try std.testing.expectEqual(@as(usize, 210), @sizeOf(BlockQ6_K));
    try std.testing.expectEqual(@as(usize, 34), @sizeOf(BlockQ8_0));
}

test "dequantizeRowQ8_0: basic correctness" {
    var block = BlockQ8_0{
        .d = @floatCast(0.5),
        .qs = @splat(0),
    };
    block.qs[0] = 2;
    block.qs[1] = -4;
    block.qs[31] = 10;

    var dst: [32]f32 = undefined;
    const blocks = [_]BlockQ8_0{block};
    dequantizeRowQ8_0(&blocks, &dst);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dst[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), dst[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), dst[31], 1e-5);
}

test "dequantizeRowQ4_K: zero block yields zero or expected min" {
    const block = BlockQ4_K{
        .d = @floatCast(1.0),
        .dmin = @floatCast(0.0),
        .scales = @splat(0),
        .qs = @splat(0),
    };
    var dst: [256]f32 = undefined;
    const blocks = [_]BlockQ4_K{block};
    dequantizeRowQ4_K(&blocks, &dst);

    for (dst) |val| {
        try std.testing.expectEqual(@as(f32, 0.0), val);
    }
}

test "dequantizeRowQ6_K and dotRowQ6_K equivalence" {
    const block = BlockQ6_K{
        .ql = @splat(0x21),
        .qh = @splat(0x00),
        .scales = @splat(2),
        .d = @floatCast(0.25),
    };
    var dst: [256]f32 = undefined;
    const blocks = [_]BlockQ6_K{block};
    dequantizeRowQ6_K(&blocks, &dst);

    var x: [256]f32 = undefined;
    for (&x, 0..) |*val, idx| {
        val.* = if (idx % 2 == 0) 1.0 else -1.0;
    }

    var expected_dot: f32 = 0.0;
    for (dst, x) |d_val, x_val| {
        expected_dot += d_val * x_val;
    }

    const computed_dot = dotRowQ6_K(&blocks, &x);
    try std.testing.expectApproxEqAbs(expected_dot, computed_dot, 1e-4);
}

test "dequantizeRowQ4_K and dotRowQ4_K equivalence" {
    const block = BlockQ4_K{
        .d = @floatCast(0.5),
        .dmin = @floatCast(0.1),
        .scales = @splat(3),
        .qs = @splat(0x57),
    };
    var dst: [256]f32 = undefined;
    const blocks = [_]BlockQ4_K{block};
    dequantizeRowQ4_K(&blocks, &dst);

    var x: [256]f32 = undefined;
    for (&x, 0..) |*val, idx| {
        val.* = if (idx % 3 == 0) 1.5 else -0.5;
    }

    var expected_dot: f32 = 0.0;
    for (dst, x) |d_val, x_val| {
        expected_dot += d_val * x_val;
    }

    const computed_dot = dotRowQ4_K(&blocks, &x);
    try std.testing.expectApproxEqAbs(expected_dot, computed_dot, 1e-3);
}

test "dequantizeRowQ8_0 and dotRowQ8_0 equivalence" {
    var block = BlockQ8_0{
        .d = @floatCast(0.25),
        .qs = @splat(0),
    };
    block.qs[0] = 5;
    block.qs[1] = -10;
    block.qs[31] = 20;

    var dst: [32]f32 = undefined;
    const blocks = [_]BlockQ8_0{block};
    dequantizeRowQ8_0(&blocks, &dst);

    var x: [32]f32 = undefined;
    for (&x, 0..) |*val, idx| {
        val.* = @as(f32, @floatFromInt(idx)) * 0.1;
    }

    var expected_dot: f32 = 0.0;
    for (dst, x) |d_val, x_val| {
        expected_dot += d_val * x_val;
    }

    const computed_dot = dotRowQ8_0(&blocks, &x);
    try std.testing.expectApproxEqAbs(expected_dot, computed_dot, 1e-4);
}


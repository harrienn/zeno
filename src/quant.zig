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

/// Number of values in a Q5_0 block.
pub const QK5_0: usize = 32;

/// Q5_0 block format: 32 5-bit values per block (22 bytes).
pub const BlockQ5_0 = extern struct {
    d: f16,
    qh: [4]u8,
    qs: [QK5_0 / 2]u8,
};

// Compile-time byte size assertions matching GGML binary layout.
comptime {
    std.debug.assert(@sizeOf(BlockQ4_K) == 144);
    std.debug.assert(@sizeOf(BlockQ6_K) == 210);
    std.debug.assert(@sizeOf(BlockQ8_0) == 34);
    std.debug.assert(@sizeOf(BlockQ5_0) == 22);
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

inline fn loadVec8(slice: []const f32, offset: usize) @Vector(8, f32) {
    const arr: [8]f32 = slice[offset..][0..8].*;
    return arr;
}

/// Precomputes sum of elements in each 32-element group of an f32 vector x.
pub fn computeXSums32(x: []const f32, out_sums: []f32) void {
    const num_groups = x.len / 32;
    std.debug.assert(out_sums.len >= num_groups);

    for (0..num_groups) |g| {
        const base = g * 32;
        const v0 = loadVec8(x, base);
        const v1 = loadVec8(x, base + 8);
        const v2 = loadVec8(x, base + 16);
        const v3 = loadVec8(x, base + 24);
        out_sums[g] = @reduce(.Add, v0 + v1 + v2 + v3);
    }
}

/// Computes dot product between a row of Q6_K blocks and an f32 vector x using SIMD operations.
pub fn dotRowQ6_K(blocks: []const BlockQ6_K, x: []const f32) f32 {
    const total_elements = blocks.len * QK_K;
    std.debug.assert(x.len == total_elements);
    const Vec8 = @Vector(8, f32);

    var total_acc: Vec8 = @splat(0.0);
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

            inline for (0..2) |is| {
                const base_l = is * 16;
                const sc0 = d * @as(f32, @floatFromInt(sc[is + 0]));
                const sc2 = d * @as(f32, @floatFromInt(sc[is + 2]));
                const sc4 = d * @as(f32, @floatFromInt(sc[is + 4]));
                const sc6 = d * @as(f32, @floatFromInt(sc[is + 6]));

                var acc1: Vec8 = @splat(0.0);
                var acc2: Vec8 = @splat(0.0);
                var acc3: Vec8 = @splat(0.0);
                var acc4: Vec8 = @splat(0.0);

                inline for (0..2) |sub| {
                    const l_off = base_l + sub * 8;
                    var q1_arr: [8]f32 = undefined;
                    var q2_arr: [8]f32 = undefined;
                    var q3_arr: [8]f32 = undefined;
                    var q4_arr: [8]f32 = undefined;

                    inline for (0..8) |k| {
                        const l = l_off + k;
                        const q1: i32 = @as(i32, @intCast((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4))) - 32;
                        const q2: i32 = @as(i32, @intCast((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4))) - 32;
                        const q3: i32 = @as(i32, @intCast((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4))) - 32;
                        const q4: i32 = @as(i32, @intCast((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4))) - 32;

                        q1_arr[k] = @floatFromInt(q1);
                        q2_arr[k] = @floatFromInt(q2);
                        q3_arr[k] = @floatFromInt(q3);
                        q4_arr[k] = @floatFromInt(q4);
                    }

                    const v_x1 = loadVec8(x_block, x_idx + l_off);
                    const v_x2 = loadVec8(x_block, x_idx + l_off + 32);
                    const v_x3 = loadVec8(x_block, x_idx + l_off + 64);
                    const v_x4 = loadVec8(x_block, x_idx + l_off + 96);

                    const v_q1: Vec8 = q1_arr;
                    const v_q2: Vec8 = q2_arr;
                    const v_q3: Vec8 = q3_arr;
                    const v_q4: Vec8 = q4_arr;

                    acc1 = @mulAdd(Vec8, v_q1, v_x1, acc1);
                    acc2 = @mulAdd(Vec8, v_q2, v_x2, acc2);
                    acc3 = @mulAdd(Vec8, v_q3, v_x3, acc3);
                    acc4 = @mulAdd(Vec8, v_q4, v_x4, acc4);
                }

                total_acc = @mulAdd(Vec8, acc1, @splat(sc0), total_acc);
                total_acc = @mulAdd(Vec8, acc2, @splat(sc2), total_acc);
                total_acc = @mulAdd(Vec8, acc3, @splat(sc4), total_acc);
                total_acc = @mulAdd(Vec8, acc4, @splat(sc6), total_acc);
            }

            x_idx += 128;
            ql_idx += 64;
            qh_idx += 32;
            sc_idx += 8;
        }
    }
    return @reduce(.Add, total_acc);
}

/// Computes dot product between a row of Q4_K blocks and an f32 vector x with precomputed 32-element x sums.
pub fn dotRowQ4_K_with_xsums(blocks: []const BlockQ4_K, x: []const f32, x_sums: []const f32) f32 {
    const total_elements = blocks.len * QK_K;
    std.debug.assert(x.len == total_elements);
    std.debug.assert(x_sums.len >= total_elements / 32);

    const Vec8 = @Vector(8, f32);
    var acc: Vec8 = @splat(0.0);
    var min_sum: f32 = 0.0;
    var x_idx: usize = 0;
    var group_idx: usize = 0;

    for (blocks) |block| {
        const d: f32 = @floatCast(block.d);
        const min: f32 = @floatCast(block.dmin);

        var is: usize = 0;
        var q_idx: usize = 0;
        var j: usize = 0;
        while (j < QK_K) : (j += 64) {
            var sc1: u8 = undefined;
            var m1: u8 = undefined;
            getScaleMinK4(is + 0, &block.scales, &sc1, &m1);
            const d1 = d * @as(f32, @floatFromInt(sc1));
            const m_val1 = min * @as(f32, @floatFromInt(m1));

            var sc2: u8 = undefined;
            var m2: u8 = undefined;
            getScaleMinK4(is + 1, &block.scales, &sc2, &m2);
            const d2 = d * @as(f32, @floatFromInt(sc2));
            const m_val2 = min * @as(f32, @floatFromInt(m2));

            const q_slice = block.qs[q_idx .. q_idx + 32];

            var v_q1_dot: Vec8 = @splat(0.0);
            var v_q2_dot: Vec8 = @splat(0.0);

            inline for (0..4) |sub| {
                const off = sub * 8;
                const raw: [8]u8 = q_slice[off..][0..8].*;
                const v_raw: @Vector(8, u8) = raw;

                const v_q1_u8 = v_raw & @as(@Vector(8, u8), @splat(0x0F));
                const v_q1_i32: @Vector(8, i32) = v_q1_u8;
                const v_q1: Vec8 = @floatFromInt(v_q1_i32);

                const v_q2_u8 = v_raw >> @as(@Vector(8, u3), @splat(4));
                const v_q2_i32: @Vector(8, i32) = v_q2_u8;
                const v_q2: Vec8 = @floatFromInt(v_q2_i32);

                const v_x1 = loadVec8(x, x_idx + off);
                const v_x2 = loadVec8(x, x_idx + 32 + off);

                v_q1_dot = @mulAdd(Vec8, v_q1, v_x1, v_q1_dot);
                v_q2_dot = @mulAdd(Vec8, v_q2, v_x2, v_q2_dot);
            }

            acc = @mulAdd(Vec8, v_q1_dot, @splat(d1), acc);
            acc = @mulAdd(Vec8, v_q2_dot, @splat(d2), acc);
            min_sum += m_val1 * x_sums[group_idx] + m_val2 * x_sums[group_idx + 1];

            x_idx += 64;
            q_idx += 32;
            is += 2;
            group_idx += 2;
        }
    }
    return @reduce(.Add, acc) - min_sum;
}

/// Computes dot product between a row of Q4_K blocks and an f32 vector x using SIMD operations.
pub fn dotRowQ4_K(blocks: []const BlockQ4_K, x: []const f32) f32 {
    const total_elements = blocks.len * QK_K;
    std.debug.assert(x.len == total_elements);

    var stack_sums: [256]f32 = undefined;
    const num_groups = total_elements / 32;
    if (num_groups <= stack_sums.len) {
        computeXSums32(x, stack_sums[0..num_groups]);
        return dotRowQ4_K_with_xsums(blocks, x, stack_sums[0..num_groups]);
    } else {
        const Vec8 = @Vector(8, f32);
        var sum: f32 = 0.0;
        var x_idx: usize = 0;
        for (blocks) |block| {
            const d: f32 = @floatCast(block.d);
            const min: f32 = @floatCast(block.dmin);

            var is: usize = 0;
            var q_idx: usize = 0;
            var j: usize = 0;
            while (j < QK_K) : (j += 64) {
                var sc1: u8 = undefined;
                var m1: u8 = undefined;
                getScaleMinK4(is + 0, &block.scales, &sc1, &m1);
                const d1 = d * @as(f32, @floatFromInt(sc1));
                const m_val1 = min * @as(f32, @floatFromInt(m1));

                var sc2: u8 = undefined;
                var m2: u8 = undefined;
                getScaleMinK4(is + 1, &block.scales, &sc2, &m2);
                const d2 = d * @as(f32, @floatFromInt(sc2));
                const m_val2 = min * @as(f32, @floatFromInt(m2));

                const q_slice = block.qs[q_idx .. q_idx + 32];
                var v_q1_dot: Vec8 = @splat(0.0);
                var v_x1_sum: Vec8 = @splat(0.0);
                var v_q2_dot: Vec8 = @splat(0.0);
                var v_x2_sum: Vec8 = @splat(0.0);

                inline for (0..4) |sub| {
                    const off = sub * 8;
                    const raw: [8]u8 = q_slice[off..][0..8].*;
                    const v_raw: @Vector(8, u8) = raw;

                    const v_q1_u8 = v_raw & @as(@Vector(8, u8), @splat(0x0F));
                    const v_q1_i32: @Vector(8, i32) = v_q1_u8;
                    const v_q1: Vec8 = @floatFromInt(v_q1_i32);

                    const v_q2_u8 = v_raw >> @as(@Vector(8, u3), @splat(4));
                    const v_q2_i32: @Vector(8, i32) = v_q2_u8;
                    const v_q2: Vec8 = @floatFromInt(v_q2_i32);

                    const v_x1 = loadVec8(x, x_idx + off);
                    const v_x2 = loadVec8(x, x_idx + 32 + off);

                    v_q1_dot = @mulAdd(Vec8, v_q1, v_x1, v_q1_dot);
                    v_x1_sum += v_x1;
                    v_q2_dot = @mulAdd(Vec8, v_q2, v_x2, v_q2_dot);
                    v_x2_sum += v_x2;
                }

                sum += (d1 * @reduce(.Add, v_q1_dot) - m_val1 * @reduce(.Add, v_x1_sum)) +
                    (d2 * @reduce(.Add, v_q2_dot) - m_val2 * @reduce(.Add, v_x2_sum));
                x_idx += 64;
                q_idx += 32;
                is += 2;
            }
        }
        return sum;
    }
}

/// Computes dot product between a row of Q8_0 blocks and an f32 vector x using SIMD operations.
pub fn dotRowQ8_0(blocks: []const BlockQ8_0, x: []const f32) f32 {
    const total_elements = blocks.len * QK8_0;
    std.debug.assert(x.len == total_elements);
    const Vec8 = @Vector(8, f32);

    var total_acc: Vec8 = @splat(0.0);
    for (blocks, 0..) |block, i| {
        const d: f32 = @floatCast(block.d);
        const v_d: Vec8 = @splat(d);
        const x_start = i * QK8_0;

        const raw0: [8]i8 = block.qs[0..8].*;
        const v_q0: Vec8 = @floatFromInt(@as(@Vector(8, i32), @as(@Vector(8, i8), raw0)));
        const v_x0 = loadVec8(x, x_start);

        const raw1: [8]i8 = block.qs[8..16].*;
        const v_q1: Vec8 = @floatFromInt(@as(@Vector(8, i32), @as(@Vector(8, i8), raw1)));
        const v_x1 = loadVec8(x, x_start + 8);

        const raw2: [8]i8 = block.qs[16..24].*;
        const v_q2: Vec8 = @floatFromInt(@as(@Vector(8, i32), @as(@Vector(8, i8), raw2)));
        const v_x2 = loadVec8(x, x_start + 16);

        const raw3: [8]i8 = block.qs[24..32].*;
        const v_q3: Vec8 = @floatFromInt(@as(@Vector(8, i32), @as(@Vector(8, i8), raw3)));
        const v_x3 = loadVec8(x, x_start + 24);

        const p0 = v_q0 * v_x0;
        const p1 = v_q1 * v_x1;
        const p2 = v_q2 * v_x2;
        const p3 = v_q3 * v_x3;

        const b_acc = (p0 + p1) + (p2 + p3);
        total_acc = @mulAdd(Vec8, b_acc, v_d, total_acc);
    }
    return @reduce(.Add, total_acc);
}

/// Dequantizes a row of Q5_0 blocks into an f32 destination buffer.
pub fn dequantizeRowQ5_0(blocks: []const BlockQ5_0, dst: []f32) void {
    std.debug.assert(dst.len >= blocks.len * QK5_0);

    for (blocks, 0..) |block, b_idx| {
        const d: f32 = @floatCast(block.d);
        const qh = std.mem.readInt(u32, &block.qh, .little);
        const base = b_idx * QK5_0;

        for (0..16) |j| {
            const shift0 = @as(u5, @intCast(j));
            const shift1 = @as(u5, @intCast(j + 12));
            const xh_0: u8 = @as(u8, @truncate((qh >> shift0) << 4)) & 0x10;
            const xh_1: u8 = @as(u8, @truncate(qh >> shift1)) & 0x10;

            const x0: i32 = @as(i32, (block.qs[j] & 0x0F) | xh_0) - 16;
            const x1: i32 = @as(i32, (block.qs[j] >> 4) | xh_1) - 16;

            dst[base + j] = @as(f32, @floatFromInt(x0)) * d;
            dst[base + j + 16] = @as(f32, @floatFromInt(x1)) * d;
        }
    }
}

/// Computes dot product between a row of Q5_0 blocks and an f32 vector x using SIMD operations.
pub fn dotRowQ5_0(blocks: []const BlockQ5_0, x: []const f32) f32 {
    const total_elements = blocks.len * QK5_0;
    std.debug.assert(x.len == total_elements);
    const Vec8 = @Vector(8, f32);

    var total_acc: Vec8 = @splat(0.0);
    for (blocks, 0..) |block, b_idx| {
        const d: f32 = @floatCast(block.d);
        const qh = std.mem.readInt(u32, &block.qh, .little);
        const base = b_idx * QK5_0;

        var acc: Vec8 = @splat(0.0);
        inline for (0..2) |chunk| {
            const j_off = chunk * 8;
            var x0_arr: [8]f32 = undefined;
            var x1_arr: [8]f32 = undefined;
            inline for (0..8) |k| {
                const j = j_off + k;
                const shift0 = @as(u5, @intCast(j));
                const shift1 = @as(u5, @intCast(j + 12));
                const xh_0: u8 = @as(u8, @truncate((qh >> shift0) << 4)) & 0x10;
                const xh_1: u8 = @as(u8, @truncate(qh >> shift1)) & 0x10;

                const x0: i32 = @as(i32, (block.qs[j] & 0x0F) | xh_0) - 16;
                const x1: i32 = @as(i32, (block.qs[j] >> 4) | xh_1) - 16;
                x0_arr[k] = @floatFromInt(x0);
                x1_arr[k] = @floatFromInt(x1);
            }
            const v_x0: Vec8 = x0_arr;
            const v_x1: Vec8 = x1_arr;
            const v_in0 = loadVec8(x, base + j_off);
            const v_in1 = loadVec8(x, base + j_off + 16);

            acc = @mulAdd(Vec8, v_x0, v_in0, acc);
            acc = @mulAdd(Vec8, v_x1, v_in1, acc);
        }

        total_acc = @mulAdd(Vec8, acc, @splat(d), total_acc);
    }
    return @reduce(.Add, total_acc);
}

test "Block sizes match GGML specifications" {
    try std.testing.expectEqual(@as(usize, 144), @sizeOf(BlockQ4_K));
    try std.testing.expectEqual(@as(usize, 210), @sizeOf(BlockQ6_K));
    try std.testing.expectEqual(@as(usize, 34), @sizeOf(BlockQ8_0));
    try std.testing.expectEqual(@as(usize, 22), @sizeOf(BlockQ5_0));
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

test "dequantizeRowQ5_0 and dotRowQ5_0 equivalence" {
    var block = BlockQ5_0{
        .d = @floatCast(0.125),
        .qh = [_]u8{ 0b00000001, 0b00000010, 0b00000100, 0b00001000 },
        .qs = @splat(0),
    };
    block.qs[0] = 0x53; // low=3, high=5
    block.qs[1] = 0x8a; // low=10, high=8
    block.qs[15] = 0x1f; // low=15, high=1

    var dst: [32]f32 = undefined;
    const blocks = [_]BlockQ5_0{block};
    dequantizeRowQ5_0(&blocks, &dst);

    var x: [32]f32 = undefined;
    for (&x, 0..) |*val, idx| {
        val.* = @as(f32, @floatFromInt(idx)) * 0.2 - 3.0;
    }

    var expected_dot: f32 = 0.0;
    for (dst, x) |d_val, x_val| {
        expected_dot += d_val * x_val;
    }

    const computed_dot = dotRowQ5_0(&blocks, &x);
    try std.testing.expectApproxEqAbs(expected_dot, computed_dot, 1e-4);
}

//! Zeno — Math and activation primitives.
//!
//! Provides RMSNorm, numerically stable softmax, entropy calculation, logit softcapping,
//! and standard activations (GeLU-tanh, SwiGLU).

const std = @import("std");

/// Computes Root Mean Square Layer Normalization with learned scale weights:
/// out_i = (x_i / sqrt(mean(x^2) + eps)) * weight_i
pub fn rmsNorm(x: []const f32, weight: []const f32, eps: f32, out: []f32) void {
    std.debug.assert(x.len == weight.len);
    std.debug.assert(x.len == out.len);

    var sum_sq: f32 = 0.0;
    for (x) |val| {
        sum_sq += val * val;
    }
    const mean_sq = sum_sq / @as(f32, @floatFromInt(x.len));
    const inv_rms = 1.0 / @sqrt(mean_sq + eps);

    for (x, weight, 0..) |xv, wv, i| {
        out[i] = xv * inv_rms * wv;
    }
}

/// Computes unweighted RMSNorm (used for router preprocessing):
/// out_i = x_i / sqrt(mean(x^2) + eps)
pub fn unweightedRmsNorm(x: []const f32, eps: f32, out: []f32) void {
    std.debug.assert(x.len == out.len);

    var sum_sq: f32 = 0.0;
    for (x) |val| {
        sum_sq += val * val;
    }
    const mean_sq = sum_sq / @as(f32, @floatFromInt(x.len));
    const inv_rms = 1.0 / @sqrt(mean_sq + eps);

    for (x, 0..) |xv, i| {
        out[i] = xv * inv_rms;
    }
}

/// Applies logit softcapping: x -> cap * tanh(x / cap)
pub inline fn logitSoftcap(x: f32, cap: f32) f32 {
    if (cap <= 0.0) return x;
    return cap * std.math.tanh(x / cap);
}

/// Applies logit softcapping in-place across a slice of logits.
pub fn logitSoftcapSlice(logits: []f32, cap: f32) void {
    if (cap <= 0.0) return;
    for (logits) |*l| {
        l.* = logitSoftcap(l.*, cap);
    }
}

/// Numerically stable softmax with temperature scaling:
/// p_i = exp((z_i / T) - max) / sum(exp((z_j / T) - max))
pub fn softmax(logits: []const f32, temperature: f32, out: []f32) void {
    std.debug.assert(logits.len == out.len);
    std.debug.assert(temperature > 0.0 and std.math.isFinite(temperature));

    if (logits.len == 0) return;

    // Find maximum scaled logit
    const inv_temp = 1.0 / temperature;
    var max_val: f32 = logits[0] * inv_temp;
    for (logits[1..]) |val| {
        const scaled = val * inv_temp;
        if (scaled > max_val) {
            max_val = scaled;
        }
    }

    // Compute exponents and sum
    var sum: f32 = 0.0;
    for (logits, 0..) |val, i| {
        const exp_val = @exp(val * inv_temp - max_val);
        out[i] = exp_val;
        sum += exp_val;
    }

    // Normalize
    const inv_sum = 1.0 / sum;
    for (out) |*p| {
        p.* *= inv_sum;
    }
}

/// Calculates Shannon entropy in nats over a probability distribution:
/// H(P) = -sum(p_i * ln(p_i))
pub fn shannonEntropyNats(probs: []const f32) f32 {
    var entropy: f32 = 0.0;
    for (probs) |p| {
        if (p > 1e-12) {
            entropy -= p * @log(p);
        }
    }
    return @max(0.0, entropy);
}

/// Computes normalized Shannon entropy H_norm = H(P) / ln(N) in [0, 1].
pub fn normalizedEntropy(entropy_nats: f32, num_candidates: usize) f32 {
    if (num_candidates <= 1) return 0.0;
    const ln_k = @log(@as(f32, @floatFromInt(num_candidates)));
    if (ln_k <= 0.0) return 0.0;
    return std.math.clamp(entropy_nats / ln_k, 0.0, 1.0);
}

/// GeLU with tanh approximation (Gemma 4 MoE expert activation):
/// gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub inline fn geluTanh(x: f32) f32 {
    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    const inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}

/// SwiGLU activation branch:
/// out_i = (gate_i * sigmoid(gate_i)) * up_i
pub fn swiglu(gate: []const f32, up: []const f32, out: []f32) void {
    std.debug.assert(gate.len == up.len);
    std.debug.assert(gate.len == out.len);

    for (gate, up, 0..) |g, u, i| {
        const sigmoid = 1.0 / (1.0 + @exp(-g));
        out[i] = (g * sigmoid) * u;
    }
}

test "rmsNorm: uniform vector normalization" {
    const x = [_]f32{ 2.0, 2.0, 2.0, 2.0 };
    const w = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var out: [4]f32 = undefined;
    rmsNorm(&x, &w, 1e-6, &out);

    for (out) |val| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), val, 1e-3);
    }
}

test "logitSoftcap: capping behavior" {
    const cap: f32 = 30.0;
    // Small value passes almost linearly
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logitSoftcap(1.0, cap), 1e-2);
    // Huge value is bounded at cap
    const capped = logitSoftcap(1000.0, cap);
    try std.testing.expect(capped <= 30.0 and capped >= 29.99);
    // Intermediate value
    try std.testing.expectApproxEqAbs(@as(f32, 30.0 * 0.761594), logitSoftcap(30.0, cap), 1e-4);
}

test "softmax and entropy: uniform distribution" {
    const logits = [_]f32{ 5.0, 5.0 };
    var probs: [2]f32 = undefined;
    softmax(&logits, 1.0, &probs);

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), probs[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), probs[1], 1e-5);

    const h_nats = shannonEntropyNats(&probs);
    // For 2 uniform choices, H = ln(2) nats
    try std.testing.expectApproxEqAbs(@log(2.0), h_nats, 1e-4);

    const h_norm = normalizedEntropy(h_nats, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), h_norm, 1e-4);
}

test "softmax and entropy: certain distribution" {
    const logits = [_]f32{ 100.0, -100.0 };
    var probs: [2]f32 = undefined;
    softmax(&logits, 1.0, &probs);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), probs[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), probs[1], 1e-5);

    const h_nats = shannonEntropyNats(&probs);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), h_nats, 1e-5);

    const h_norm = normalizedEntropy(h_nats, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), h_norm, 1e-5);
}

test "geluTanh: reference points" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), geluTanh(0.0), 1e-5);
    // gelu(1.0) approx 0.8413
    try std.testing.expectApproxEqAbs(@as(f32, 0.8413), geluTanh(1.0), 1e-3);
}

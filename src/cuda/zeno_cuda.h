#ifndef ZENO_CUDA_H
#define ZENO_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZENO_CUDA_MAX_DEVICES 8

typedef struct {
    int device_id;
    char name[256];
    int major;
    int minor;
    size_t total_memory_bytes;
    size_t free_memory_bytes;
    int multi_processor_count;
    int warp_size;
    // can_access_peer[j] == 1 if device_id can direct access peer device j VRAM (0 for self and non-capable peers)
    int can_access_peer[ZENO_CUDA_MAX_DEVICES];
} ZenoCudaDeviceInfo;

typedef struct {
    int total_visible_devices; // Total devices reported by CUDA runtime
    int reported_devices;      // Number of devices populated in devices[] (up to ZENO_CUDA_MAX_DEVICES)
    ZenoCudaDeviceInfo devices[ZENO_CUDA_MAX_DEVICES];
} ZenoCudaTopology;

// Error codes
enum {
    ZENO_CUDA_SUCCESS = 0,
    ZENO_CUDA_ERR_NO_DEVICE = 1,
    ZENO_CUDA_ERR_DRIVER_ERROR = 2,
    ZENO_CUDA_ERR_INVALID_PARAM = 3,
    ZENO_CUDA_ERR_GET_DEVICE_FAILED = 4,
    ZENO_CUDA_ERR_DEVICE_PROP_FAILED = 5,
    ZENO_CUDA_ERR_SET_DEVICE_FAILED = 6,
    ZENO_CUDA_ERR_MEM_INFO_FAILED = 7,
    ZENO_CUDA_ERR_PEER_QUERY_FAILED = 8,
    ZENO_CUDA_ERR_RESTORE_DEVICE_FAILED = 9,
    ZENO_CUDA_ERR_OUT_OF_MEMORY = 10,
    ZENO_CUDA_ERR_COPY_FAILED = 11,
    ZENO_CUDA_ERR_STREAM_FAILED = 12,
    ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED = 13,
    ZENO_CUDA_ERR_INVALID_DEVICE = 14,
    ZENO_CUDA_ERR_DEVICE_MISMATCH = 15,
    ZENO_CUDA_ERR_UNKNOWN = 99
};

// Stage 1 Topology Functions
int zeno_cuda_get_device_count(int *out_count);
int zeno_cuda_get_topology(ZenoCudaTopology *out_topo);
const char *zeno_cuda_get_error_string(int error_code);

// Stage 2 Device Memory & Stream Functions
int zeno_cuda_malloc(void **out_ptr, size_t bytes, int device_id);
int zeno_cuda_free(void *ptr, int device_id);
int zeno_cuda_memcpy_h2d(void *dst, const void *src, size_t bytes, int device_id);
int zeno_cuda_memcpy_d2h(void *dst, const void *src, size_t bytes, int device_id);
int zeno_cuda_memcpy_d2d(void *dst, const void *src, size_t bytes, int device_id, void *stream);
int zeno_cuda_memset(void *dst, int value, size_t bytes, int device_id);
int zeno_cuda_stream_create(void **out_stream, int device_id);
int zeno_cuda_stream_destroy(void *stream, int device_id);
int zeno_cuda_stream_sync(void *stream, int device_id);
int zeno_cuda_graph_begin_capture(void *stream, int device_id);
int zeno_cuda_graph_end_capture(void *stream, void **out_graph_exec, int device_id);
int zeno_cuda_graph_launch(void *graph_exec, void *stream, int device_id);
int zeno_cuda_graph_destroy(void *graph_exec, int device_id);

// Stage 2 Fused Elementwise Kernels
int zeno_cuda_rms_norm(
    const float *x,
    const float *weight,
    float *out,
    int num_rows,
    int dim,
    float eps,
    int device_id,
    void *stream
);
int zeno_cuda_geglu(
    const float *gate,
    const float *up,
    float *out,
    size_t count,
    int device_id,
    void *stream
);
int zeno_cuda_rope(
    float *vec,
    int num_tokens,
    int num_heads,
    int head_dim,
    int pos_offset,
    float theta,
    const float *freq_factors,
    int device_id,
    void *stream
);
int zeno_cuda_add_residual(
    float *x,
    const float *residual,
    size_t count,
    int device_id,
    void *stream
);
int zeno_cuda_scale(
    float *x,
    float scale,
    size_t count,
    int device_id,
    void *stream
);

// Stage 3 Quantized Matrix-Vector Multiplication (GEMV) Kernels
int zeno_cuda_gemv_q8_0(
    const void *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int batch_size,
    int device_id,
    void *stream
);
int zeno_cuda_gemv_q4k(
    const void *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int batch_size,
    int device_id,
    void *stream
);
int zeno_cuda_gemv_q6k(
    const void *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int batch_size,
    int device_id,
    void *stream
);
int zeno_cuda_gemv_q5_0(
    const void *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int batch_size,
    int device_id,
    void *stream
);
int zeno_cuda_gemv_f32(
    const float *W,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int batch_size,
    int device_id,
    void *stream
);

// Stage 4 Unified Attention & MoE Execution ABI
int zeno_cuda_attention(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    int num_tokens,
    int prompt_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int stride_q,
    int stride_k,
    int stride_v,
    int stride_out,
    int device_id,
    void *stream
);

int zeno_cuda_moe_router_prep(
    float *r_in,
    const float *router_scale,
    float root_size,
    int num_tokens,
    int dim,
    int device_id,
    void *stream
);

int zeno_cuda_moe_router_topk(
    const float *router_logits,
    const float *exp_down_scale,
    int *out_indices,
    float *out_weights,
    int num_tokens,
    int device_id,
    void *stream
);

int zeno_cuda_accumulate_scaled(
    float *accum,
    const float *x,
    float scale,
    size_t count,
    int device_id,
    void *stream
);

// Stage 5 Execution Pipeline Helpers
int zeno_cuda_layer_output(
    float *out,
    const float *x_mid,
    const float *post_ffn,
    int prompt_len,
    int num_tokens,
    int dim,
    float enc_scale,
    float layer_scale,
    int device_id,
    void *stream
);

int zeno_cuda_memcpy_peer(
    void *dst,
    int dst_device_id,
    const void *src,
    int src_device_id,
    size_t bytes
);

int zeno_cuda_enable_peer_access(int current_dev, int peer_dev);
int zeno_cuda_memcpy_peer_async(
    void *dst,
    int dst_device_id,
    const void *src,
    int src_device_id,
    size_t bytes,
    void *stream
);

int zeno_cuda_event_create(void **out_event, int device_id);
int zeno_cuda_event_destroy(void *event, int device_id);
int zeno_cuda_event_record(void *event, void *stream, int device_id);
int zeno_cuda_stream_wait_event(void *stream, void *event, int device_id);

// Stage 6 Batched Device-Side MoE Execution ABI
int zeno_cuda_moe_gemv_gate_up_batched(
    const uint8_t *all_expert_weights,
    const float *tokens_in,
    const int *top_indices,
    float *out_gate_up,
    int num_tokens,
    int in_dim,
    int out_dim,
    int device_id,
    void *stream
);

int zeno_cuda_moe_geglu_batched(
    const float *gate_up,
    float *act_out,
    int num_instances,
    int half_dim,
    int device_id,
    void *stream
);

int zeno_cuda_moe_gemv_down_batched(
    const uint8_t *all_expert_weights,
    int expert_type,
    const float *acts_in,
    const int *top_indices,
    float *out_down,
    int num_tokens,
    int in_dim,
    int out_dim,
    int device_id,
    void *stream
);

int zeno_cuda_moe_accumulate_scaled_batched(
    float *moe_accum,
    const float *down_out,
    const float *top_weights,
    int num_tokens,
    int dim,
    int device_id,
    void *stream
);

#ifdef __cplusplus
}
#endif

#endif // ZENO_CUDA_H

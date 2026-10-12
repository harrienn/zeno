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
int zeno_cuda_memcpy_d2d(void *dst, const void *src, size_t bytes, int device_id);
int zeno_cuda_memset(void *dst, int value, size_t bytes, int device_id);
int zeno_cuda_stream_create(void **out_stream, int device_id);
int zeno_cuda_stream_destroy(void *stream, int device_id);
int zeno_cuda_stream_sync(void *stream);

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

#ifdef __cplusplus
}
#endif

#endif // ZENO_CUDA_H

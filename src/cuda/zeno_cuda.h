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
    ZENO_CUDA_ERR_UNKNOWN = 99
};

int zeno_cuda_get_device_count(int *out_count);
int zeno_cuda_get_topology(ZenoCudaTopology *out_topo);
const char *zeno_cuda_get_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif // ZENO_CUDA_H

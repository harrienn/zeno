#include "zeno_cuda.h"
#include <cuda_runtime.h>
#include <string.h>

extern "C" {

int zeno_cuda_get_device_count(int *out_count) {
    if (!out_count) return ZENO_CUDA_ERR_INVALID_PARAM;
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        *out_count = 0;
        return (err == cudaErrorNoDevice) ? ZENO_CUDA_ERR_NO_DEVICE : ZENO_CUDA_ERR_DRIVER_ERROR;
    }
    if (count == 0) {
        *out_count = 0;
        return ZENO_CUDA_ERR_NO_DEVICE;
    }
    *out_count = count;
    return ZENO_CUDA_SUCCESS;
}

int zeno_cuda_get_topology(ZenoCudaTopology *out_topo) {
    if (!out_topo) return ZENO_CUDA_ERR_INVALID_PARAM;
    memset(out_topo, 0, sizeof(ZenoCudaTopology));

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return (err == cudaErrorNoDevice) ? ZENO_CUDA_ERR_NO_DEVICE : ZENO_CUDA_ERR_DRIVER_ERROR;
    }
    if (count == 0) {
        return ZENO_CUDA_ERR_NO_DEVICE;
    }

    out_topo->total_visible_devices = count;
    const int reported = (count > ZENO_CUDA_MAX_DEVICES) ? ZENO_CUDA_MAX_DEVICES : count;
    out_topo->reported_devices = reported;

    // Record caller's current device before enumeration; fail immediately if device state cannot be queried
    int orig_device = 0;
    if (cudaGetDevice(&orig_device) != cudaSuccess) {
        return ZENO_CUDA_ERR_GET_DEVICE_FAILED;
    }

    for (int i = 0; i < reported; i++) {
        ZenoCudaDeviceInfo *dev = &out_topo->devices[i];
        dev->device_id = i;

        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
            (void)cudaSetDevice(orig_device); // best-effort restore
            return ZENO_CUDA_ERR_DEVICE_PROP_FAILED;
        }

        strncpy(dev->name, prop.name, sizeof(dev->name) - 1);
        dev->name[sizeof(dev->name) - 1] = '\0';
        dev->major = prop.major;
        dev->minor = prop.minor;
        dev->multi_processor_count = prop.multiProcessorCount;
        dev->warp_size = prop.warpSize;
        dev->total_memory_bytes = prop.totalGlobalMem;

        if (cudaSetDevice(i) != cudaSuccess) {
            (void)cudaSetDevice(orig_device); // best-effort restore
            return ZENO_CUDA_ERR_SET_DEVICE_FAILED;
        }

        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
            (void)cudaSetDevice(orig_device); // best-effort restore
            return ZENO_CUDA_ERR_MEM_INFO_FAILED;
        }
        dev->free_memory_bytes = free_bytes;

        // Query P2P peer access capabilities against reported devices
        for (int j = 0; j < reported; j++) {
            if (i == j) {
                dev->can_access_peer[j] = 0; // Device is not its own peer
            } else {
                int can_access = 0;
                if (cudaDeviceCanAccessPeer(&can_access, i, j) != cudaSuccess) {
                    (void)cudaSetDevice(orig_device); // best-effort restore
                    return ZENO_CUDA_ERR_PEER_QUERY_FAILED;
                }
                dev->can_access_peer[j] = can_access;
            }
        }
    }

    if (cudaSetDevice(orig_device) != cudaSuccess) {
        return ZENO_CUDA_ERR_RESTORE_DEVICE_FAILED;
    }
    return ZENO_CUDA_SUCCESS;
}

const char *zeno_cuda_get_error_string(int error_code) {
    switch (error_code) {
        case ZENO_CUDA_SUCCESS: return "Success";
        case ZENO_CUDA_ERR_NO_DEVICE: return "No CUDA-capable device detected";
        case ZENO_CUDA_ERR_DRIVER_ERROR: return "CUDA driver or runtime error";
        case ZENO_CUDA_ERR_INVALID_PARAM: return "Invalid parameter passed to CUDA ABI";
        case ZENO_CUDA_ERR_GET_DEVICE_FAILED: return "Failed to query current active CUDA device";
        case ZENO_CUDA_ERR_DEVICE_PROP_FAILED: return "Failed to query CUDA device properties";
        case ZENO_CUDA_ERR_SET_DEVICE_FAILED: return "Failed to set active CUDA device";
        case ZENO_CUDA_ERR_MEM_INFO_FAILED: return "Failed to query CUDA device memory information";
        case ZENO_CUDA_ERR_PEER_QUERY_FAILED: return "Failed to query peer-to-peer access capability";
        case ZENO_CUDA_ERR_RESTORE_DEVICE_FAILED: return "Failed to restore original active CUDA device";
        case ZENO_CUDA_ERR_OUT_OF_MEMORY: return "CUDA out of memory";
        case ZENO_CUDA_ERR_COPY_FAILED: return "CUDA memory copy failed";
        case ZENO_CUDA_ERR_STREAM_FAILED: return "CUDA stream operation failed";
        case ZENO_CUDA_ERR_KERNEL_LAUNCH_FAILED: return "CUDA kernel launch failed";
        case ZENO_CUDA_ERR_INVALID_DEVICE: return "Invalid CUDA device ID";
        case ZENO_CUDA_ERR_DEVICE_MISMATCH: return "CUDA device mismatch";
        case ZENO_CUDA_PEER_UNSUPPORTED: return "CUDA peer-to-peer memory access unsupported by topology";
        case ZENO_CUDA_ERR_PEER_ENABLE_FAILED: return "CUDA peer-to-peer memory access enablement failed";
        default: return "Unknown CUDA error";
    }
}

}

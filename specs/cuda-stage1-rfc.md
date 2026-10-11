# RFC: Zeno CUDA Backend — Stage 1: Build Pipeline & Device Topology Discovery

- **Status**: Proposed / Under Review
- **Author**: Antigravity (Pair Programming with User)
- **Target Release**: Zeno Phase 5 (CUDA & Multi-GPU Execution)
- **Governing Specs**: [`AGENTS.md`](../AGENTS.md), [`specs/cuda-roadmap.md`](cuda-roadmap.md), [`specs/UNRESOLVED.md`](UNRESOLVED.md), [`specs/multi-gpu.md`](multi-gpu.md)
- **Target Hardware**: NVIDIA GeForce RTX 3060 12GB (Ampere `sm_86`, PCIe 4.0; dual-GPU setup)
- **Toolchain**: Zig 0.17.0 + NVIDIA `nvcc` (CUDA 13.3.73, Driver 595.99)

---

## 1. Executive Summary

This RFC specifies **Stage 1** of the Zeno CUDA backend. Its purpose is to establish native CUDA build integration via `nvcc`, define a clean C ABI boundary between Zig and CUDA C++, and implement physical GPU topology discovery (device count, compute capabilities, VRAM, and inter-GPU P2P peer access matrix) without compromising Zeno's core invariants.

### Key Objectives
1. **Zero-Dependency CPU Build Invariant**: Building Zeno without `-Dcuda` (`zig build`, `zig build test`) remains 100% pure Zig with zero C/C++ compiler or library dependencies.
2. **Explicit C ABI Boundary**: Native GPU code is written in standard CUDA C++ (`.cu`) compiled with `nvcc` into an object file (`zeno_cuda.o`) linked directly into the `zeno` executable, accessed from Zig through a minimal, extern-C header (`zeno_cuda.h`).
3. **Decoupled Topology Discovery**: Query and report visible GPU devices and inter-device P2P direct memory access capabilities via CLI (`zeno cuda-info`), without making multi-GPU availability a hard blocker for single-GPU kernel development (per Codex Review Round 10 finding R10-05).
4. **Deterministic Stack Memory**: All topology structures are value-allocated on the stack; zero dynamic heap allocations are performed during device query.

---

## 2. Architecture & Language Boundary

### 2.1 Language Separation Rationale
Per the architectural recommendation in [`reviews/REVIEW-round10.md`](../reviews/REVIEW-round10.md):
- **Host Engine in Pure Zig**: Request handling, memory lifecycle, tensor layout/ownership, scheduling, CPU reference fallback, and CLI interface remain in Zig.
- **GPU Kernels in CUDA C++**: NVIDIA `nvcc` provides standard compilation for Ampere (`sm_86`) warp intrinsics (`__shfl_sync`), shared memory tiling, and stable CUDA runtime APIs.
- **Pure C ABI Boundary**: Zig interfaces with CUDA via `extern "C"` functions taking primitive C types and structs. No C++ classes, templates, or name mangling cross the ABI boundary.

```text
┌────────────────────────────────────────────────────────┐
│                      Host (Zig)                        │
│   src/main.zig (CLI)  ──►  src/cuda.zig (Safe API)    │
└───────────────────────────────┬────────────────────────┘
                                │ C ABI: extern "C"
┌───────────────────────────────▼────────────────────────┐
│                   C ABI Boundary Layer                 │
│                 src/cuda/zeno_cuda.h                   │
└───────────────────────────────┬────────────────────────┘
                                │ Implemented in CUDA C++
┌───────────────────────────────▼────────────────────────┐
│                   CUDA Runtime Engine                  │
│   src/cuda/zeno_cuda.cu  ──►  CUDA 13.3 Runtime API   │
│   (Compiled by nvcc -arch=sm_86 -Xcompiler -fPIC)     │
└────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Technical Specification

### 3.1 C ABI Header (`src/cuda/zeno_cuda.h`)

The C ABI defines fixed-size, stack-allocatable structures and pure C function signatures:

```c
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
```

### 3.2 CUDA Implementation (`src/cuda/zeno_cuda.cu`)

Implements device enumeration, property inspection, and inter-GPU peer access testing:

```cpp
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
        default: return "Unknown CUDA error";
    }
}

}
```

> **Device Restoration Precedence Rule (R13-01)**:
> - On the normal exit path, failure to restore the caller's initial active device returns `ZENO_CUDA_ERR_RESTORE_DEVICE_FAILED` (`error.RestoreDeviceFailed`).
> - If a fatal query error occurs while inspecting an enumerated device (such as `cudaMemGetInfo` or `cudaGetDeviceProperties` failing), the function performs best-effort restoration of `orig_device` before returning the primary query error. The primary query error takes precedence; if restoration also fails on that error path, the caller's active device may remain altered.


### 3.3 Build System Integration (`build.zig`)

`build.zig` receives two configurable options:
- `-Dcuda=[bool]` (default: `false`): controls whether CUDA compilation is activated.
- `-Dcuda-arch=[string]` (default: `"sm_86"`): specifies target architecture (e.g. `sm_86` for RTX 3060/3090, `sm_89` for RTX 4090, `sm_90` for H100).

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const enable_cuda = b.option(bool, "cuda", "Enable CUDA backend (default: false)") orelse false;
    const cuda_arch = b.option([]const u8, "cuda-arch", "CUDA target compute architecture (default: sm_86)") orelse "sm_86";
    const test_model = b.option(bool, "test-model", "Run heavy full-model forward pass tests (default: false)") orelse false;

    const build_options = b.addOptions();
    build_options.addOption(bool, "test_model", test_model);
    build_options.addOption(bool, "enable_cuda", enable_cuda);
    build_options.addOption([]const u8, "cuda_arch", cuda_arch);

    // Zeno core library module
    const zeno_mod = b.addModule("zeno", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    zeno_mod.addImport("build_options", build_options.createModule());

    // Main CLI executable
    const exe = b.addExecutable(.{
        .name = "zeno",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zeno", .module = zeno_mod },
                .{ .name = "build_options", .module = build_options.createModule() },
            },
        }),
    });

    if (enable_cuda) {
        // Find nvcc compiler and derive CUDA toolkit root directory
        const nvcc_path = b.findProgram(&.{"nvcc"}, &.{
            "/usr/local/cuda-13.3/bin",
            "/usr/local/cuda/bin",
        }) catch "nvcc";

        // Derive toolkit root from nvcc path (e.g. /usr/local/cuda-13.3/bin/nvcc -> /usr/local/cuda-13.3)
        const cuda_bin_dir = std.fs.path.dirname(nvcc_path) orelse "/usr/local/cuda/bin";
        const cuda_root = std.fs.path.dirname(cuda_bin_dir) orelse "/usr/local/cuda";
        const cuda_lib_dir = b.fmt("{s}/lib64", .{cuda_root});

        const arch_flag = b.fmt("-arch={s}", .{cuda_arch});

        // Compile zeno_cuda.cu to an object file
        const nvcc_cmd = b.addSystemCommand(&.{
            nvcc_path,
            "-c",
            "-O3",
            "-Xcompiler",
            "-fPIC",
            arch_flag,
            "-Isrc/cuda",
            "src/cuda/zeno_cuda.cu",
            "-o",
        });
        const cuda_obj = nvcc_cmd.addOutputFileArg("zeno_cuda.o");

        // Single link owner: attach CUDA object and runtime library directly to executable
        exe.root_module.addObjectFile(cuda_obj);
        exe.root_module.addLibraryPath(.{ .cwd_relative = cuda_lib_dir });
        exe.root_module.linkSystemLibrary("cudart", .{});
        exe.root_module.link_libc = true;
    }

    b.installArtifact(exe);
    ...
}
```

### 3.4 Zig Host API (`src/cuda.zig`)

Provides typed, ergonomic Zig access to the C ABI with value-type stack storage:

```zig
const std = @import("std");
const build_options = @import("build_options");

pub const is_enabled = build_options.enable_cuda;

pub const CudaError = error{
    NoDevice,
    DriverError,
    InvalidParam,
    GetDeviceFailed,
    DevicePropFailed,
    SetDeviceFailed,
    MemInfoFailed,
    PeerQueryFailed,
    RestoreDeviceFailed,
    UnknownError,
    CudaDisabledAtBuild,
};

pub const DeviceInfo = struct {
    device_id: i32,
    name_buf: [256]u8,
    name_len: usize,
    compute_major: i32,
    compute_minor: i32,
    total_memory_bytes: usize,
    free_memory_bytes: usize,
    multi_processor_count: i32,
    warp_size: i32,
    can_access_peer: [8]bool,

    pub fn name(self: *const DeviceInfo) []const u8 {
        return self.name_buf[0..self.name_len];
    }
};

pub const Topology = struct {
    total_visible_devices: usize,
    reported_devices: usize,
    devices: [8]DeviceInfo,
};

pub fn getTopology() CudaError!Topology {
    if (!is_enabled) {
        return error.CudaDisabledAtBuild;
    }
    // Call C ABI zeno_cuda_get_topology, copy name inline, and convert to Zig struct
    ...
}
```

> **Unit Test Portability Rule (R14-02)**:
> In `src/cuda.zig`, the CUDA-enabled unit test `cuda module topology query when enabled` catches `error.NoDevice` and `error.DriverError` and returns `error.SkipZigTest`. This ensures `zig build test -Dcuda=true` passes cleanly on toolkit-only or headless build environments lacking an accessible GPU, while validating physical topology when GPU hardware is present.

### 3.5 CLI Subcommand (`zeno cuda-info`)

When invoked, `zeno cuda-info` outputs human-readable hardware topology:

```text
======================================================================
 Zeno CUDA Hardware Topology & Capabilities
======================================================================
CUDA Backend:        ENABLED (Compiled with -Dcuda=true, sm_86)
Visible Devices:     2 (Reported: 2)

[Device 0]
  Model:             NVIDIA GeForce RTX 3060
  Compute Cap:       8.6 (Ampere)
  SM Count:          28
  Warp Size:         32
  Total VRAM:        12,288 MiB (12.88 GB)
  Free VRAM:         10,534 MiB

[Device 1]
  Model:             NVIDIA GeForce RTX 3060
  Compute Cap:       8.6 (Ampere)
  SM Count:          28
  Warp Size:         32
  Total VRAM:        12,288 MiB (12.88 GB)
  Free VRAM:         12,275 MiB

----------------------------------------------------------------------
 Inter-GPU Peer-to-Peer (P2P) Access Capability Matrix:
 (Capability query only; does not imply P2P is currently enabled)
----------------------------------------------------------------------
              GPU 0        GPU 1
  GPU 0         —         Capable
  GPU 1      Capable         —
======================================================================
```

If built with `-Dcuda=false`:
```text
Zeno CUDA backend is DISABLED in this binary build.
To enable CUDA support, rebuild with: zig build -Dcuda=true
```

> **CLI Exit Code Contract (R14-01)**:
> - Running `zeno cuda-info` against a binary built with `-Dcuda=false` prints the informational notice and exits successfully (`exit code 0`).
> - When built with `-Dcuda=true`, any runtime failure from `getTopology()` (e.g. `error.NoDevice`, driver initialization errors) prints diagnostic context to stdout, flushes the writer buffer, and propagates the error so the process exits with a non-zero failure code (`exit code 1`). CI scripts checking exit status will correctly detect query failures.

---

## 4. Verification & Acceptance Criteria

Stage 1 exit gates conform to [`specs/cuda-roadmap.md`](cuda-roadmap.md) §3:

1. **Gate 1: Default Build Isolation**:
   - `zig build` compiles without invoking `nvcc` and without linking C/CUDA libraries.
   - `zig build test` passes all unit tests (46 passed, 3 model tests skipped) with 0 memory leaks.
2. **Gate 2: Native CUDA Build & Portable Unit Testing**:
   - `zig build -Dcuda=true` successfully invokes `nvcc`, compiles `zeno_cuda.cu`, and links `libcudart` from the derived toolkit path.
   - `zig build test -Dcuda=true` succeeds on both GPU-equipped hosts (validating device properties) and toolkit-only hosts (skipping hardware query cleanly via `error.SkipZigTest`).
3. **Gate 3: Device Discovery Execution & Exit Status**:
   - `./zig-out/bin/zeno cuda-info` executes cleanly, correctly detecting all visible CUDA devices on the test host and their peer-access capability with exit code 0.
   - Queries with missing devices (`CUDA_VISIBLE_DEVICES=""`) print diagnostic failure and terminate with non-zero exit code.
4. **Gate 4: Integration Safety**:
   - All 14 HTTP server tests (`python3 tools/test_server.py`) continue to pass without regression.

---

## 5. Next Steps Post-Stage 1

Upon review approval and Stage 1 verification:
- **Stage 2**: Device memory allocation wrappers (`src/cuda_mem.zig`) and fused elementwise kernels (`src/cuda/ops.cu`: `rms_norm`, `rope`, `geglu`).
- **Stage 3**: Quantized GEMV dot products (`Q4_K`, `Q6_K`, `Q8_0`).


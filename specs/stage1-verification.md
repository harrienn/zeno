# Stage 1 Verification Report — CUDA Build Integration & Topology Discovery

- **Date**: 2026-10-11
- **Target Hardware**: 2x NVIDIA GeForce RTX 3060 12GB (Ampere `sm_86`, PCIe 4.0)
- **Host System**: Linux (Kernel 6.17), CUDA Toolkit 13.3.73, NVIDIA Driver 595.99
- **Compiler**: Zig 0.17.0 + `nvcc` (CUDA 13.3)
- **Commits**: `457af85` (`feat(cuda): implement Stage 1 build integration and device discovery`), `ee31f5c` (`fix(cuda): address Codex review round 14 findings`), `5d47bd8` (R15 updates)

---

## 1. Summary of Exit Gates

| Gate | Description | Command | Result |
|---|---|---|---|
| **Gate 1** | Pure Zig CPU default build isolation | `zig build && zig build test` | **PASS**: 46 passed, 3 skipped (model weights). Zero C/CUDA toolchain invocation. |
| **Gate 2** | Fallback CLI diagnostic | `./zig-out/bin/zeno cuda-info` (CPU binary) | **PASS**: Prints disabled notice, exit code 0. |
| **Gate 3** | Physical CUDA topology discovery | `./zig-out/bin/zeno cuda-info` (CUDA binary) | **PASS**: Correctly reports 2x RTX 3060 12GB, SM count 28, CC 8.6, P2P matrix, exit code 0. |
| **Gate 3b** | Query failure CLI exit code (R14-01) | `CUDA_VISIBLE_DEVICES="" ./zig-out/bin/zeno cuda-info` | **PASS**: Flushes diagnostic error to stdout, terminates with non-zero exit code 1. |
| **Gate 4a** | CUDA test suite & portability (R14-02, R15-06) | `zig build test -Dcuda=true --summary all` | **PASS**: 46 passed with GPU. Skips cleanly on `error.NoDevice` when `CUDA_VISIBLE_DEVICES=""` (45 passed, 4 skipped). |
| **Gate 4b** | Full HTTP decision API regression | `python3 tools/test_server.py` | **PASS**: All 14/14 HTTP server integration tests passed. |

---

## 2. Gate 3 Output: Physical Dual-GPU Discovery

Command: `./zig-out/bin/zeno cuda-info` (compiled with `zig build -Dcuda=true`):

```text
======================================================================
 Zeno CUDA Hardware Topology & Capabilities
======================================================================
CUDA Backend:        ENABLED (Compiled with -Dcuda=true, sm_86)
Visible Devices:     2 (Reported: 2)

[Device 0]
  Model:             NVIDIA GeForce RTX 3060
  Compute Cap:       8.6
  SM Count:          28
  Warp Size:         32
  Total VRAM:        11908 MiB (12.49 GB)
  Free VRAM:         10059 MiB

[Device 1]
  Model:             NVIDIA GeForce RTX 3060
  Compute Cap:       8.6
  SM Count:          28
  Warp Size:         32
  Total VRAM:        11911 MiB (12.49 GB)
  Free VRAM:         11787 MiB

----------------------------------------------------------------------
 Inter-GPU Peer-to-Peer (P2P) Access Capability Matrix:
 (Capability query only; does not imply P2P is currently enabled)
----------------------------------------------------------------------
                 GPU 0       GPU 1   
  GPU 0            —        Not Capable  
  GPU 1       Not Capable        —       
======================================================================
```
Exit code: `0`.

---

## 3. Gate 3b Output: Query Failure Handling (R14-01)

Command: `CUDA_VISIBLE_DEVICES="" ./zig-out/bin/zeno cuda-info`:

```text
======================================================================
 Zeno CUDA Hardware Topology & Capabilities
======================================================================
CUDA Backend:        ENABLED (Compiled with -Dcuda=true, sm_86)
Error querying CUDA device topology: error.NoDevice
======================================================================
error: NoDevice
```
Exit code: `1`. Confirms proper non-zero termination for failed runtime queries.

---

## 4. Gate 4b Output: Server Regression Test

Command: `python3 tools/test_server.py`:
- 14/14 tests passed including live single-step decision query with JSON serialization, framing robustness, canvas bounds, and wire error code verification.

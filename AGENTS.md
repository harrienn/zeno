# Zeno

Zeno is a Zig inference runtime for DiffusionGemma.

## Principles

- Prefer explicit data flow over hidden abstraction.
- Prefer simple structs over deep generic hierarchies.
- `comptime` is for static model configuration and specialization.
- Runtime model state must remain runtime data.
- Tensor shape and layout must be explicit.
- No unnecessary allocation in the denoising loop.
- Every optimized kernel should have a reference implementation.
- Do not introduce an abstraction unless it is used by at least two components.

## Memory

- Ownership must be explicit.
- Every allocation has a documented owner.
- Long-lived model weights must not use per-step allocators.
- Temporary denoising state may use an arena.

## GPU

- CPU reference implementation first.
- CUDA implementation must preserve the same semantic contract.
- Never silently change tensor layout for kernel convenience.

## Development

Before changing architecture:
1. inspect existing code;
2. explain the proposed change;
3. identify affected invariants;
4. run tests after implementation.

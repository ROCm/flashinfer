# CLAUDE.md

> AMD ROCm port of FlashInfer (`amd-flashinfer`), targeting gfx942/gfx950.
> The upstream CUDA repo at <https://github.com/flashinfer-ai/flashinfer> uses
> different env vars, paths, and toolchains — its patterns don't apply here.

## Essential Commands

| Task | Command |
|------|---------|
| Install for development | `python -m pip install --no-build-isolation -ve.` |
| Run tests (fast) | `pytest -n auto --reruns 2 -m "not slow"` |
| Run all tests | `pytest -n auto --reruns 2` |
| Clear JIT cache | `rm -rf ~/.cache/flashinfer/` |
| Set target arch | `export FLASHINFER_ROCM_ARCH_LIST="gfx942,gfx950"` |
| Limit parallel build | `export MAX_JOBS=4` |
| Verbose JIT output | `export FLASHINFER_JIT_VERBOSE=1` |
| Run linting | `pre-commit run -a` |

## Installing Torch

Torch must come from AMD's ROCm repo via `--index-url`. Using `-f` (find-links)
still allows PyPI fallback and may silently install a CPU-only wheel.

```bash
pip install torch==<torch-version> \
  --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-<rocm-version>/
```

See the [GPU and ROCm Support](README.md#gpu-and-rocm-support) table in
`README.md` for current `<torch-version>` and `<rocm-version>` values.

## Non-Obvious Gotchas

**JIT build.ninja caching**: `JitSpec.build()` only writes `build.ninja` when
the file is missing. Changing env vars (`FLASHINFER_ROCM_ARCH_LIST`, extra
cflags) is a **silent no-op** unless you call `spec.write_ninja()` first.

**`FLASHINFER_JIT_DEBUG=1` is a CUDA-only no-op**: the env var is read in
[`flashinfer/jit/core.py`](flashinfer/jit/core.py) only on the `IS_CUDA` branch
(adds `-O0 -g -G`). The `IS_HIP` branch ignores it. To get a debug build on
ROCm, add `"-g"` (and remove `-O3`) via `extra_cuda_cflags` in the op's JIT
generator and clear `~/.cache/flashinfer/`.

**Framework separation**: Torch headers **must not** be included in `include/`
files. `include/` is framework-agnostic (raw pointers only);
`flashinfer/csrc_rocm/` is where PyTorch tensor handling lives. Violations
cause subtle build failures.

**Test parallelism**: `pytest -n auto` automatically halves the physical GPU
count to avoid HSA/HIPBLAS flakiness under concurrent load. The `slow` marker
gates 1M-trial sampling and 4 GB tensor tests — exclude with `-m "not slow"`
for fast iteration.

**AITER is a separate install**: The AITER backend (used by prefill attention
on gfx942) is not bundled. Install from source:

```bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter && python3 setup.py develop
```

Check availability in code: `from flashinfer.aiter_utils import is_aiter_supported`

## Key External References

Arch ↔ codename mapping (frequently needed mid-coding): MI300X / MI325X =
gfx942 = CDNA3; MI350X = gfx950 = CDNA4.

- **Composable Kernel (CK)** — ground truth for LDS layout,
  `sched_group_barrier` ratios, and tiling on CDNA. When tuning a kernel,
  read `qr_ks_vs.hpp` in [CK](https://github.com/ROCmSoftwarePlatform/composable_kernel)
  for the specific hdim/dtype combination first.
- **AITER** — performance reference for fp16/hdim=256 attention on gfx942.
  [AITER repo](https://github.com/ROCm/aiter)
- **HipKittens** (arxiv 2511.08083) — producer/consumer patterns underperform
  on CDNA; 4-wave interleave is the recommended approach.

## GitHub CLI

`gh pr edit` fails with a "Projects (classic) is being deprecated" GraphQL error on this repo. Use the REST API instead:

```bash
# Update PR description
gh api repos/ROCm/flashinfer/pulls/<number> --method PATCH --field body="<body>"

# Or from a file
gh api repos/ROCm/flashinfer/pulls/<number> --method PATCH --field body="$(cat /tmp/pr_body.md)"
```

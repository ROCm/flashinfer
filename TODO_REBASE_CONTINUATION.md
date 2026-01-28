# Comprehensive TODO List for AMD FlashInfer Fork Update

## Context
- **Fork point**: commit `83d1c745` (v0.2.5 era)
- **Current upstream**: v0.4.1 (852 commits ahead)
- **AMD work**: 152 commits of HIP/ROCm adaptations
- **Branch structure**:
  - `amd-integration` - main trunk (at original fork point)
  - `rebase-ontop-of-v0.3.0` - test branch with completed rebase to v0.3.0 (147 commits applied)
  - `feature/aot.py` - AOT/JIT infrastructure refactor work (6 commits ahead of amd-integration)

## Strategy
Incremental rebase: v0.2.5 → **v0.3.0** ✅ → v0.3.1.post1 → v0.4.1

---

## 🔥 IMMEDIATE: Fix Architecture Flag Propagation (feature/aot.py branch)

**Problem**: `validate_rocm_arch()` validates architecture list but it's never used in compilation - hardcoded `gfx942` everywhere.

**Required Changes**:

### 1. **flashinfer/aot_hip.py** (around line 208)
```python
# After this line:
rocm_arch_list = hip_utils.validate_rocm_arch(verbose=verbose)

# ADD:
os.environ["FLASHINFER_ROCM_ARCH_LIST"] = rocm_arch_list
```

### 2. **flashinfer/jit/core.py** - `check_rocm_arch()` function (lines 60-68)
Replace hardcoded check with:
```python
def check_rocm_arch():
    # Get validated architectures from environment (set by aot_hip.py)
    rocm_arch_list = os.environ.get("FLASHINFER_ROCM_ARCH_LIST", "gfx942")
    allowed_archs = [f"--offload-arch={arch.strip()}" for arch in rocm_arch_list.split(",")]

    hip_arch_flags = torch_cpp_ext._get_rocm_arch_flags()
    for arch in allowed_archs:
        if arch not in hip_arch_flags:
            raise RuntimeError(f"FlashInfer requires {', '.join(allowed_archs)}")
```

### 3. **flashinfer/jit/core.py** - `gen_jit_spec()` function (lines 145-154)
Replace hardcoded `--offload-arch=gfx942` with:
```python
if check_hip_availability():
    # Use validated architecture list from environment
    rocm_arch_list = os.environ.get("FLASHINFER_ROCM_ARCH_LIST", "gfx942")
    arch_flags = [f"--offload-arch={arch.strip()}" for arch in rocm_arch_list.split(",")]
    cflags += arch_flags
    cflags += [
        "-DFLASHINFER_ENABLE_HIP",
        "-DFLASHINFER_ENABLE_FP8",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
    ]
```

**Why**: Enables multi-arch support (e.g., gfx942,gfx90a) and respects validated arch list

---

## AOT/JIT Infrastructure Completion (feature/aot.py branch)

### Already Completed ✅
- Fixed directory path issue: lazy imports after setting `FLASHINFER_WORKSPACE_BASE`
- Updated `copy_built_kernels()` to use build_dir parameter

### Pending Tasks

**4. Test AOT Package Build**
```bash
cd /home/AMD/diptodeb/devel/flashinfer
python flashinfer/aot_hip.py --verbose
# Verify amd-flashinfer-jit-cache package is created correctly
```

**5. Remove Old CMake AOT Infrastructure**
- Delete or disable CMake-based AOT build system (if still present)
- Ensure new Python-based AOT is the only system

**6. Remove aot_build_utils Directory**
- Contains old template generation scripts (generate_batch_paged_decode_inst.py, etc.)
- Obsolete after new AOT system

**7. Update FlashInfer Modules**
- Remove `prebuilt_uri` references from flashinfer modules
- Update to use new AOT package system
- **Files to check**: All `*_hip.py` files in `flashinfer/`

**8. Align JIT with v0.3.0 Patterns**
- v0.3.0 uses `build_jit_specs()` function
- Removed `parallel_load_modules()` and `has_prebuilt_ops` in v0.3.0
- **Key file**: `flashinfer/jit/attention/pytorch_hip.py`
- Review upstream changes: `git diff v0.2.5..v0.3.0 -- python/flashinfer/`

---

## Test Updates (rebase-ontop-of-v0.3.0 branch)

**Context**: 4 original test files changed between v0.2.5 and v0.3.0, need to port changes to `*_hip.py` variants.

**9. Update Test Files**
```bash
# Changed files identified:
tests/test_batch_decode.py
tests/test_page.py
tests/test_prefill.py
tests/test_sampling.py

# Need to port changes to:
tests/test_batch_decode_hip.py
tests/test_page_hip.py
tests/test_prefill_hip.py
tests/test_sampling_hip.py
```

**Approach**:
- For each changed test, run: `git diff v0.2.5..v0.3.0 -- tests/test_<name>.py`
- Apply equivalent changes to `tests/test_<name>_hip.py`
- New test: `test_batch_decode_with_multi_item_scoring()` added in v0.3.0

---

## Integration and Continuation

**10. Cherry-pick AOT/JIT Work to Rebase Branch**
```bash
git checkout rebase-ontop-of-v0.3.0
git cherry-pick feature/aot.py  # Or use git format-patch + git am
# Resolve any conflicts
git push origin rebase-ontop-of-v0.3.0 --force-with-lease
```

**11. Continue Incremental Rebase to v0.3.1.post1**
```bash
git checkout -b rebase-ontop-of-v0.3.1.post1 rebase-ontop-of-v0.3.0
git rebase --onto v0.3.1.post1 v0.3.0
# Resolve conflicts incrementally
```

**12. Final Rebase to v0.4.1**
```bash
git checkout -b rebase-ontop-of-v0.4.1 rebase-ontop-of-v0.3.1.post1
git rebase --onto v0.4.1 v0.3.1.post1
# Resolve conflicts
# Extensive testing required
```

**13. Validation and Merge**
- Run full test suite on rebase branch
- Coordinate with team for review
- Force-push to `amd-integration` after approval:
  ```bash
  git push origin rebase-ontop-of-v0.4.1:amd-integration --force-with-lease
  ```

---

## Important Technical Details

### File Changes from v0.2.5 to v0.3.0
- **24 files modified** (including test files)
- **16 files deleted** (TVM bindings, old setup.py)
- **195 files added** (HIP kernels, CMake, gpu_iface abstraction layer)

### Environment Variables
- `FLASHINFER_WORKSPACE_BASE` - must be set BEFORE importing jit modules
- `FLASHINFER_ROCM_ARCH_LIST` - comma-separated arch list (e.g., "gfx942,gfx90a")
- `FLASHINFER_JIT_VERBOSE` - set to "1" for debug output

### Key Patterns
- **Lazy imports**: Import jit modules inside functions AFTER setting env vars
- **Module-level constants**: Bind at import time in `flashinfer/jit/env.py`
- **Architecture validation**: `hip_utils.validate_rocm_arch()` checks against ROCm version matrix

### Build System
- Migrated from setuptools to scikit-build-core
- AOT now uses standalone Python package (amd-flashinfer-jit-cache)
- JIT uses template-based generation with Jinja2

### Execution Flow for Architecture Flags
```
aot_hip.py::compile_and_package_modules()
  → validates ROCm arch
  → sets FLASHINFER_ROCM_ARCH_LIST env var
  → calls gen_*_module() functions
    → pytorch_hip.py::gen_single_decode_module(), gen_batch_decode_module()
      → calls gen_jit_spec()
        → core.py::gen_jit_spec()
          → reads FLASHINFER_ROCM_ARCH_LIST from env
          → generates --offload-arch flags
          → returns JitSpec for compilation
```

---

## Quick Start Commands for New Agent

```bash
# Navigate to workspace
cd /home/AMD/diptodeb/devel/flashinfer

# Check current branch (should be feature/aot.py)
git branch -a
git log --oneline -5

# View architecture flag issue in current files
grep -n "offload-arch=gfx942" flashinfer/jit/core.py
grep -n "rocm_arch_list = hip_utils.validate_rocm_arch" flashinfer/aot_hip.py

# Start with immediate fix (task #1 - architecture flags)
# Edit the 3 locations identified above
# Then proceed through tasks #2-13 sequentially
```

---

## Notes from Previous Session

### What Was Completed
1. ✅ Defined incremental rebase strategy (v0.3.0 → v0.3.1.post1 → v0.4.1)
2. ✅ Completed rebase to v0.3.0 (147 commits, all conflicts resolved)
3. ✅ Analyzed file changes (24 modified, 16 deleted, 195 added)
4. ✅ Identified test infrastructure blocker
5. ✅ Fixed directory path issue in aot_hip.py (lazy imports)
6. ✅ Traced execution flow for architecture flag propagation

### What Was In-Progress
- Architecture flag propagation fix (identified 3 locations to change)
- Code changes prepared but not applied due to Docker container issue

### Why Docker Restart
User reported: "Something is off with my docker container and I need to start fresh"

### Critical Context for New Container
- Working directory: `/home/AMD/diptodeb/devel/flashinfer`
- Active branch: `feature/aot.py` (6 commits ahead of amd-integration)
- Python environment: `amd-flashinfer-rocm7.1.1-torch2.8.0` (conda)
- The architecture flag fix is ready to implement - just needs the 3 file edits above

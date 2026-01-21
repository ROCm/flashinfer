# v0.2.5+rocm.2

## Added

### Prefill Kernels
- Port BatchPrefillWithPagedKVCacheDevice kernel to HIP (#63) @Madduri, Rishi
- Port BatchPrefillWithRaggedKVCache to HIP (#50) @Madduri, Rishi
- Add HIPified version of SinglePrefillWithKVCacheDevice kernel (#31) @Diptorup Deb
- Add tests for LSE in single prefill example script (#46) @Debasis Mandal
- Add test_batch_prefill.cpp to HIP (#43) @Madduri, Rishi
- Batch prefill example script (#58) @Debasis Mandal
- Single prefill example script (#36) @Debasis Mandal

### Decode & Tensor Operations
- Enable FP8 support for Flashinfer ROCm decode kernels on CDNA3 (#40) @Madduri, Rishi
- Add mfma_fp32_16x16x16fp16 op (#62) @Madduri, Rishi
- Add mfma row sum operation (#68) @Madduri, Rishi
- Add wrapper for hgemm kernel (#69) @Madduri, Rishi

### Infrastructure & Testing
- pytest configuration improvements for ROCm/HIP testing (#65) @Diptorup Deb
- Unify pytests for batch prefill for HIP and add to CI (#90) @Debasis Mandal
- Add more PyTests from upstream to ROCm CI (#93) @Debasis Mandal
- Run tests/test_non_contiguous_prefill.py on HIP (#96) @Debasis Mandal
- Run tests/test_activation.py on HIP (#98) @Debasis Mandal
- Port unit tests to use gpu_iface (#52) @Madduri, Rishi
- Setup pytests (#78) @Madduri, Rishi
- Make single prefill example script standalone (#104) @Debasis Mandal
- Add b64_t (uint2) type support to smem_t class (#53) @Diptorup Deb

### Build System
- Enable JIT Installation for Prefill (#22) @Madduri, Rishi
- Enable AOT Installation for Prefill (#21) @Madduri, Rishi
- Enable JIT via gpu_iface (#57) @Madduri, Rishi
- Building AOT with gpu_iface (#54) @Madduri, Rishi
- Add HIP support to AOT Build Utils (#34) @Madduri, Rishi
- Port flashinfer build system to scikit-build-core (#14) @Diptorup Deb
- Infrastructure for enabling Decode via gpu_iface (#43) @Madduri, Rishi
- Add norm, page and rope to jit build infra (#46) @Madduri, Rishi

## Changed

### Performance Optimizations
- Change Swizzle mode from kLinear to k128B for prefill (#103) @Debasis Mandal
- Decode feature chunking logic and shared mem optimization (#25) @Madduri, Rishi

### Version Updates
- Update ROCm+Torch versions in Dev Dockerfile (ROCm 7.1.1, PyTorch 2.8.0) (#118) @Debasis Mandal
- Upgrade to ROCm 6.4, Ubuntu 24.04, Python 3.12, PyTorch 2.7.1 (#73) @Mandal, Debasis
- Update devcontainer dockerfile to ROCm 7.0.2 (#24) @Debasis Mandal

### Architecture Support
- Support only gfx942 arch for ROCm (#45) @Debasis Mandal

## Fixed

### Kernel Bugs
- Fix datatypes for HIP when using customized attention kernels (#111) @Debasis Mandal
- Fix partition-kv=True case and memory allocation issues in batch prefill (#89) @Debasis Mandal
- Fix single prefill kernel dispatch for HEAD_DIM_QK values > 64 (#86) @Diptorup Deb
- Fix threadblock sync mdo (#62) @Diptorup Deb
- Fix single prefill dispatch for HIP devices (#64) @Diptorup Deb
- Fix Log-sum-exp (LSE) write back for single prefill kernels (#42) @Diptorup Deb
- Fix write_o_reg_gmem kernel (#39) @Diptorup Deb
- Decode Bug fix (#72) @Madduri, Rishi
- Fix fragment loading to properly pack 16b values into 32b register (#2) @Diptorup Deb

### Build & Packaging
- Add custom ROCm version scheme to fix wheels version naming (#110) @Diptorup Deb
- Change pyproject.toml to generate revised whl name (#113) @Madduri, Rishi
- Improvements to Python packaging infrastructure (#76) @Diptorup Deb
- Require v9.2 of setuptools_scm (#74) @Diptorup Deb
- Fix header installation for editable installs (#20) @Diptorup Deb
- Fixing the JIT build command (#47) @Madduri, Rishi

### Testing & Examples
- Skip failing C++ tests and fix mma_debug_utils (#59) @Diptorup Deb
- Fix batch prefill example script for ragged kv cache (#73) @Debasis Mandal
- Replace test_cascade with the upstream version (#51) @Madduri, Rishi

### Documentation & Infrastructure
- Mark copied git repo as a safe directory (#114) @Diptorup Deb
- Update README Wheel installation section (#112) @Diptorup Deb
- Update README with clearer usage guide (#72) @Diptorup Deb
- Update build instructions for C++ tests in README.md (#27) @Debasis Mandal
- Update hyperlinks in the TOC of README (#33) @Debasis Mandal
- Fix SPDX headers for AMD authored files (#37) @Diptorup Deb
- Fix some compiler warnings in Cxx unit tests (#13) @Diptorup Deb

### Docker & CI
- Update Dockerfile (#116) @Madduri, Rishi
- Update Dockerfile.rocm_ci (#78, #67, #6) @Diptorup Deb, @Madduri, Rishi, @Clint
- Fix Dockerfile.rocm_ci (#77) @Diptorup Deb
- Dockerfile fix (#81) @Madduri, Rishi
- Fix dockerfile rocm.ci (#70) @Diptorup Deb
- Revert tar changes from Dockerfile (#68) @Madduri, Rishi

## Maintenance

### Codebase Refactoring
- Refactor codebase to remove libflashinfer (#88) @Diptorup Deb
- Reduce tech debt by removing CUDA sections from generic/prefill.cuh (#87) @Diptorup Deb
- Refactor single prefill tests (#51) @Debasis Mandal
- Refactor C++ test suite (#77) @Madduri, Rishi
- Clean up pytests (#44) @Madduri, Rishi
- Cleanup/release prep (#79) @Diptorup Deb

### Docker & CI Improvements
- Change base image to rocm/dev-ubuntu (#69) @Diptorup Deb
- Update ROCm CI docker (#58) @Madduri, Rishi
- Enhance devcontainer Dockerfile for ROCm (#28) @Debasis Mandal
- Remove hardcoded mamba env name from Docker image (#71) @Diptorup Deb
- Remove USER requirement from Docker (#50) @Madduri, Rishi

### Code Quality
- Update C++/CUDA/HIP code base to follow upstream clang-format (#75) @Madduri, Rishi
- Update .clangd to support HIP specific flags (#74) @Debasis Mandal
- Add clangd configuration (#32) @Diptorup Deb
- Ignore nfs related files in git tracking (#75) @Debasis Mandal
- Fix formatting issues identified by pre-commit linters @Madduri, Rishi

### Build System
- Update logging for using JIT in absence of AOT (#94) @Debasis Mandal
- Remove verbose CMake installation messages for editable JIT (#97) @Debasis Mandal
- Update CMake infra to run HIP CXX tests using top-level cmake (#10) @Diptorup Deb
- Initial CMake changes for HIP support (#16) @Diptorup Deb

## Removed

- Remove leftover src and all tvm bindings (#99) @Diptorup Deb
- Remove xfail markers about HIP support from pytests (#92) @Debasis Mandal
- Remove unused submodules (#55) @Madduri, Rishi
- Deprecate Decode C++ Unit Test and capture Cascade test values (#15) @Madduri, Rishi
- Remove test_transpose_4x4_half_registers (#11) @Diptorup Deb
- Remove unnecessary gitmodules (#21) @Diptorup Deb
- Disable test_custom_allreduce (#108) @Madduri, Rishi
- Remove other arch (keep only gfx942) (#70) @Madduri, Rishi

## Deprecated

- Deprecate hip headers (#91) @Madduri, Rishi

---

**Contributors**: @Diptorup Deb, @Debasis Mandal, @Madduri, Rishi, @Clint

**Summary**: This release brings full prefill kernel support to ROCm, including single and batch prefill with paged and ragged KV cache. Major performance improvements include k128B swizzle mode and FP8 support. Significant infrastructure improvements include complete pytest coverage, improved build system, and updated to ROCm 7.1.1 / PyTorch 2.8.0.


# v0.2.5+rocm.1

## Added

- Decode feature chunking logic and shared mem optimization (#25) @Madduri, Rishi

# v0.2.5+rocm

The initial technical preview release of a ROCm port of FlashInfer. The release
has the port of the decode kernels and some infrastructure changes.

## Added

- A port of the decode kernels to HIP. (#38, #34) @Madduri, Rishi
- Add norm, page and rope to jit build infra (#46) @Madduri, Rishi
- Initial gpu interoperability interface for HIP/CUDA. (#22) @diptorupd
- Initial CDMA3 MFMA asbtractions (#62, #64, #68) @Madduri, Rishi
- Port build system to scikit-build-core. (#14) @diptorupd

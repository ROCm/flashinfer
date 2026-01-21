# v0.2.5+amd.2

## Added

- Make single prefill example script standalone (#104) @Debasis Mandal
- Run tests/test_non_contiguous_prefill.py on HIP (#96) @Debasis Mandal
- Run tests/test_activation.py on HIP (#98) @Debasis Mandal
- Add more PyTests from upstream to ROCm CI (#93) @Debasis Mandal
- Unify pytests for batch prefill for HIP and add to pyproject.toml to run on CI (#90) @Debasis Mandal
- Port over BatchPrefillWithPagedKVCacheDevice kernel to HIP (#63) @rtmadduri
- Batch prefill example script (#58) @Debasis Mandal
- Port over BatchPrefillWithRaggedKVCache to HIP (#50) @rtmadduri
- Add test_batch_prefill.cpp to HIP (#43) @rtmadduri
- PyTest script for single prefill (#48) @Debasis Mandal
- Add tests for LSE in single prefill example script (#46) @Debasis Mandal
- Enable FP8 support for Flashinfer ROCm decode kernels on CDNA3 (#40) @rtmadduri
- Single prefill example script (#36) @Debasis Mandal
- Support only gfx942 arch for ROCm (#45) @Debasis Mandal
- Enable JIT Installation for Prefill (#22) @rtmadduri
- Enable AOT Installation for Prefill (#21) @rtmadduri
- Adds a HIPified version of the SinglePrefillWithKVCacheDevice kernel (#31) @Diptorup Deb
- Copy CUDA versions of prefill into attention/generic (#30) @Diptorup Deb
- Port changes to vec_dtypes from prefill branch to amd-integration (#9) @Diptorup Deb
- Add bit-width specific masks (#7) @Diptorup Deb
- Feature/layout transformation function to transform from A matrix layout to B matrix layout for MFMA (#5) @Diptorup Deb

## Changed

- Updated changelog @Diptorup Deb
- Update REAMDE Wheel installation section (#112) @Diptorup Deb
- Change pyproject.toml to generate revised whl name (#113) @rtmadduri
- Update README with clearer usage guide (#72) @Diptorup Deb
- Change Swizzle mode from kLinear to k128B for prefill (#103) @Debasis Mandal
- Update logging for using JIT in absence of AOT (#94) @Debasis Mandal
- Improvements to Python packaging infrastructure (#76) @Diptorup Deb
- pytest Configuration Improvements for ROCm/HIP Testing (#65) @Diptorup Deb
- Refactor single prefill tests (#51) @Debasis Mandal
- Update CHANGELOG for v0.2.5+rocm.0.2 release (#34) @rtmadduri
- Decode feature chunking logic and shared mem optimization (#25) @rtmadduri
- Update hyperlinks in the TOC of README (#33) @Debasis Mandal
- Update build instructions for C++ tests in README.md (#27) @Debasis Mandal
- Incoporate gpu_iface changes needed for prefill (#14) @Diptorup Deb
- Update CMake infra to run HIP CXX tests using top-level cmake (#10) @Diptorup Deb
- Updates to permuted_smem.cuh (#8) @Diptorup Deb

## Removed

- Removes leftover src and all tvm bindings (#99) @Diptorup Deb
- Remove verbose CMake installation messages for editable JIT (#97) @Debasis Mandal
- Chore: Refactors the codebase to remove libflashinfer (#88) @Diptorup Deb
- Remove xfail markers about HIP support from pytests (#92) @Debasis Mandal
- Chore: Reduce tech debt by removing CUDA sections from generic/prefill.cuh (#87) @Diptorup Deb
- Removes the test_transpose_4x4_half_registers (#11) @Diptorup Deb

## Fixed

- Add custom ROCm version scheme to fix wheels version naming (#110) @Diptorup Deb
- Fix datatypes for HIP when using customized attention kernels (#111) @Debasis Mandal
- Fix partition-kv=True case and memory allocation issues in batch prefill (#89) @Debasis Mandal
- Fixes the single prefill kernel dispatch for HEAD_DIM_QK values gt. 64 (#86) @Diptorup Deb
- Fix/threadblock sync mdo (#62) @Diptorup Deb
- Fix batch prefill example script for ragged kv cache (#73) @Debasis Mandal
- Fixes to the single prefill dispatch for HIP devices (#64) @Diptorup Deb
- Skip failing C++ tests and fix mma_debug_utils (#59) @Diptorup Deb
- Fix Log-sum-exp (LSE) write back for single prefill kernels for CDNA3 (#42) @Diptorup Deb
- Implemented fix for the write_o_reg_gmem kernel (#39) @Diptorup Deb
- Fix few more leftover SPDX headers (#38) @Diptorup Deb
- Fix SPDX headers for AMD authored files (#37) @Diptorup Deb
- Improvements to the S-matrix (s_frag) materialization to LDS for debugging (#20) @Diptorup Deb
- fix the pipe at the end of a table (#19) @demandal25
- Fix some compiler warnings in Cxx unit tests (#13) @Diptorup Deb
- Adds debug utility functions for CDNA3 MMA ops. (#3) @Diptorup Deb
- Fixes fragment loading to properly pack 16b values into a 32b register (#2) @Diptorup Deb

## Maintenance

- Update rocm+torch versions in Dev Dockerfile (#118) @Debasis Mandal
- Update Dockerfile (#116) @rtmadduri
- Update Dockerfile.rocm_ci (#78) @Diptorup Deb
- Fix Fockerfile.rocm_ci (#77) @Diptorup Deb
- Removes hardcoded mamba env name from Docker image (#71) @Diptorup Deb
- Fix/dockerfile rocm.ci (#70) @Diptorup Deb
- Change base image to rocm/dev-ubuntu (#69) @Diptorup Deb
- Revert tar changes from Dockerfile (#68) @rtmadduri
- Update ROCm CI Dockerfile (#67) @rtmadduri
- Update devcontainer dockerfile to rocm7.0.2 (#24) @Debasis Mandal
- Enhance devcontainer Dockerfile for ROCm (#28) @Debasis Mandal
- Update/devcontainer dockerfile (#23) @Diptorup Deb
- Update Dockerfile.rocm_ci (#6) @Clint

## Miscellaneous

- Mark copied git repo as a safe directory (#114) @Diptorup Deb
- Disable test_custom_allreduce (#108) @rtmadduri
- Ignore nfs related files in git tracking (#75) @Debasis Mandal
- Update .clangd to support HIP specific flags (#74) @Debasis Mandal
- Add clangd configuration (#32) @Debasis Mandal
- Lint README.md with suggestions from markdown linter (#17) @demandal25

---

**Contributors**: @Diptorup Deb, @Debasis Mandal, @rtmadduri, @demandal25, @Clint

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

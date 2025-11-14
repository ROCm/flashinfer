# SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
#
# SPDX - License - Identifier : Apache 2.0

python -m pytest \
../tests/test_batch_decode_kernels_hip_fp8.py \
../tests/test_batch_decode_kernels_hip.py \
../tests/test_batch_decode_vllm.py \
../tests/test_logits_cap_hip.py \
../tests/test_non_contiguous_decode_hip.py \
../tests/test_norm_hip.py \
../tests/test_page.py \
../tests/test_rope.py \
../tests/test_single_prefill_kernels_hip.py \
../tests/test_sliding_window_hip.py \

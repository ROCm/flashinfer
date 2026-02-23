"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# HIP-specific version of test_rope.py
#
# This file includes tests from v0.2.5+amd.2 (known working on HIP) updated to v0.3.1 API.
#
# Included tests (from v0.2.5+amd.2):
# - test_rope: Tests basic apply_rope and apply_llama31_rope APIs
# - test_rope_pos_ids: Tests RoPE with position IDs (v0.2.5+amd.2 version without idtype parameter)
#
# Excluded tests:
# - test_rope_pos_ids with idtype parameter: New parameter added in v0.3.1, not yet tested on HIP
# - test_rope_cos_sin_cache: Numerical accuracy issues on HIP
# - test_mla_rope_quantize: MLA (Multi-Latent Attention) not supported on HIP

import pytest
import torch
from rope_reference import *

import flashinfer


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("qkv_len", [1, 4, 19, 204])
@pytest.mark.parametrize("num_qo_heads", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("offset", [0, 15, 99])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("llama_version", ["llama", "llama31"])
@pytest.mark.parametrize("partial_rotary_factor", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("inplace", [False, True])
def test_rope(
    batch_size,
    qkv_len,
    num_qo_heads,
    num_kv_heads,
    offset,
    head_dim,
    llama_version,
    partial_rotary_factor,
    inplace,
):
    rotary_dim = int(head_dim * partial_rotary_factor)
    nnz = batch_size * qkv_len
    qkv_packed = torch.randn(
        nnz,
        (num_qo_heads + 2 * num_kv_heads) * head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    k = qkv_packed[
        :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ].reshape(nnz, num_kv_heads, head_dim)
    indptr = torch.tensor(
        [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    )
    offsets = torch.full((batch_size,), offset, dtype=torch.int32, device="cuda:0")

    # reference implementation
    if llama_version == "llama":
        freqs_cis = precompute_freqs_cis(
            rotary_dim, qkv_len + offset, 10000.0, use_scaled=False, device="cuda:0"
        ).to("cuda:0")
    else:
        freqs_cis = precompute_freqs_cis(
            rotary_dim, qkv_len + offset, 5e5, use_scaled=True, device="cuda:0"
        ).to("cuda:0")
    q_rot_ref, k_rot_ref = apply_rotary_emb(
        q.reshape(batch_size, qkv_len, num_qo_heads, head_dim)[..., :rotary_dim],
        k.reshape(batch_size, qkv_len, num_kv_heads, head_dim)[..., :rotary_dim],
        freqs_cis[offset : offset + qkv_len],
    )
    q_pass_ref = q.reshape(batch_size, qkv_len, num_qo_heads, head_dim)[
        ..., rotary_dim:
    ]
    k_pass_ref = k.reshape(batch_size, qkv_len, num_kv_heads, head_dim)[
        ..., rotary_dim:
    ]
    q_rope_ref = torch.cat([q_rot_ref, q_pass_ref], dim=-1).reshape(
        nnz, num_qo_heads, head_dim
    )
    k_rope_ref = torch.cat([k_rot_ref, k_pass_ref], dim=-1).reshape(
        nnz, num_kv_heads, head_dim
    )

    # flashinfer implementation
    if llama_version == "llama":
        if inplace:
            flashinfer.apply_rope_inplace(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=True,
                rope_theta=1e4,
            )
            q_rope, k_rope = q, k
        else:
            q_rope, k_rope = flashinfer.apply_rope(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=True,
                rope_theta=1e4,
            )
    else:
        if inplace:
            flashinfer.apply_llama31_rope_inplace(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=True,
                rope_theta=5e5,
            )
            q_rope, k_rope = q, k
        else:
            q_rope, k_rope = flashinfer.apply_llama31_rope(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=True,
                rope_theta=5e5,
            )

    # compare
    torch.testing.assert_close(q_rope_ref, q_rope, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k_rope_ref, k_rope, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("qkv_len", [1, 4, 19, 204])
@pytest.mark.parametrize("num_qo_heads", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("offset", [0, 15, 99])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("llama_version", ["llama", "llama31"])
@pytest.mark.parametrize("partial_rotary_factor", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("interleave", [True, False])
def test_rope_pos_ids(
    batch_size,
    qkv_len,
    num_qo_heads,
    num_kv_heads,
    offset,
    head_dim,
    llama_version,
    partial_rotary_factor,
    inplace,
    interleave,
):
    rotary_dim = int(head_dim * partial_rotary_factor)
    nnz = batch_size * qkv_len
    qkv_packed = torch.randn(
        nnz,
        (num_qo_heads + 2 * num_kv_heads) * head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    k = qkv_packed[
        :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ].reshape(nnz, num_kv_heads, head_dim)
    indptr = torch.tensor(
        [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    )
    offsets = torch.full((batch_size,), offset, dtype=torch.int32, device="cuda:0")

    pos_ids = torch.cat(
        [
            torch.arange(offset, qkv_len + offset, dtype=torch.int32)
            for _ in range(batch_size)
        ]
    ).to("cuda:0")

    if llama_version == "llama":
        if inplace:
            q_clone, k_clone = q.clone(), k.clone()
            flashinfer.apply_rope_inplace(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=1e4,
            )
            q_rope, k_rope = q, k
            flashinfer.apply_rope_pos_ids_inplace(
                q_clone,
                k_clone,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=1e4,
            )
            q_rope_pos_ids, k_rope_pos_ids = q_clone, k_clone
        else:
            q_rope, k_rope = flashinfer.apply_rope(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=1e4,
            )

            q_rope_pos_ids, k_rope_pos_ids = flashinfer.apply_rope_pos_ids(
                q,
                k,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=1e4,
            )
    else:
        if inplace:
            q_clone, k_clone = q.clone(), k.clone()
            flashinfer.apply_llama31_rope_inplace(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=5e5,
            )
            q_rope, k_rope = q, k
            flashinfer.apply_llama31_rope_pos_ids_inplace(
                q_clone,
                k_clone,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=5e5,
            )
            q_rope_pos_ids, k_rope_pos_ids = q_clone, k_clone
        else:
            q_rope, k_rope = flashinfer.apply_llama31_rope(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=5e5,
            )

            q_rope_pos_ids, k_rope_pos_ids = flashinfer.apply_llama31_rope_pos_ids(
                q,
                k,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=5e5,
            )

    # compare
    torch.testing.assert_close(q_rope_pos_ids, q_rope, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k_rope_pos_ids, k_rope, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_rope(2, 1, 8, 8, 1, 128, "llama", 1.0, False)
    test_rope_pos_ids(2, 1, 8, 8, 1, 128, "llama", 1.0, False, True)

import torch

import flashinfer
import aiter
from aiter.ops import mha as aiter_mha_module

def verify_tensors(tensor1, tensor2):
    try:
        torch.testing.assert_close(tensor1, tensor2, rtol=1e-3, atol=1e-3)
        print("Tensors are close. Verification passed.")
    except AssertionError as e:
        print(e)
        print("Tensors are not close. Verification failed.")


def batch_prefill_with_paged_kv_cache_example(
    batch_size: int,
    kv_len: int,
    qo_len: int,
    page_size: int,
    num_kv_heads: int,
    num_qo_heads: int,
    head_dim: int,
    causal: bool,
    kv_layout: str,
    pos_encoding_mode: str,
    logits_soft_cap: float,
    return_lse: bool,
    contiguous_kv: bool,
    backend = "auto",
):
    """
    Run batch_prefill_with_paged_kv_cache and verify the output against single_prefill_with_kv_cache implementation
    """
    print("\nRunning configuration:")
    print(f"  batch_size={batch_size}")
    print(f"  kv_len={kv_len}")
    print(f"  qo_len={qo_len}")
    print(f"  page_size={page_size}")
    print(f"  num_kv_heads={num_kv_heads}")
    print(f"  num_qo_heads={num_qo_heads}")
    print(f"  head_dim={head_dim}")
    print(f"  causal={causal}")
    print(f"  kv_layout={kv_layout}")
    print(f"  pos_encoding_mode={pos_encoding_mode}")
    print(f"  logits_soft_cap={logits_soft_cap}")
    print(f"  return_lse={return_lse}")
    print(f"  contiguous_kv={contiguous_kv}")
    print(f"  backend={backend}")
    print("\n")

    if qo_len > kv_len and causal:
        raise ValueError("qo_len > kv_len and causal is not supported")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Create flattened query tensor: [batch_size * qo_len, num_qo_heads, head_dim]
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )

    # Create query indptr (index pointers for batch boundaries)
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len

    # Setup paged KV cache
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    # Create KV cache data
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:  # NHD
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]

    # Create KV cache data in non-contiguous memory if requested
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.half()
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.half()

    # Create KV indptr and indices
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    # Move tensors to GPU
    q_indptr_gpu = q_indptr_cpu.to("cuda:0")
    kv_indptr_gpu = kv_indptr_cpu.to("cuda:0")
    kv_indices_gpu = kv_indices_cpu.to("cuda:0")
    kv_last_page_len_gpu = kv_last_page_len_cpu.to("cuda:0")

    # Create workspace buffer and wrapper
    # NOTE: 512 MB workspace is needed for configurations with high GQA ratios
    # (num_qo_heads >> num_kv_heads) and small page sizes, which increase the
    # temporary buffer requirements for split-KV attention.
    o = None
    lse = None

    if backend == "aiter":
      
        k_data = kv_data[:, 0].contiguous()  # [total_num_pages, page_size, num_kv_heads, head_dim] (NHD)
        v_data = kv_data[:, 1].contiguous()  # [total_num_pages, page_size, num_kv_heads, head_dim] (NHD)
        o = aiter_mha_module.mha_batch_prefill_func(
                q=q, k=k_data, v=v_data,
                cu_seqlens_q=q_indptr_gpu, kv_indptr=kv_indptr_gpu, kv_page_indices=kv_indices_gpu,
                max_seqlen_q=qo_len, max_seqlen_k=kv_len,
                dropout_p=0.0, softmax_scale=None, causal=causal,
                window_size=(-1, -1), return_lse=False, return_attn_probs=False,
                kv_last_page_lens=kv_last_page_len_gpu,
            )
        if return_lse:
            return o[0], o[1]
        else:
            return o, None
    else:
        workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        # Create auxiliary data structures for batch prefill attention
        logits_soft_cap = logits_soft_cap if logits_soft_cap > 0 else None
        wrapper.plan(
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            kv_last_page_len_gpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        # Run batch prefill
        if return_lse:
            o, lse = wrapper.run(q, kv_data, return_lse=True)
            print(f"  FlashInfer batch output shape: {o.shape}, LSE shape: {lse.shape}")
            return o, lse
        else:
            o = wrapper.run(q, kv_data)
            print(f"  FlashInfer batch output shape: {o.shape}")
            return o, None

    # # Verify each sequence in the batch with single prefill
    # print("\n  Verifying individual sequences:")
    # all_passed = True

    # for i in range(batch_size):
    #     # Extract K and V for the i-th sequence from paged KV cache
    #     perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
    #     perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]

    #     qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]

    #     # Reconstruct full pages and last page for K
    #     ki = torch.cat(
    #         [
    #             kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 0]
    #             .permute(*perm_dims)
    #             .reshape(-1, num_kv_heads, head_dim),
    #             (
    #                 kv_data_fp32[
    #                     kv_indptr_cpu[i + 1] - 1, 0, :, : kv_last_page_len_cpu[i]
    #                 ]
    #                 if kv_layout == "HND"
    #                 else kv_data_fp32[
    #                     kv_indptr_cpu[i + 1] - 1, 0, : kv_last_page_len_cpu[i], :
    #                 ]
    #             )
    #             .permute(*perm_dims_last)
    #             .reshape(-1, num_kv_heads, head_dim),
    #         ],
    #         dim=0,
    #     ).half()

    #     # Reconstruct full pages and last page for V
    #     vi = torch.cat(
    #         [
    #             kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 1]
    #             .permute(*perm_dims)
    #             .reshape(-1, num_kv_heads, head_dim),
    #             (
    #                 kv_data_fp32[
    #                     kv_indptr_cpu[i + 1] - 1, 1, :, : kv_last_page_len_cpu[i]
    #                 ]
    #                 if kv_layout == "HND"
    #                 else kv_data_fp32[
    #                     kv_indptr_cpu[i + 1] - 1, 1, : kv_last_page_len_cpu[i], :
    #                 ]
    #             )
    #             .permute(*perm_dims_last)
    #             .reshape(-1, num_kv_heads, head_dim),
    #         ],
    #         dim=0,
    #     ).half()

    #     # Compare with single_prefill_with_kv_cache
    #     if return_lse:
    #         o_ref_i, lse_ref_i = (
    #             flashinfer.prefill.single_prefill_with_kv_cache_return_lse(
    #                 qi,
    #                 ki,
    #                 vi,
    #                 causal=causal,
    #                 kv_layout=kv_layout,
    #                 pos_encoding_mode=pos_encoding_mode,
    #                 logits_soft_cap=logits_soft_cap,
    #             )
    #         )
    #     else:
    #         o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
    #             qi,
    #             ki,
    #             vi,
    #             causal=causal,
    #             pos_encoding_mode=pos_encoding_mode,
    #             logits_soft_cap=logits_soft_cap,
    #         )

    #     # Extract the i-th sequence from batch output
    #     o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
    #     if return_lse:
    #         lse_i = lse[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]

    #     try:
    #         torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)
    #         if return_lse:
    #             torch.testing.assert_close(lse_i, lse_ref_i, rtol=1e-3, atol=1e-3)
    #         single_pass = True
    #         print(f"    Sequence {i}: ✓ PASS")
    #     except AssertionError:
    #         single_pass = False
    #         max_diff_o = (o_i - o_ref_i).abs().max().item()
    #         print(
    #             f"    Sequence {i}: ✗ FAIL vs single_prefill (max diff in output: {max_diff_o})"
    #         )
    #         if return_lse:
    #             max_diff_lse = (lse_i - lse_ref_i).abs().max().item()
    #             print(
    #                 f"    Sequence {i}: ✗ FAIL vs single_prefill (max diff in LSE: {max_diff_lse})"
    #             )
    #     all_passed = all_passed and single_pass

    # if all_passed:
    #     print("\n  ✓ ALL SEQUENCES PASSED")
    # else:
    #     print("\n  ✗ SOME SEQUENCES FAILED")

if __name__ == "__main__":
    print("=" * 60)
    print("FlashInfer Batch Prefill Example")
    print("=" * 60)

    # Basic test with small batch
    # Seed RNG so both backends get identical input tensors
    aiter_output, aiter_lse = batch_prefill_with_paged_kv_cache_example(
        4, 128, 128, 16, 8, 8, 64, False, "NHD", "NONE", 0.0, False, True, backend="aiter"
    )
    
    flashinfer_output, flashinfer_lse = batch_prefill_with_paged_kv_cache_example(
        4, 128, 128, 16, 8, 8, 64, False, "NHD", "NONE", 0.0, False, True, backend="flashinfer"
    )

    verify_tensors(aiter_output, flashinfer_output)

    if aiter_lse is not None and flashinfer_lse is not None:
        verify_tensors(aiter_lse, flashinfer_lse)

import torch
import flashinfer
import time

import triton

TOTAL_TIME = 0.0


            
def test_batch_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
    bench=False
):
    if qo_len > kv_len and causal:
        print("qo_len > kv_len and causal is not supported. Returning.")
        return

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
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

    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_indptr_cpu.to(0)
    kv_indices_gpu = kv_indices_cpu.to(0)
    kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
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

    print("Running batch prefill with paged kv cache...")
    if return_lse:
        o, _ = wrapper.run(q, kv_data, return_lse=return_lse)
    else:
        o = wrapper.run(q, kv_data, return_lse=return_lse)

    if bench:
        
    # global TOTAL_TIME
    # start_time = time.time()
    # if return_lse:
        ms = triton.testing.do_bench(lambda: wrapper.run(q, kv_data), warmup=100, rep=1000)
    # else:
    #     o = wrapper.run(q, kv_data)
    # end_time = time.time()
    # TOTAL_TIME += end_time - start_time

        def flops(ms):
            if causal:
                return (
                    batch_size * qo_len * kv_len * num_qo_heads * head_dim * 2 / ms / 1e9
                )
            else:
                return (
                    batch_size * qo_len * kv_len * num_qo_heads * head_dim * 4 / ms / 1e9
                )

        print(
            f"bench_batch_paged_prefill: fa2-template: {flops(ms):.3f} TFLOPs/s "
        )


if __name__ == "__main__":

    # print("Running this script: ROCm")

    num_iters = 1
    # print("\n Running benchmark...")

    # Partition kv = false
    # for _ in range(num_iters):
    #     test_batch_prefill_with_paged_kv_cache(
    #         batch_size = 128,
    #         kv_len = 2048,
    #         qo_len = 577,
    #         page_size = 16,
    #         num_kv_heads = 4,
    #         num_qo_heads = 32,
    #         head_dim = 256,
    #         causal = True,
    #         kv_layout = "NHD",
    #         pos_encoding_mode = "NONE",
    #         use_cuda_graph = False,
    #         logits_soft_cap = 8.0,
    #         return_lse = True,
    #         contiguous_kv = True,
    #     )

    # Partition kv = true
    for _ in range(num_iters):
        test_batch_prefill_with_paged_kv_cache(
            batch_size = 12,
            kv_len = 512,
            qo_len = 37,
            page_size = 5,
            num_kv_heads = 4,
            num_qo_heads = 32,
            head_dim = 128,
            causal = False,
            kv_layout = "NHD",
            pos_encoding_mode = "NONE",
            use_cuda_graph = False,
            logits_soft_cap = 0.0,
            return_lse = True,
            contiguous_kv = True,
            bench=False
        )

    # print(f"Average time per iteration: {TOTAL_TIME / num_iters * 1000:.2f} ms")


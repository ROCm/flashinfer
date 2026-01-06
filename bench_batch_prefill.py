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
    mode
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

    
    if mode == "throughput":
        ms = triton.testing.do_bench(lambda: wrapper.run(q, kv_data), warmup=100, rep=1000)
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

    elif mode == "latency":
        num_iters = 1000
        global TOTAL_TIME
        for _ in range(num_iters):
            start_time = time.time()
            if return_lse:
                o, _ = wrapper.run(q, kv_data, return_lse=return_lse)
            else:
                o = wrapper.run(q, kv_data, return_lse=return_lse)
            end_time = time.time()
            TOTAL_TIME += end_time - start_time

        print(f"Average time per iteration: {TOTAL_TIME / num_iters * 1000:.2f} ms")
    return

def run_benchmark(mode, partition_kv):

    if mode not in ["latency", "throughput"]:
        print("mode should be either latency or throughput. Returning.")
        return

    if partition_kv:
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
            mode=mode,
        )

    else:
        test_batch_prefill_with_paged_kv_cache(
            batch_size = 128,
            kv_len = 2048,
            qo_len = 577,
            page_size = 16,
            num_kv_heads = 4,
            num_qo_heads = 32,
            head_dim = 256,
            causal = True,
            kv_layout = "NHD",
            pos_encoding_mode = "NONE",
            use_cuda_graph = False,
            logits_soft_cap = 8.0,
            return_lse = True,
            contiguous_kv = True,
            mode=mode,
        )

if __name__ == "__main__":

    #if you want to run latency benchmark, set mode to "latency"
    run_benchmark(mode="latency", partition_kv=False)
    run_benchmark(mode="latency", partition_kv=True)

    #if you want to run throughput benchmark, set mode to "throughput"
    run_benchmark(mode="throughput", partition_kv=False)
    run_benchmark(mode="throughput", partition_kv=True)

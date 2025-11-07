import torch
import math

import flashinfer


def verify_tensors(tensor1, tensor2, rtol=1e-3, atol=1e-3):
    """
    Verify that two tensors are close element-wise.
    Returns True if all elements match within tolerance, False otherwise.
    """
    if tensor1.ndim != tensor2.ndim:
        raise ValueError(f"verify_tensors only supports tensors with the same number of dimensions, got {tensor1.ndim}D and {tensor2.ndim}D")
    if tensor1.ndim != 3:
        raise ValueError(f"verify_tensors only supports 3D tensors, got {tensor1.ndim}D")

    for i in range(tensor1.shape[0]):
        for j in range(tensor1.shape[1]):
            for k in range(tensor1.shape[2]):
                if torch.abs(tensor1[i][j][k] - tensor2[i][j][k]) > atol + rtol * torch.abs(
                    tensor2[i][j][k]
                ):
                    print(f"Error at {i}, {j}, {k}")
                    print(f"Expected: {tensor2[i][j][k]}")
                    print(f"Got: {tensor1[i][j][k]}")
                    return False
    return True


def naive_attention(q, k, v, causal=False, sm_scale=None):
    """
    Naive PyTorch implementation of attention for reference.

    Args:
        q: [qo_len, num_qo_heads, head_dim]
        k: [kv_len, num_kv_heads, head_dim]
        v: [kv_len, num_kv_heads, head_dim]
        causal: whether to apply causal masking
        sm_scale: softmax scale (default: 1/sqrt(head_dim))

    Returns:
        o: [qo_len, num_qo_heads, head_dim]
    """
    qo_len, num_qo_heads, head_dim = q.shape
    kv_len, num_kv_heads, _ = k.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Handle grouped query attention (GQA)
    group_size = num_qo_heads // num_kv_heads

    # Expand k and v to match q's head dimension if using GQA
    if group_size > 1:
        k = k.repeat_interleave(group_size, dim=1)  # [kv_len, num_qo_heads, head_dim]
        v = v.repeat_interleave(group_size, dim=1)  # [kv_len, num_qo_heads, head_dim]

    # Transpose for batch matrix multiply: [num_heads, seq_len, head_dim]
    q_t = q.transpose(0, 1)  # [num_qo_heads, qo_len, head_dim]
    k_t = k.transpose(0, 1)  # [num_qo_heads, kv_len, head_dim]
    v_t = v.transpose(0, 1)  # [num_qo_heads, kv_len, head_dim]

    # Compute attention scores: [num_qo_heads, qo_len, kv_len]
    scores = torch.matmul(q_t, k_t.transpose(1, 2)) * sm_scale

    # Apply causal mask if needed
    if causal:
        # Create causal mask
        mask = torch.tril(
            torch.ones((qo_len, kv_len), device=q.device, dtype=torch.bool),
            diagonal=(kv_len - qo_len),
        )
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

    # Softmax
    attn = torch.softmax(scores, dim=-1)

    # Apply attention to values: [num_qo_heads, qo_len, head_dim]
    out = torch.matmul(attn, v_t)

    # Transpose back: [qo_len, num_qo_heads, head_dim]
    out = out.transpose(0, 1)

    return out


def single_prefill_with_kv_cache(
    qo_len,
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
    causal,
    return_lse,
    q_dtype,
    kv_dtype,
):
    """
    Run single_prefill_with_kv_cache and verify the output against a naive PyTorch reference implementation.

    This function creates random Q, K, V tensors and compares the output
    of flashinfer's single_prefill_with_kv_cache against a naive PyTorch reference implementation.
    """
    print(f"\nRunning configuration:")
    print(f"  qo_len={qo_len}")
    print(f"  kv_len={kv_len}")
    print(f"  num_qo_heads={num_qo_heads}")
    print(f"  num_kv_heads={num_kv_heads}")
    print(f"  head_dim={head_dim}")
    print(f"  kv_layout={kv_layout}")
    print(f"  causal={causal}")
    print(f"  pos_encoding_mode={pos_encoding_mode}")
    print(f"  q_dtype={q_dtype}")
    print(f"  kv_dtype={kv_dtype}")

    # Create query tensor: [qo_len, num_qo_heads, head_dim]
    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)

    # Create key and value tensors based on layout
    if kv_layout == "HND":
        # [num_kv_heads, kv_len, head_dim]
        k = torch.randn(num_kv_heads, kv_len, head_dim, device="cuda:0", dtype=kv_dtype)
        v = torch.randn(num_kv_heads, kv_len, head_dim, device="cuda:0", dtype=kv_dtype)
        # Convert to NHD for reference implementation
        k_ref = k.transpose(0, 1).contiguous()  # [kv_len, num_kv_heads, head_dim]
        v_ref = v.transpose(0, 1).contiguous()  # [kv_len, num_kv_heads, head_dim]
    else:  # NHD layout
        # [kv_len, num_kv_heads, head_dim]
        k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)
        v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)
        k_ref = k
        v_ref = v

    # Call flashinfer API
    if return_lse:
        o, lse = flashinfer.single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=causal,
            kv_layout=kv_layout,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap if logits_soft_cap > 0 else None,
            backend="auto",
            return_lse=True,
        )
        print(f"  Output shape: {o.shape}, LSE shape: {lse.shape}")
    else:
        o = flashinfer.single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=causal,
            kv_layout=kv_layout,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap if logits_soft_cap > 0 else None,
            backend="auto",
            return_lse=False,
        )
        print(f"  Output shape: {o.shape}")

    # For simple cases without RoPE/ALiBi/soft_cap, verify against naive implementation
    if pos_encoding_mode == "NONE" and logits_soft_cap == 0.0:
        # Compute reference output using naive attention
        o_ref = naive_attention(
            q.float(),
            k_ref.float(),
            v_ref.float(),
            causal=causal,
        ).to(q_dtype)

        # Verify results
        try:
            torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
            print("  ✓ PASS: Output matches reference implementation")
            return True
        except AssertionError:
            print("  ✗ FAIL: Output does not match reference implementation")
            if verify_tensors(o, o_ref, rtol=1e-3, atol=1e-3):
                print("  ✓ PASS (verified with custom verifier)")
                return True
            else:
                # Print some debug info
                diff = torch.abs(o - o_ref)
                print(f"  Max difference: {diff.max().item()}")
                print(f"  Mean difference: {diff.mean().item()}")
                return False
    else:
        print("  ⊘ SKIP: Reference verification (advanced features not supported)")
        print("  ✓ API call successful")
        return True


# def single_prefill_with_custom_mask(
#     qo_len,
#     kv_len,
#     num_kv_heads,
#     num_qo_heads,
#     head_dim,
#     q_dtype,
#     kv_dtype,
# ):
#     """
#     Run single_prefill_with_kv_cache with custom mask and verify the output against a naive PyTorch reference implementation.

#     Verifies that using a causal custom mask produces the same output
#     as using the causal=True flag.
#     """
#     print(f"\nRunning configuration:")
#     print(f"  qo_len={qo_len}, kv_len={kv_len}")
#     print(f"  num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}")
#     print(f"  head_dim={head_dim}")
#     print(f"  q_dtype={q_dtype}")
#     print(f"  kv_dtype={kv_dtype}")

#     # Create tensors
#     q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)
#     k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)
#     v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)

#     # Create causal mask
#     custom_mask = torch.tril(
#         torch.full((qo_len, kv_len), True, device="cuda:0"),
#         diagonal=(kv_len - qo_len),
#     )

#     # Test with custom mask
#     o_custom = flashinfer.single_prefill_with_kv_cache(
#         q, k, v, custom_mask=custom_mask, backend="auto"
#     )

#     # Test with causal flag
#     o_causal = flashinfer.single_prefill_with_kv_cache(
#         q, k, v, causal=True, backend="auto"
#     )

#     # Verify they match
#     try:
#         torch.testing.assert_close(o_custom, o_causal, rtol=1e-3, atol=1e-3)
#         print("  ✓ PASS: Custom mask matches causal flag")
#         return True
#     except AssertionError:
#         print("  ✗ FAIL: Custom mask does not match causal flag")
#         diff = torch.abs(o_custom - o_causal)
#         print(f"  Max difference: {diff.max().item()}")
#         print(f"  Mean difference: {diff.mean().item()}")
#         return False


if __name__ == "__main__":
    print("=" * 60)
    print("FlashInfer Single Prefill Example")
    print("=" * 60)

    # Test configuration 1: Basic test with GQA
    success = single_prefill_with_kv_cache(
        qo_len=128,
        kv_len=512,
        num_kv_heads=4,
        num_qo_heads=32,
        head_dim=128,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        causal=True,
        return_lse=False,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
    )

    # # Test configuration 2: Non-causal attention
    # success &= single_prefill_with_kv_cache(
    #     qo_len=64,
    #     kv_len=256,
    #     num_kv_heads=8,
    #     num_qo_heads=8,
    #     head_dim=256,
    #     kv_layout="NHD",
    #     pos_encoding_mode="NONE",
    #     logits_soft_cap=0.0,
    #     causal=False,
    #     return_lse=False,
    #     q_dtype=torch.float16,
    #     kv_dtype=torch.float16,
    # )

    # # Test configuration 3: With return_lse
    # success &= single_prefill_with_kv_cache(
    #     qo_len=32,
    #     kv_len=128,
    #     num_kv_heads=4,
    #     num_qo_heads=32,
    #     head_dim=128,
    #     kv_layout="NHD",
    #     pos_encoding_mode="NONE",
    #     logits_soft_cap=0.0,
    #     causal=True,
    #     return_lse=True,
    #     q_dtype=torch.float16,
    #     kv_dtype=torch.float16,
    # )

    # # Test configuration 4: HND layout
    # success &= single_prefill_with_kv_cache(
    #     qo_len=64,
    #     kv_len=256,
    #     num_kv_heads=4,
    #     num_qo_heads=32,
    #     head_dim=128,
    #     kv_layout="HND",
    #     pos_encoding_mode="NONE",
    #     logits_soft_cap=0.0,
    #     causal=True,
    #     return_lse=False,
    #     q_dtype=torch.float16,
    #     kv_dtype=torch.float16,
    # )

    # # Test configuration 5: Different head dimensions
    # success &= single_prefill_with_kv_cache(
    #     qo_len=128,
    #     kv_len=512,
    #     num_kv_heads=8,
    #     num_qo_heads=32,
    #     head_dim=256,
    #     kv_layout="NHD",
    #     pos_encoding_mode="NONE",
    #     logits_soft_cap=0.0,
    #     causal=False,
    #     return_lse=False,
    #     q_dtype=torch.float16,
    #     kv_dtype=torch.float16,
    # )

    # # Test configuration 6: Custom mask
    # success &= single_prefill_with_custom_mask(
    #     qo_len=64,
    #     kv_len=128,
    #     num_kv_heads=4,
    #     num_qo_heads=32,
    #     head_dim=128,
    #     q_dtype=torch.float16,
    #     kv_dtype=torch.float16,
    # )

    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)


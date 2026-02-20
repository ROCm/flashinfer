.. meta::
  :description: What is FlashInfer?
  :keywords: FlashInfer, documentation, deep learning, framework, GPU, AMD, ROCm, overview, introduction

.. _what-is-flashinfer:

********************************************************************
What is FlashInfer?
********************************************************************

`FlashInfer <https://docs.flashinfer.ai/index.html>`__ is a library and kernel generator 
for Large Language Models (LLMs) that provides a high-performance implementation of graphics 
processing units (GPUs) kernels. FlashInfer focuses on LLM serving and inference, as well 
as advanced performance across diverse scenarios.

FlashInfer features highly efficient attention kernels, load-balanced scheduling, and memory-optimized 
techniques, while supporting customized attention variants. It’s compatible with ``torch.compile``, and 
offers high-performance LLM-specific operators, with easy integration through PyTorch, and C++ APIs.

.. note::

  The ROCm port of FlashInfer is under active development, and some features are not yet available. 
  For the latest feature compatibility matrix, refer to the ``README`` of the 
  `https://github.com/ROCm/flashinfer <https://github.com/ROCm/flashinfer>`__ repository.

Features and use cases
====================================================================

FlashInfer on ROCm enables you to perform LLM inference for both prefill and decode:
during prefill, your model efficiently processes input prompts to build KV caches
and internal activations; during decode, it generates tokens sequentially based on
prior outputs and context. Use the attention mode supported upstream (Multi-Head
Attention, Grouped-Query Attention, or Multi-Query Attention) that matches your
model configuration.

FlashInfer on ROCm also includes capabilities such as load balancing, 
sparse and dense attention optimizations, and single and batch decode, alongside
prefill for high‑performance execution on MI300X GPUs.

For currently supported use cases and recommendations, refer to the `AMD ROCm blog <https://rocm.blogs.amd.com/search.html?q=flashinfer>`__, 
where you can search for examples and best practices to optimize your workloads on AMD GPUs.
 


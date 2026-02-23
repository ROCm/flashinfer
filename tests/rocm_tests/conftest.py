# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from flashinfer.hip_utils import get_available_gpu_count


def pytest_configure(config):
    """Pin each pytest-xdist worker to a dedicated physical GPU.

    When running under xdist (pytest -n <N>), each worker receives a unique
    PYTEST_XDIST_WORKER env var (e.g. "gw0", "gw1", ...).  We map the numeric
    suffix directly to a GPU ordinal and restrict the worker process to that
    device by setting HIP_VISIBLE_DEVICES (and CUDA_VISIBLE_DEVICES for
    compatibility).  Tests continue to address the device as index 0 and are
    transparently routed to their assigned physical GPU — no test changes
    required.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is None or not worker_id.startswith("gw"):
        return

    gpu_index = int(worker_id[2:])
    n_gpus = get_available_gpu_count()
    if gpu_index >= n_gpus:
        raise RuntimeError(
            f"xdist worker {worker_id} requires GPU {gpu_index} but only "
            f"{n_gpus} GPU(s) are available. Pass a lower value to -n."
        )
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


def pytest_xdist_auto_num_workers(config):
    """Return the number of available AMD GPUs for 'pytest -n auto'.

    Raises RuntimeError if no GPUs are detected so the failure is explicit
    rather than silently falling back to single-process execution.
    """
    n = get_available_gpu_count()
    if n == 0:
        raise RuntimeError(
            "pytest -n auto: no AMD GPUs detected (torch.cuda.device_count() == 0). "
            "Check HIP_VISIBLE_DEVICES or ROCm installation."
        )
    return n

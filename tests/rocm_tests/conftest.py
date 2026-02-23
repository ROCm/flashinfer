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

from flashinfer.hip_utils import get_available_gpu_count


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

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

from flashinfer.hip_utils import get_physical_card_device_indices

# pytest_xdist_auto_num_workers is only available when pytest-xdist is installed
# and loaded (i.e. when -n / --dist is passed). Guard the definition so that
# running without xdist does not produce an "unknown hook" PluginValidationError.
try:
    import xdist  # noqa: F401

    def pytest_xdist_auto_num_workers(config):
        """Return the recommended worker count for 'pytest -n auto'.

        Halved from one-per-physical-card. One worker per physical card
        still produces sporadic HSA / HIPBLAS failures across the wider
        ROCm test suite (rope, single_prefill, logits_cap) under residual
        concurrent CPX load even with --reruns 2; halving eliminates them
        reliably at a ~1.6× wall-time cost. Users who want every device
        used can pass an explicit -n N.
        """
        n_physical = len(get_physical_card_device_indices())
        if n_physical == 0:
            raise RuntimeError(
                "pytest -n auto: no FlashInfer-supported AMD GPUs detected. "
                "Check HIP_VISIBLE_DEVICES or ROCm installation."
            )
        return max(1, n_physical // 2)

except ImportError:
    pass

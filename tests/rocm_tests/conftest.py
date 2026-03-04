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

from flashinfer.hip_utils import get_supported_device_indices


# pytest_xdist_auto_num_workers is only available when pytest-xdist is installed
# and loaded (i.e. when -n / --dist is passed). Guard the definition so that
# running without xdist does not produce an "unknown hook" PluginValidationError.
try:
    import xdist  # noqa: F401

    def pytest_xdist_auto_num_workers(config):
        """Return the number of FlashInfer-supported AMD GPUs for 'pytest -n auto'.

        Only devices whose architecture is in FLASHINFER_SUPPORTED_ROCM_ARCHS are
        counted so that xdist does not spawn workers for unsupported integrated GPUs.
        Raises RuntimeError if no supported GPUs are detected so the failure is
        explicit rather than silently falling back to single-process execution.
        """
        n = len(get_supported_device_indices())
        if n == 0:
            raise RuntimeError(
                "pytest -n auto: no FlashInfer-supported AMD GPUs detected. "
                "Check HIP_VISIBLE_DEVICES or ROCm installation."
            )
        return n

except ImportError:
    pass

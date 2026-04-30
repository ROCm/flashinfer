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

import functools
import json
import subprocess

from flashinfer.hip_utils import get_supported_device_indices


@functools.cache
def get_physical_card_device_indices() -> tuple:
    """Return one supported device index per physical AMD card.

    On CDNA3 CPX systems each physical card exposes 4 logical XCD-sized
    devices that share the card's HBM. Running one xdist worker per logical
    device causes intermittent HSA hardware exceptions when multiple workers
    on the same physical card concurrently allocate large tensors. We pick
    one "primary" index per card (the one that reports the full card capacity
    via rocm-smi) so xdist workers spread one-per-card.

    On non-CPX systems all supported devices report identical VRAM and the
    helper returns them unchanged.

    Falls back to all supported devices if rocm-smi is unavailable or its
    output cannot be parsed.
    """
    supported = get_supported_device_indices()
    if not supported:
        return ()

    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return supported
        data = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return supported

    vram_by_idx: dict[int, int] = {}
    for key, val in data.items():
        if not key.startswith("card"):
            continue
        try:
            idx = int(key[4:])
        except ValueError:
            continue
        total = val.get("VRAM Total Memory (B)") if isinstance(val, dict) else None
        if total is None:
            continue
        try:
            vram_by_idx[idx] = int(total)
        except (TypeError, ValueError):
            continue

    if not vram_by_idx:
        return supported

    max_vram = max(vram_by_idx.values())
    # A "primary" index is one whose card-capacity row matches the largest
    # capacity reported by any device. On CPX systems the secondary siblings
    # report ~25% of the primary's capacity and are filtered out.
    threshold = int(max_vram * 0.95)
    primary = tuple(
        idx for idx in supported if vram_by_idx.get(idx, 0) >= threshold
    )
    return primary or supported


# pytest_xdist_auto_num_workers is only available when pytest-xdist is installed
# and loaded (i.e. when -n / --dist is passed). Guard the definition so that
# running without xdist does not produce an "unknown hook" PluginValidationError.
try:
    import xdist  # noqa: F401

    def pytest_xdist_auto_num_workers(config):
        """Return the number of physical AMD cards for 'pytest -n auto'.

        On CPX systems this is the count of physical 192GB cards rather than
        the count of logical XCD devices. Pinning one worker per physical
        card avoids HSA hardware exceptions caused by multiple xdist workers
        concurrently hammering the same card's HBM. See
        get_physical_card_device_indices() for the detection rule.
        """
        n = len(get_physical_card_device_indices())
        if n == 0:
            raise RuntimeError(
                "pytest -n auto: no FlashInfer-supported AMD GPUs detected. "
                "Check HIP_VISIBLE_DEVICES or ROCm installation."
            )
        return n

except ImportError:
    pass

"""
Copyright (c) 2025 by FlashInfer team.
Copyright (c) 2025 by AMD ROCm team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Global compilation context management for FlashInfer on ROCm.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)


class CompilationContext:
    """Manages ROCm compilation targets and generates hipcc flags."""

    def __init__(self):
        self.TARGET_ROCM_ARCHS = set()

        if "FLASHINFER_ROCM_ARCH_LIST" in os.environ:
            # Parse from env var: "gfx942,gfx90a" or "gfx942"
            arch_list = os.environ["FLASHINFER_ROCM_ARCH_LIST"]
            for arch in arch_list.replace(",", " ").split():
                arch = arch.strip()
                if arch:
                    self.TARGET_ROCM_ARCHS.add(arch)
        else:
            # Try to auto-detect from system
            try:
                for device in range(torch.cuda.device_count()):
                    # Get gcnArchName which returns something like "gfx942:sramecc+:xnack-"
                    props = torch.cuda.get_device_properties(device)
                    if hasattr(props, "gcnArchName"):
                        # Extract base gfx architecture (e.g., "gfx942" from "gfx942:sramecc+:xnack-")
                        arch_name = props.gcnArchName.split(":")[0]
                        self.TARGET_ROCM_ARCHS.add(arch_name)
            except Exception as e:
                logger.warning(f"Failed to auto-detect ROCm device architectures: {e}")
                # Default fallback
                self.TARGET_ROCM_ARCHS.add("gfx942")

        if not self.TARGET_ROCM_ARCHS:
            # Ultimate fallback
            self.TARGET_ROCM_ARCHS.add("gfx942")
            logger.warning("No ROCm architectures detected, defaulting to gfx942")

    def get_hipcc_flags_list(self) -> list[str]:
        """
        Generate hipcc compiler flags for target architectures.

        Returns:
            List of flags like ["--offload-arch=gfx942", "--offload-arch=gfx90a"]
        """
        flags = [f"--offload-arch={arch}" for arch in sorted(self.TARGET_ROCM_ARCHS)]
        return flags

    def get_target_archs(self) -> set[str]:
        """
        Get the set of target architectures.

        Returns:
            Set of architecture strings like {"gfx942", "gfx90a"}
        """
        return self.TARGET_ROCM_ARCHS.copy()

    def has_arch(self, arch: str) -> bool:
        """
        Check if a specific architecture is targeted.

        Args:
            arch: Architecture string like "gfx942"

        Returns:
            True if the architecture is in the target set
        """
        return arch in self.TARGET_ROCM_ARCHS

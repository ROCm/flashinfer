"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# HIP backend - import available functions from pytorch_hip
from .pytorch_hip import gen_batch_decode_module as gen_batch_decode_module
from .pytorch_hip import gen_batch_prefill_module as gen_batch_prefill_module
from .pytorch_hip import (
    gen_customize_batch_decode_module as gen_customize_batch_decode_module,
)
from .pytorch_hip import (
    gen_customize_batch_prefill_module as gen_customize_batch_prefill_module,
)
from .pytorch_hip import (
    gen_customize_single_decode_module as gen_customize_single_decode_module,
)
from .pytorch_hip import (
    gen_customize_single_prefill_module as gen_customize_single_prefill_module,
)
from .pytorch_hip import gen_single_decode_module as gen_single_decode_module
from .pytorch_hip import gen_single_prefill_module as gen_single_prefill_module
from .pytorch_hip import get_batch_decode_uri as get_batch_decode_uri
from .pytorch_hip import get_batch_prefill_uri as get_batch_prefill_uri
from .pytorch_hip import get_single_decode_uri as get_single_decode_uri
from .pytorch_hip import get_single_prefill_uri as get_single_prefill_uri


# Stubs for functions not available in HIP backend
def _not_implemented(*args, **kwargs):
    raise NotImplementedError("This function is not supported on HIP/ROCm backend")


gen_batch_decode_mla_module = _not_implemented
gen_batch_mla_module = _not_implemented
gen_pod_module = _not_implemented
get_batch_decode_mla_uri = _not_implemented
get_batch_mla_uri = _not_implemented
get_pod_uri = _not_implemented

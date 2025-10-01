#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

// GPU interface headers
#include <flashinfer/attention/generic/pos_enc.cuh>
#include <flashinfer/attention/generic/prefill.cuh>
#include <flashinfer/attention/generic/variant_helper.cuh>
#include <flashinfer/attention/generic/variants.cuh>
#include <gpu_iface/fastdiv.cuh>
#include <gpu_iface/layout.cuh>
#include <gpu_iface/math_ops.hpp>
#include <gpu_iface/platform.hpp>
#include <gpu_iface/utils.cuh>

namespace flashinfer {

// Parameter struct for SinglePrefill
template <typename DTypeOs, typename IdTypes>
struct SinglePrefillParams {
  using DTypeQ = half;
  using DTypeKV = half;
  using DTypeO = DTypeOs;
  using IdType = IdTypes;

  half* q;
  half* k;
  half* v;
  DTypeO* o;
  float* lse;
  uint_fastdiv group_size;

  uint8_t* maybe_custom_mask;
  float* maybe_alibi_slopes;
  double logits_soft_cap;
  double sm_scale;
  double rope_rcp_scale;
  double rope_rcp_theta;

  uint32_t qo_len;
  uint32_t kv_len;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t k_stride_n;
  uint32_t k_stride_h;
  uint32_t v_stride_n;
  uint32_t v_stride_h;
  uint32_t head_dim;
  int32_t window_left;

  bool partition_kv;

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return qo_len;
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_len;
  }
};

}  // namespace flashinfer

// CPU reference implementation for validation
namespace reference {

template <typename T>
std::vector<T> single_mha(const std::vector<T>& q, const std::vector<T>& k, const std::vector<T>& v,
                          size_t qo_len, size_t kv_len, size_t num_qo_heads, size_t num_kv_heads,
                          size_t head_dim, bool causal, flashinfer::QKVLayout kv_layout,
                          flashinfer::PosEncodingMode pos_encoding_mode, float rope_scale = 1.0f,
                          float rope_theta = 10000.0f) {
  float sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::vector<T> o(qo_len * num_qo_heads * head_dim, static_cast<T>(0.0f));
  std::vector<float> att(kv_len);
  size_t group_size = num_qo_heads / num_kv_heads;

  for (size_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
    size_t kv_head_idx = qo_head_idx / group_size;

    for (size_t q_idx = 0; q_idx < qo_len; ++q_idx) {
      // 1. Compute attention scores
      float max_val = -5e4f;

      for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
        if (causal && kv_idx > kv_len + q_idx - qo_len) {
          att[kv_idx] = -5e4f;
          continue;
        }

        // Compute dot product between Q and K
        float score = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
          float q_val = 0.0f;
          float k_val = 0.0f;

          // Get Q value - always NHD layout
          size_t q_offset = q_idx * num_qo_heads * head_dim + qo_head_idx * head_dim + d;
          q_val = static_cast<float>(q[q_offset]);

          // Get K value - depends on layout
          if (kv_layout == flashinfer::QKVLayout::kNHD) {
            size_t k_offset = kv_idx * num_kv_heads * head_dim + kv_head_idx * head_dim + d;
            k_val = static_cast<float>(k[k_offset]);
          } else {
            size_t k_offset = kv_head_idx * kv_len * head_dim + kv_idx * head_dim + d;
            k_val = static_cast<float>(k[k_offset]);
          }

          score += q_val * k_val;
        }
        score *= sm_scale;

        att[kv_idx] = score;
        max_val = std::max(max_val, score);
      }

      // 2. Apply softmax
      float sum_exp = 0.0f;
      for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
        if (causal && kv_idx > kv_len + q_idx - qo_len) {
          att[kv_idx] = 0.0f;
        } else {
          att[kv_idx] = std::exp(att[kv_idx] - max_val);
          sum_exp += att[kv_idx];
        }
      }

      // Normalize
      for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
        if (sum_exp > 0.0f) {
          att[kv_idx] /= sum_exp;
        }
      }

      // 3. Compute weighted sum of values
      for (size_t d = 0; d < head_dim; ++d) {
        float weighted_sum = 0.0f;

        for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
          float v_val = 0.0f;

          // Get V value - depends on layout
          if (kv_layout == flashinfer::QKVLayout::kNHD) {
            size_t v_offset = kv_idx * num_kv_heads * head_dim + kv_head_idx * head_dim + d;
            v_val = static_cast<float>(v[v_offset]);
          } else {
            size_t v_offset = kv_head_idx * kv_len * head_dim + kv_idx * head_dim + d;
            v_val = static_cast<float>(v[v_offset]);
          }

          weighted_sum += att[kv_idx] * v_val;
        }

        // Store result in output
        size_t o_offset = q_idx * num_qo_heads * head_dim + qo_head_idx * head_dim + d;
        o[o_offset] = static_cast<T>(weighted_sum);
      }
    }
  }

  return o;
}

}  // namespace reference

// Helper function to generate random data (without Thrust)
void generate_random_data(half* data, size_t size, float min_val = -1.0f, float max_val = 1.0f) {
  std::vector<half> host_data(size);
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(min_val, max_val);

  for (size_t i = 0; i < size; ++i) {
    host_data[i] = static_cast<half>(dist(rng));
  }

  // Copy to device
  FI_GPU_CALL(gpuMemcpy(data, host_data.data(), size * sizeof(half), gpuMemcpyHostToDevice));
}

// Function to validate GPU results against CPU reference (simplified)
bool validate_results(const half* gpu_output, size_t gpu_size, const std::vector<half>& cpu_output,
                      float rtol = 1e-3f, float atol = 1e-3f) {
  if (gpu_size != cpu_output.size()) {
    std::cerr << "Size mismatch: GPU=" << gpu_size << " vs CPU=" << cpu_output.size() << std::endl;
    return false;
  }

  // Copy GPU data to host for comparison
  std::vector<half> host_output(gpu_size);
  FI_GPU_CALL(
      gpuMemcpy(host_output.data(), gpu_output, gpu_size * sizeof(half), gpuMemcpyDeviceToHost));

  int errors = 0;
  float max_diff = 0.0f;
  float max_rel_diff = 0.0f;

  for (size_t i = 0; i < gpu_size; ++i) {
    float gpu_val = static_cast<float>(host_output[i]);
    float cpu_val = static_cast<float>(cpu_output[i]);
    float abs_diff = std::abs(gpu_val - cpu_val);
    float rel_diff = (cpu_val != 0.0f) ? abs_diff / std::abs(cpu_val) : abs_diff;

    max_diff = std::max(max_diff, abs_diff);
    max_rel_diff = std::max(max_rel_diff, rel_diff);

    bool close = (abs_diff <= atol + rtol * std::abs(cpu_val));
    if (!close) {
      errors++;
      if (errors <= 10) {  // Print just a few examples
        std::cerr << "Mismatch at " << i << ": GPU=" << gpu_val << " CPU=" << cpu_val
                  << " (diff=" << abs_diff << ")" << std::endl;
      }
    }
  }

  float error_rate = static_cast<float>(errors) / gpu_size;
  std::cout << "\nValidation Results:" << std::endl;
  std::cout << "  Max absolute difference: " << max_diff << std::endl;
  std::cout << "  Max relative difference: " << max_rel_diff << std::endl;
  std::cout << "  Error rate: " << (error_rate * 100) << "% (" << errors << " / " << gpu_size
            << " elements)" << std::endl;
  std::cout << "  Status: " << (error_rate < 0.05 ? "PASSED" : "FAILED") << std::endl;

  // Allow up to 5% error rate
  return error_rate < 0.05;
}

using namespace flashinfer;

// Helper class to convert strings to parameters
class ArgParser {
 public:
  static bool get_bool(const char* arg, bool default_val) {
    return arg == nullptr ? default_val : (std::string(arg) == "1" || std::string(arg) == "true");
  }

  static int get_int(const char* arg, int default_val) {
    return arg == nullptr ? default_val : std::atoi(arg);
  }

  static float get_float(const char* arg, float default_val) {
    return arg == nullptr ? default_val : std::atof(arg);
  }

  static PosEncodingMode get_pos_encoding_mode(const char* arg) {
    if (arg == nullptr) return PosEncodingMode::kNone;
    std::string str_val = arg;
    if (str_val == "none") return PosEncodingMode::kNone;
    if (str_val == "rope") return PosEncodingMode::kRoPELlama;
    if (str_val == "alibi") return PosEncodingMode::kALiBi;
    return PosEncodingMode::kNone;
  }

  static QKVLayout get_layout(const char* arg) {
    if (arg == nullptr) return QKVLayout::kNHD;
    std::string str_val = arg;
    if (str_val == "nhd") return QKVLayout::kNHD;
    if (str_val == "hnd") return QKVLayout::kHND;
    return QKVLayout::kNHD;
  }
};

// Dispatch function for half precision
gpuError_t dispatch_single_prefill(half* q_ptr, half* k_ptr, half* v_ptr, half* o_ptr,
                                   half* tmp_ptr, float* lse_ptr, uint32_t num_qo_heads,
                                   uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
                                   uint32_t head_dim, QKVLayout kv_layout,
                                   PosEncodingMode pos_encoding_mode, bool causal,
                                   bool use_fp16_qk_reduction, double sm_scale, int32_t window_left,
                                   double rope_scale, double rope_theta, gpuStream_t stream) {
  // Compute strides based on layout
  uint32_t q_stride_n = num_qo_heads * head_dim;
  uint32_t q_stride_h = head_dim;
  uint32_t k_stride_n, k_stride_h, v_stride_n, v_stride_h;

  if (kv_layout == QKVLayout::kNHD) {
    k_stride_n = num_kv_heads * head_dim;
    k_stride_h = head_dim;
    v_stride_n = num_kv_heads * head_dim;
    v_stride_h = head_dim;
  } else {
    k_stride_h = kv_len * head_dim;
    k_stride_n = head_dim;
    v_stride_h = kv_len * head_dim;
    v_stride_n = head_dim;
  }

  // Configure mask mode
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;

  // Constants for prefill kernel
  constexpr uint32_t HEAD_DIM_QK = 128;
  constexpr uint32_t HEAD_DIM_VO = 128;
  constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kRoPELlama;
  constexpr bool USE_FP16_QK_REDUCTION = false;

  gpuError_t status = gpuSuccess;

  if (causal) {
    // Causal attention
    using AttentionVariantType = DefaultAttention<false, false, false, false>;
    using Params = SinglePrefillParams<half, int32_t>;

    Params params;
    params.q = q_ptr;
    params.k = k_ptr;
    params.v = v_ptr;
    params.o = o_ptr;
    params.lse = lse_ptr;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = num_kv_heads;
    params.group_size = uint_fastdiv(num_qo_heads / num_kv_heads);
    params.qo_len = qo_len;
    params.kv_len = kv_len;
    params.q_stride_n = q_stride_n;
    params.q_stride_h = q_stride_h;
    params.k_stride_n = k_stride_n;
    params.k_stride_h = k_stride_h;
    params.v_stride_n = v_stride_n;
    params.v_stride_h = v_stride_h;
    params.head_dim = head_dim;
    params.window_left = window_left;
    params.partition_kv = false;
    params.maybe_custom_mask = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.logits_soft_cap = 0.0;
    params.sm_scale = sm_scale;
    params.rope_rcp_scale = 1.0 / rope_scale;
    params.rope_rcp_theta = 1.0 / rope_theta;

    status = SinglePrefillWithKVCacheDispatched<HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
                                                USE_FP16_QK_REDUCTION, MaskMode::kCausal,
                                                AttentionVariantType>(params, tmp_ptr, stream);
  } else {
    // Non-causal attention
    using AttentionVariantType = DefaultAttention<false, false, false, false>;
    using Params = SinglePrefillParams<half, int32_t>;

    Params params;
    params.q = q_ptr;
    params.k = k_ptr;
    params.v = v_ptr;
    params.o = o_ptr;
    params.lse = lse_ptr;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = num_kv_heads;
    params.group_size = uint_fastdiv(num_qo_heads / num_kv_heads);
    params.qo_len = qo_len;
    params.kv_len = kv_len;
    params.q_stride_n = q_stride_n;
    params.q_stride_h = q_stride_h;
    params.k_stride_n = k_stride_n;
    params.k_stride_h = k_stride_h;
    params.v_stride_n = v_stride_n;
    params.v_stride_h = v_stride_h;
    params.head_dim = head_dim;
    params.window_left = window_left;
    params.partition_kv = false;
    params.maybe_custom_mask = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.logits_soft_cap = 0.0;
    params.sm_scale = sm_scale;
    params.rope_rcp_scale = 1.0 / rope_scale;
    params.rope_rcp_theta = 1.0 / rope_theta;

    status = SinglePrefillWithKVCacheDispatched<HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
                                                USE_FP16_QK_REDUCTION, MaskMode::kNone,
                                                AttentionVariantType>(params, tmp_ptr, stream);
  }

  return status;
}

// Function to calculate FLOPs for single_prefill
double calculate_flops(uint32_t qo_len, uint32_t kv_len, uint32_t num_qo_heads, uint32_t head_dim,
                       bool causal) {
  double flops;
  if (causal) {
    // For causal attention: qo_len * (2 * kv_len - qo_len) * 2 *
    // num_qo_heads * head_dim
    flops = static_cast<double>(qo_len) * (2.0 * kv_len - qo_len) * 2.0 * num_qo_heads * head_dim;
  } else {
    // For non-causal attention: qo_len * kv_len * 4 * num_qo_heads *
    // head_dim
    flops = static_cast<double>(qo_len) * kv_len * 4.0 * num_qo_heads * head_dim;
  }
  return flops;
}

void print_usage(const char* program_name) {
  std::cerr << "Usage: " << program_name << " [options]\n"
            << "Options:\n"
            << "  --qo_len <int>            : Query sequence length (default: "
               "512)\n"
            << "  --kv_len <int>            : Key/value sequence length (default: "
               "512)\n"
            << "  --num_qo_heads <int>      : Number of query heads (default: 32)\n"
            << "  --num_kv_heads <int>      : Number of key/value heads (default: "
               "32)\n"
            << "  --head_dim <int>          : Head dimension (default: 128)\n"
            << "  --layout <nhd|hnd>        : KV tensor layout (default: nhd)\n"
            << "  --pos_encoding <none|rope|alibi> : Position encoding mode "
               "(default: none)\n"
            << "  --causal <0|1>            : Use causal mask (default: 1)\n"
            << "  --use_fp16_qk <0|1>       : Use FP16 for QK reduction (default: "
               "0)\n"
            << "  --window_left <int>       : Window left size (default: -1)\n"
            << "  --rope_scale <float>      : RoPE scale factor (default: 1.0)\n"
            << "  --rope_theta <float>      : RoPE theta (default: 10000.0)\n"
            << "  --iterations <int>        : Number of iterations for timing "
               "(default: 10)\n"
            << "  --warmup <int>            : Number of warmup iterations "
               "(default: 5)\n"
            << "  --validate <0|1>          : Validate against CPU reference "
               "(default: 0)\n";
}

// Main function with simplified memory management
int main(int argc, char* argv[]) {
  if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    print_usage(argv[0]);
    return 0;
  }

  // Process parameter pairs (--param value)
  uint32_t qo_len = 512;
  uint32_t kv_len = 512;
  uint32_t num_qo_heads = 32;
  uint32_t num_kv_heads = 32;
  uint32_t head_dim = 128;
  QKVLayout kv_layout = QKVLayout::kNHD;
  PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone;
  bool causal = true;
  bool use_fp16_qk_reduction = false;
  int32_t window_left = -1;
  float rope_scale = 1.0f;
  float rope_theta = 10000.0f;
  int iterations = 10;
  int warmup = 5;
  bool validate = false;

  for (int i = 1; i < argc; i += 2) {
    std::string arg = argv[i];
    if (i + 1 >= argc && arg != "--help" && arg != "-h") {
      std::cerr << "Missing value for parameter " << arg << std::endl;
      print_usage(argv[0]);
      return 1;
    }

    if (arg == "--qo_len") {
      qo_len = ArgParser::get_int(argv[i + 1], 512);
    } else if (arg == "--kv_len") {
      kv_len = ArgParser::get_int(argv[i + 1], 512);
    } else if (arg == "--num_qo_heads") {
      num_qo_heads = ArgParser::get_int(argv[i + 1], 32);
    } else if (arg == "--num_kv_heads") {
      num_kv_heads = ArgParser::get_int(argv[i + 1], 32);
    } else if (arg == "--head_dim") {
      head_dim = ArgParser::get_int(argv[i + 1], 128);
    } else if (arg == "--layout") {
      kv_layout = ArgParser::get_layout(argv[i + 1]);
    } else if (arg == "--pos_encoding") {
      pos_encoding_mode = ArgParser::get_pos_encoding_mode(argv[i + 1]);
    } else if (arg == "--causal") {
      causal = ArgParser::get_bool(argv[i + 1], true);
    } else if (arg == "--use_fp16_qk") {
      use_fp16_qk_reduction = ArgParser::get_bool(argv[i + 1], false);
    } else if (arg == "--window_left") {
      window_left = ArgParser::get_int(argv[i + 1], -1);
    } else if (arg == "--rope_scale") {
      rope_scale = ArgParser::get_float(argv[i + 1], 1.0f);
    } else if (arg == "--rope_theta") {
      rope_theta = ArgParser::get_float(argv[i + 1], 10000.0f);
    } else if (arg == "--iterations") {
      iterations = ArgParser::get_int(argv[i + 1], 10);
    } else if (arg == "--warmup") {
      warmup = ArgParser::get_int(argv[i + 1], 5);
    } else if (arg == "--validate") {
      validate = ArgParser::get_bool(argv[i + 1], false);
    } else {
      std::cerr << "Unknown parameter: " << arg << std::endl;
      print_usage(argv[0]);
      return 1;
    }
  }

  // Print configuration
  std::cout << "Configuration:" << std::endl
            << "  QO Length: " << qo_len << std::endl
            << "  KV Length: " << kv_len << std::endl
            << "  QO Heads: " << num_qo_heads << std::endl
            << "  KV Heads: " << num_kv_heads << std::endl
            << "  Head Dimension: " << head_dim << std::endl
            << "  KV Layout: " << (kv_layout == QKVLayout::kNHD ? "NHD" : "HND") << std::endl
            << "  Position Encoding: "
            << (pos_encoding_mode == PosEncodingMode::kNone        ? "None"
                : pos_encoding_mode == PosEncodingMode::kRoPELlama ? "RoPE"
                                                                   : "ALiBi")
            << std::endl
            << "  Causal: " << (causal ? "Yes" : "No") << std::endl
            << "  Use FP16 QK Reduction: " << (use_fp16_qk_reduction ? "Yes" : "No") << std::endl
            << "  Window Left: " << window_left << std::endl
            << "  RoPE Scale: " << rope_scale << std::endl
            << "  RoPE Theta: " << rope_theta << std::endl
            << "  Iterations: " << iterations << std::endl
            << "  Warmup: " << warmup << std::endl
            << "  Validation: " << (validate ? "Yes" : "No") << std::endl;

  // Create stream
  gpuStream_t stream;
  FI_GPU_CALL(gpuStreamCreate(&stream));

  // Allocate device memory using gpuMalloc instead of Thrust
  half *q_dev, *k_dev, *v_dev, *o_dev, *tmp_dev;
  float* lse_dev;

  size_t q_size = qo_len * num_qo_heads * head_dim;
  size_t k_size = kv_len * num_kv_heads * head_dim;
  size_t v_size = kv_len * num_kv_heads * head_dim;
  size_t o_size = qo_len * num_qo_heads * head_dim;
  size_t lse_size = qo_len * num_qo_heads;

  FI_GPU_CALL(gpuMalloc(&q_dev, q_size * sizeof(half)));
  FI_GPU_CALL(gpuMalloc(&k_dev, k_size * sizeof(half)));
  FI_GPU_CALL(gpuMalloc(&v_dev, v_size * sizeof(half)));
  FI_GPU_CALL(gpuMalloc(&o_dev, o_size * sizeof(half)));
  FI_GPU_CALL(gpuMalloc(&tmp_dev, o_size * sizeof(half)));
  FI_GPU_CALL(gpuMalloc(&lse_dev, lse_size * sizeof(float)));

  // Initialize data
  generate_random_data(q_dev, q_size);
  generate_random_data(k_dev, k_size);
  generate_random_data(v_dev, v_size);

  // Zero out output arrays
  FI_GPU_CALL(gpuMemset(o_dev, 0, o_size * sizeof(half)));
  FI_GPU_CALL(gpuMemset(tmp_dev, 0, o_size * sizeof(half)));
  FI_GPU_CALL(gpuMemset(lse_dev, 0, lse_size * sizeof(float)));

  // Calculate SM scale
  float sm_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Warmup runs
  for (int i = 0; i < warmup; ++i) {
    gpuError_t status = dispatch_single_prefill(
        q_dev, k_dev, v_dev, o_dev, tmp_dev, lse_dev, num_qo_heads, num_kv_heads, qo_len, kv_len,
        head_dim, kv_layout, pos_encoding_mode, causal, use_fp16_qk_reduction, sm_scale,
        window_left, rope_scale, rope_theta, stream);

    if (status != gpuSuccess) {
      std::cerr << "Error during warmup: " << gpuGetErrorString(status) << std::endl;
      return 1;
    }
  }

  // Timing runs
  gpuEvent_t start, stop;
  FI_GPU_CALL(gpuEventCreate(&start));
  FI_GPU_CALL(gpuEventCreate(&stop));

  FI_GPU_CALL(gpuEventRecord(start, stream));

  for (int i = 0; i < iterations; ++i) {
    gpuError_t status = dispatch_single_prefill(
        q_dev, k_dev, v_dev, o_dev, tmp_dev, lse_dev, num_qo_heads, num_kv_heads, qo_len, kv_len,
        head_dim, kv_layout, pos_encoding_mode, causal, use_fp16_qk_reduction, sm_scale,
        window_left, rope_scale, rope_theta, stream);

    if (status != gpuSuccess) {
      std::cerr << "Error during benchmark: " << gpuGetErrorString(status) << std::endl;
      return 1;
    }
  }

  FI_GPU_CALL(gpuEventRecord(stop, stream));
  FI_GPU_CALL(gpuEventSynchronize(stop));

  float elapsed_ms;
  FI_GPU_CALL(gpuEventElapsedTime(&elapsed_ms, start, stop));
  float avg_ms = elapsed_ms / iterations;

  // Calculate and report performance
  double flops = calculate_flops(qo_len, kv_len, num_qo_heads, head_dim, causal);
  double tflops = flops / (avg_ms * 1e-3) / 1e12;

  // Report results
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Performance Results:" << std::endl;
  std::cout << "  Average time: " << avg_ms << " ms" << std::endl;
  std::cout << "  Performance: " << tflops << " TFLOPS" << std::endl;

  // Run validation if requested
  if (validate) {
    std::cout << "\nRunning validation..." << std::endl;

    // Copy input data to host for CPU reference
    std::vector<half> h_q(q_size), h_k(k_size), h_v(v_size);
    FI_GPU_CALL(gpuMemcpy(h_q.data(), q_dev, q_size * sizeof(half), gpuMemcpyHostToDevice));
    FI_GPU_CALL(gpuMemcpy(h_k.data(), k_dev, k_size * sizeof(half), gpuMemcpyHostToDevice));
    FI_GPU_CALL(gpuMemcpy(h_v.data(), v_dev, v_size * sizeof(half), gpuMemcpyHostToDevice));

    // Compute reference output on CPU
    std::vector<half> ref_output =
        reference::single_mha(h_q, h_k, h_v, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim,
                              causal, kv_layout, pos_encoding_mode, rope_scale, rope_theta);

    // Validate results
    bool validation_passed = validate_results(o_dev, o_size, ref_output);

    // Report validation status
    std::cout << "Validation " << (validation_passed ? "PASSED" : "FAILED") << std::endl;
  }

  // Cleanup
  FI_GPU_CALL(gpuEventDestroy(start));
  FI_GPU_CALL(gpuEventDestroy(stop));
  FI_GPU_CALL(gpuStreamDestroy(stream));
  FI_GPU_CALL(gpuFree(q_dev));
  FI_GPU_CALL(gpuFree(k_dev));
  FI_GPU_CALL(gpuFree(v_dev));
  FI_GPU_CALL(gpuFree(o_dev));
  FI_GPU_CALL(gpuFree(tmp_dev));
  FI_GPU_CALL(gpuFree(lse_dev));

  return 0;
}

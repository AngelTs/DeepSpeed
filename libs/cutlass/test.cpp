#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "gemm_api.h"
torch::Tensor ds_cutlass_gemm(torch::Tensor& A, torch::Tensor& B, torch::Tensor& alpha, torch::Tensor& alpha1, torch::Tensor& beta)
{
  int M = A.size(0);
  int N = B.size(0);
  int K = A.size(1);
  int groups = alpha.size(0);
  int groups1 = alpha1.size(0);
  auto options = at::TensorOptions()
    .dtype(at::kHalf)
    .layout(B.options().layout())
    .device(at::kCUDA)
    .requires_grad(false);
  int bsz1 = (M % 128 == 0) ? M : M + (128 - (M % 128));
  auto C = at::empty({bsz1, N}, options);

  run_gemm(A.data_ptr(), B.data_ptr(), C.data_ptr(), alpha.data_ptr(), alpha1.data_ptr(), M, N, K, groups, groups1, at::cuda::getCurrentCUDAStream());
  return at::narrow(C, 0, 0, M);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cutlass_gemm", &ds_cutlass_gemm, "DeepSpeed bias_gelu forward(CUDA)");
}

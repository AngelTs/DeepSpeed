

#pragma once

void run_gemm(void* A,
              void* B,
              void* C,
              void* a,
              void* aa,
              int M,
              int N,
              int K,
              int groups,
              int groups1,
              cudaStream_t stream);

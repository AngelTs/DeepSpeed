

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

void run_gemm_int4(void* A,
                   void* B,
                   void* C,
                   void* alpha1,
                   void* alpha2,
                   int M,
                   int N,
                   int K,
                   int groups,
                   int groups1,
                   cudaStream_t stream);

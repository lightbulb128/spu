// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>

#include "spdlog/spdlog.h"

#include "libspu/core/parallel_utils.h"
#include "libspu/core/prelude.h"

#include "include/int-gemm/int_gemm.h"

#define EIGEN_HAS_OPENMP

#include "Eigen/Core"

namespace spu::mpc::linalg {

namespace detail {

void setEigenParallelLevel(int64_t expected_threads);

}  // namespace detail

/**
 * @brief C := op( A )*op( B )
 *
 * @tparam T Type of A, B, C
 * @param M   Number of rows in A
 * @param N   Number of columns in B
 * @param K   Number of columns in A and number of rows in B
 * @param A   Pointer to A
 * @param LDA Leading dimension stride of A
 * @param IDA Inner dimension stride of A
 * @param B   Pointer to B
 * @param LDB Leading dimension stride of B
 * @param IDB Inner dimension stride of B
 * @param C   Pointer to C
 * @param LDC Leading dimension stride of C
 * @param IDC Inner dimension stride of C
 */
template <typename T>
void matmul_general(int64_t M, int64_t N, int64_t K, const T* A, int64_t LDA,
            int64_t IDA, const T* B, int64_t LDB, int64_t IDB, T* C,
            int64_t LDC, int64_t IDC) {
  using StrideT = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
  using MapMatrixConstT = Eigen::Map<
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::Unaligned, StrideT>;
  using MapMatrixT = Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
      Eigen::Unaligned, StrideT>;

  MapMatrixConstT a(A, M, K, StrideT(LDA, IDA));
  MapMatrixConstT b(B, K, N, StrideT(LDB, IDB));
  MapMatrixT c(C, M, N, StrideT(LDC, IDC));

  // if (M == 1) {
  //   // GEMV case 1*K * K*N -> 1*N
  //   auto work_load_size = computeTaskSize(N);
  //   yacl::parallel_for(0, N, work_load_size, [&](int64_t begin, int64_t end)
  //   {
  //     auto block_size = end - begin;
  //     c.block(0, begin, 1, block_size) =
  //         a.row(0) * b.block(0, begin, K, block_size);
  //   });
  //   return;
  // } else if (N == 1) {
  //   // GEMV case M*K * K*1 -> M*1
  //   auto work_load_size = computeTaskSize(M);
  //   yacl::parallel_for(0, M, work_load_size, [&](int64_t begin, int64_t end)
  //   {
  //     auto block_size = end - begin;
  //     c.block(begin, 0, block_size, 1) =
  //         a.block(begin, 0, block_size, K) * b.col(0);
  //   });
  //   return;
  // }

  // If we don't limit # threads, eigen may overloading omp tasks (especially
  // under relative small tasks, MLP for example)
  //
  // FIXME: Investigate what can happen once we support ILP
  //        The performance is extremely bad when multi-process all tries to use
  //        num_cores.
  // auto expected_num_threads = std::max((M * K + kMinTaskSize) / kMinTaskSize,
  //                                     (N * K + kMinTaskSize) / kMinTaskSize);
  detail::setEigenParallelLevel(2);

  c.noalias() = a * b;
}

template <typename T>
void matrix_copy(int64_t M, int64_t N, 
    const T* A, int64_t LDA, int64_t IDA, 
    T* B, int64_t LDB, int64_t IDB)
{
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            B[i * LDB + j * IDB] = A[i * LDA + j * IDA];
        }
    }
}

template <typename T>
void matmul(int64_t M, int64_t N, int64_t K, const T* A, int64_t LDA,
            int64_t IDA, const T* B, int64_t LDB, int64_t IDB, T* C,
            int64_t LDC, int64_t IDC) {

    bool type_compatible = 
        std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value ||
        std::is_same<T, uint64_t>::value || std::is_same<T, int64_t>::value ||
        std::is_same<T, unsigned __int128>::value || std::is_same<T, __int128>::value;

    if (!type_compatible) {
        matmul_general(M, N, K, A, LDA, IDA, B, LDB, IDB, C, LDC, IDC);
        return;
    }

    bool stride_A_compatible = LDA == K * IDA;
    bool stride_B_compatible = LDB == N * IDB;
    bool stride_C_compatible = LDC == N * IDC;
    
    const T* A_ptr;
    if (!stride_A_compatible) {
        T* A_copied = new T[M * K];
        matrix_copy(M, K, A, LDA, IDA, A_copied, K, 1);
        A_ptr = A_copied;
    } else A_ptr = A;

    const T* B_ptr;
    if (!stride_B_compatible) {
        T* B_copied = new T[K * N];
        matrix_copy(K, N, B, LDB, IDB, B_copied, N, 1);
        B_ptr = B_copied;
    } else B_ptr = B;

    T* C_ptr;
    if (!stride_C_compatible) {
        T* C_copied = new T[M * N];
        C_ptr = C_copied;
    } else C_ptr = C;
    
    // printf("%d %d %d", stride_A_compatible, stride_B_compatible, stride_C_compatible);
    
    if (std::is_same<T, uint32_t>::value) {
        int_gemm::hostUint32MatmulStrided(
            M, N, K, 
            reinterpret_cast<const uint32_t*>(A_ptr), 
            reinterpret_cast<const uint32_t*>(B_ptr), 
            reinterpret_cast<uint32_t*>(C_ptr), 
            stride_A_compatible ? IDA : 1, stride_B_compatible ? IDB : 1, stride_C_compatible ? IDC : 1
        );
    } else if (std::is_same<T, int32_t>::value) {
        int_gemm::hostInt32MatmulStrided(
            M, N, K, 
            reinterpret_cast<const int32_t*>(A_ptr), 
            reinterpret_cast<const int32_t*>(B_ptr), 
            reinterpret_cast<int32_t*>(C_ptr), 
            stride_A_compatible ? IDA : 1, stride_B_compatible ? IDB : 1, stride_C_compatible ? IDC : 1
        );
    } else if (std::is_same<T, uint64_t>::value) {
        int_gemm::hostUint64MatmulStrided(
            M, N, K, 
            reinterpret_cast<const uint64_t*>(A_ptr), 
            reinterpret_cast<const uint64_t*>(B_ptr), 
            reinterpret_cast<uint64_t*>(C_ptr), 
            stride_A_compatible ? IDA : 1, stride_B_compatible ? IDB : 1, stride_C_compatible ? IDC : 1
        );
    } else if (std::is_same<T, int64_t>::value) {
        int_gemm::hostInt64MatmulStrided(
            M, N, K, 
            reinterpret_cast<const int64_t*>(A_ptr), 
            reinterpret_cast<const int64_t*>(B_ptr), 
            reinterpret_cast<int64_t*>(C_ptr), 
            stride_A_compatible ? IDA : 1, stride_B_compatible ? IDB : 1, stride_C_compatible ? IDC : 1
        );
    }  else if (std::is_same<T, unsigned __int128>::value) {
        int_gemm::hostUint128MatmulStrided(
            M, N, K, 
            reinterpret_cast<const unsigned __int128*>(A_ptr), 
            reinterpret_cast<const unsigned __int128*>(B_ptr), 
            reinterpret_cast<unsigned __int128*>(C_ptr), 
            stride_A_compatible ? IDA : 1, stride_B_compatible ? IDB : 1, stride_C_compatible ? IDC : 1
        );
    } else if (std::is_same<T, __int128_t>::value) {
        int_gemm::hostInt128MatmulStrided(
            M, N, K, 
            reinterpret_cast<const __int128*>(A_ptr), 
            reinterpret_cast<const __int128*>(B_ptr), 
            reinterpret_cast<__int128*>(C_ptr), 
            stride_A_compatible ? IDA : 1, stride_B_compatible ? IDB : 1, stride_C_compatible ? IDC : 1
        );
    } else {
        matmul_general(M, N, K, A, LDA, IDA, B, LDB, IDB, C, LDC, IDC);
    }

    if (!stride_A_compatible) delete[] A_ptr;
    if (!stride_B_compatible) delete[] B_ptr;
    if (!stride_C_compatible) {
        matrix_copy(M, N, C_ptr, N, 1, C, LDC, IDC);
        delete[] C_ptr;
    }
}  

}  // namespace spu::mpc::linalg
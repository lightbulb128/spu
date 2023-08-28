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

#include "libspu/mpc/utils/linalg.h"

#include <vector>

#include "gtest/gtest.h"

namespace spu::mpc::linalg {

TEST(LinalgTest, MatMulBasic) {
  std::vector<float> A = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15};  // 4x3
  std::vector<float> B = {1, 2, 3, 4, 5, 6};                         // 3x2
  std::vector<float> C(8);                                           // 4x2

  matmul(4, 2, 3, A.data(), 3, 1, B.data(), 2, 1, C.data(), 2, 1);

  std::vector<float> expected = {22.f, 28.f,  58.f,  76.f,
                                 94.f, 124.f, 130.f, 172.f};

  EXPECT_EQ(C, expected);
}

TEST(LinalgTest, MatMulStrides) {
  std::vector<float> A = {1,  100, 2,  100, 3,  100,   //
                          5,  100, 6,  100, 7,  100,   //
                          9,  100, 10, 100, 11, 100,   //
                          13, 100, 14, 100, 15, 100};  // 4x3

  std::vector<float> B = {1, 100, 2, 100,   //
                          3, 100, 4, 100,   //
                          5, 100, 6, 100};  // 3x2
  std::vector<float> C(8);                  // 4x2

  matmul(4, 2, 3, A.data(), 6, 2, B.data(), 4, 2, C.data(), 2, 1);

  std::vector<float> expected = {22.f, 28.f,  58.f,  76.f,
                                 94.f, 124.f, 130.f, 172.f};

  EXPECT_EQ(C, expected);
}

template<typename T>
std::vector<T> random_vector(size_t count) {
  std::vector<T> v(count);
  for (size_t i = 0; i < count; ++i) {
    v[i] = static_cast<T>(rand());
  }
  return v;
}

template<typename T>
void matmul_cpu(int64_t M, int64_t N, int64_t K, const T* A, int64_t LDA,
            int64_t IDA, const T* B, int64_t LDB, int64_t IDB, T* C,
            int64_t LDC, int64_t IDC)
{
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      T sum = 0;
      for (int64_t k = 0; k < K; ++k) {
        sum += A[i * LDA + k * IDA] * B[k * LDB + j * IDB];
      }
      C[i * LDC + j * IDC] = sum;
    }
  }
}

template<typename T>
bool matrix_equal(int64_t M, int64_t N, 
        const T* a, const T* b, int64_t LD, int64_t ID)
{
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      if (a[i * LD + j * ID] != b[i * LD + j * ID]) {
        return false;
      }
    }
  }
  return true;
}

template<typename T>
void test_matmul_basic(int64_t m, int64_t n, int64_t k) {
  std::vector<T> A = random_vector<T>(m * k);
  std::vector<T> B = random_vector<T>(k * n);
  std::vector<T> C(m * n);
  std::vector<T> C_cpu(m * n);
  matmul(m, n, k, A.data(), k, 1, B.data(), n, 1, C.data(), n, 1);
  matmul_cpu(m, n, k, A.data(), k, 1, B.data(), n, 1, C_cpu.data(), n, 1);
  bool equal = matrix_equal(m, n, C.data(), C_cpu.data(), n, 1);
  EXPECT_EQ(equal, true);
}

template<typename T>
void test_matmul_uniformly_strided(int64_t m, int64_t n, int64_t k, int64_t stride) {
  
  std::vector<T> A = random_vector<T>(m * k * stride);
  std::vector<T> B = random_vector<T>(k * n * stride);
  std::vector<T> C(m * n * stride);
  std::vector<T> C_cpu(m * n * stride);
  matmul(m, n, k, 
    A.data(), k * stride, stride, 
    B.data(), n * stride, stride, 
    C.data(), n * stride, stride);
  matmul_cpu(m, n, k, 
    A.data(), k * stride, stride, 
    B.data(), n * stride, stride, 
    C_cpu.data(), n * stride, stride);
  bool equal = matrix_equal(m, n, C.data(), C_cpu.data(), n * stride, stride);
  EXPECT_EQ(equal, true);
}

template<typename T>
void test_matmul_unevenly_strided(
  int64_t m, int64_t n, int64_t k, 
  int64_t A_stride_m, int64_t A_stride_k,
  int64_t B_stride_k, int64_t B_stride_n,
  int64_t C_stride_m, int64_t C_stride_n
) {

  std::vector<T> A = random_vector<T>(m * k * A_stride_m * A_stride_k);
  std::vector<T> B = random_vector<T>(k * n * B_stride_k * B_stride_n);
  std::vector<T> C(m * n * C_stride_m * C_stride_n);
  std::vector<T> C_cpu(m * n * C_stride_m * C_stride_n);
  matmul(m, n, k, 
    A.data(), k * A_stride_k * A_stride_m, A_stride_k,
    B.data(), n * B_stride_n * B_stride_k, B_stride_n,
    C.data(), n * C_stride_n * C_stride_m, C_stride_n);
  matmul_cpu(m, n, k, 
    A.data(), k * A_stride_k * A_stride_m, A_stride_k,
    B.data(), n * B_stride_n * B_stride_k, B_stride_n,
    C_cpu.data(), n * C_stride_n * C_stride_m, C_stride_n);
  bool equal = matrix_equal(m, n, C.data(), C_cpu.data(), n * C_stride_n * C_stride_m, C_stride_n);
  EXPECT_EQ(equal, true);
}

TEST(LinalgTest, MatMulUint32Basic) {
  int64_t m = 200; int64_t n = 200; int64_t k = 200;
  test_matmul_basic<uint32_t>(m, n, k);
}

TEST(LinalgTest, MatMulUint32UniformlyStrided) {
  int64_t m = 200; int64_t n = 200; int64_t k = 200; int64_t stride = 2;
  test_matmul_uniformly_strided<uint32_t>(m, n, k, stride);
}

TEST(LinalgTest, MatMulUint32UnevenlyStrided) {
  int64_t m = 200; int64_t n = 200; int64_t k = 200;
  test_matmul_unevenly_strided<uint32_t>(m, n, k, 1, 1, 2, 2, 3, 3);
}

TEST(LinalgTest, MatMulUint64Basic) {
  int64_t m = 200; int64_t n = 200; int64_t k = 200;
  test_matmul_basic<uint64_t>(m, n, k);
}

TEST(LinalgTest, MatMulUint64UniformlyStrided) {
  int64_t m = 200; int64_t n = 200; int64_t k = 200; int64_t stride = 2;
  test_matmul_uniformly_strided<uint64_t>(m, n, k, stride);
}

TEST(LinalgTest, MatMulUint128Basic) {
  int64_t m = 200; int64_t n = 200; int64_t k = 200;
  test_matmul_basic<unsigned __int128>(m, n, k);
}

TEST(LinalgTest, MatMulUint128UniformlyStrided) {
  int64_t m = 200; int64_t n = 200; int64_t k = 200; int64_t stride = 2;
  test_matmul_uniformly_strided<unsigned __int128>(m, n, k, stride);
}

}  // namespace spu::mpc::linalg

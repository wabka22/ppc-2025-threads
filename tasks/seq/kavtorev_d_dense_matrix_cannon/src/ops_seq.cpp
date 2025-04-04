// Copyright 2025 Kavtorev Dmitry
#include "seq/kavtorev_d_dense_matrix_cannon/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

std::vector<double> kavtorev_d_dense_matrix_cannon_seq::CannonMatrixMultiplication(const std::vector<double>& a,
                                                                                   const std::vector<double>& b, int n,
                                                                                   int m) {
  int size_block = std::min(n, m);

  std::vector<double> mtrx_c(n * m, 0.0);

  if (n == 0 || m == 0) {
    return {};
  }

  for (int i = 0; i < n; i += size_block) {
    for (int j = 0; j < m; j += size_block) {
      for (int k = 0; k < m; k += size_block) {
        int i_end = std::min(i + size_block, n);
        int j_end = std::min(j + size_block, m);
        int k_end = std::min(k + size_block, m);

        for (int ii = i; ii < i_end; ++ii) {
          for (int kk = k; kk < k_end; ++kk) {
            double a_ik = a[(ii * m) + kk];
            for (int jj = j; jj < j_end; ++jj) {
              mtrx_c[(ii * m) + jj] += a_ik * b[(kk * m) + jj];
            }
          }
        }
      }
    }
  }

  return mtrx_c;
}

std::vector<double> kavtorev_d_dense_matrix_cannon_seq::MultiplyMatrix(const std::vector<double>& a,
                                                                       const std::vector<double>& b, int rows_a,
                                                                       int col_b) {
  int col_a = rows_a;
  std::vector<double> mtrx_c(rows_a * col_b, 0.0);

  if (rows_a == 0 || col_b == 0) {
    return {};
  }

  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < col_b; ++j) {
      for (int k = 0; k < col_a; ++k) {
        mtrx_c[(i * col_b) + j] += a[(i * col_a) + k] * b[(k * col_b) + j];
      }
    }
  }
  return mtrx_c;
}

bool kavtorev_d_dense_matrix_cannon_seq::TestTaskSequential::PreProcessingImpl() {
  A_ = std::vector<double>(task_data->inputs_count[0]);
  B_ = std::vector<double>(task_data->inputs_count[1]);
  n_ = *reinterpret_cast<int*>(task_data->inputs[2]);
  m_ = *reinterpret_cast<int*>(task_data->inputs[3]);

  auto* tmp_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  for (size_t i = 0; i < task_data->inputs_count[0]; i++) {
    A_[i] = tmp_ptr_a[i];
  }

  auto* tmp_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);
  for (size_t i = 0; i < task_data->inputs_count[1]; i++) {
    B_[i] = tmp_ptr_b[i];
  }
  return true;
}

bool kavtorev_d_dense_matrix_cannon_seq::TestTaskSequential::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[1] &&
         task_data->inputs_count[0] == task_data->outputs_count[0] &&
         task_data->inputs_count[1] == task_data->outputs_count[0] && task_data->inputs_count[1] > 0 &&
         task_data->inputs_count[0] > 0;
}

bool kavtorev_d_dense_matrix_cannon_seq::TestTaskSequential::RunImpl() {
  res_ = CannonMatrixMultiplication(A_, B_, n_, m_);
  return true;
}

bool kavtorev_d_dense_matrix_cannon_seq::TestTaskSequential::PostProcessingImpl() {
  std::ranges::copy(res_.begin(), res_.end(), reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
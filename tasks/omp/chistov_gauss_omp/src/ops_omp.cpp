#include "omp/chistov_gauss_omp/include/ops_omp.hpp"

#include <omp.h>
#include <cmath>
#include <iostream>
#include <cstddef>
#include <vector>

bool chistov_gauss_omp::TestTaskOpenMP::PreProcessingImpl() {
  kernel.assign(reinterpret_cast<double *>(task_data->inputs[1]), reinterpret_cast<double *>(task_data->inputs[1]) + 3);
  width = static_cast<size_t>(task_data->inputs_count[1]);
  height = static_cast<size_t>(task_data->inputs_count[2]);
  result_image = std::vector<double>(width * height, 0);
  return true;
}

bool chistov_gauss_omp::TestTaskOpenMP::ValidationImpl() {
  if (task_data->inputs[0] == nullptr || task_data->inputs_count[0] == 0) {
    return false;
  }

  image.assign(reinterpret_cast<double *>(task_data->inputs[0]), 
             reinterpret_cast<double *>(task_data->inputs[0]) + task_data->inputs_count[0]);

  for (size_t i = 0; i < task_data->inputs_count[1] * task_data->inputs_count[2]; ++i) {
    if (image[i] < 0 || image[i] > 255) {
      return false;
    }
  }

  return task_data->inputs_count[0] > 0 &&
         task_data->inputs_count[0] == task_data->inputs_count[1] * task_data->inputs_count[2] &&
         task_data->outputs_count[0] == task_data->inputs_count[1] * task_data->inputs_count[2] &&
         task_data->inputs_count[1] >= 3 && task_data->inputs_count[2] >= 3;
}

bool chistov_gauss_omp::TestTaskOpenMP::RunImpl() {
  double sum = kernel[0] + kernel[1] + kernel[2];
  double sum_inv = 1.0 / sum;

#pragma omp parallel
  {
    if (omp_get_thread_num() == 0) {
      std::cout << "Количество потоков: " << omp_get_num_threads() << std::endl;
    }

#pragma omp for
    for (int i = 0; i < static_cast<int>(height); ++i) {
      for (int j = 0; j < static_cast<int>(width); ++j) {
        double value = 0.0;
        for (ptrdiff_t k = -1; k <= 1; ++k) {
          ptrdiff_t tmp = static_cast<ptrdiff_t>(j) + k;
          if (tmp >= 0 && tmp < static_cast<ptrdiff_t>(width)) {
            value += image[(i * width) + tmp] * kernel[k + 1];
          }
        }

        result_image[(i * width) + j] = value * sum_inv;
      }
    }
  }

  return true;
}

bool chistov_gauss_omp::TestTaskOpenMP::PostProcessingImpl() {
  std::copy(result_image.begin(), result_image.end(), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
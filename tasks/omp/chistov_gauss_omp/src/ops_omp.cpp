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
  double sum_inv = 1.0 / (kernel[0] + kernel[1] + kernel[2]);
  int h = static_cast<int>(height);
  int w = static_cast<int>(width);

#pragma omp parallel firstprivate(sum_inv) shared(w, h) num_threads(8)
  {
#pragma omp for
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        double pixel_0 = (j > 0) ? image[i * width + (j - 1)] * kernel[0] : 0.0;
        double pixel_1 = image[i * width + j] * kernel[1];
        double pixel_2 = (j < width - 1) ? image[i * width + (j + 1)] * kernel[2] : 0.0;

        result_image[i * width + j] = (pixel_0 + pixel_1 + pixel_2) * sum_inv;
      }
    }
  }

  return true;
}

bool chistov_gauss_omp::TestTaskOpenMP::PostProcessingImpl() {
  std::copy(result_image.begin(), result_image.end(), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
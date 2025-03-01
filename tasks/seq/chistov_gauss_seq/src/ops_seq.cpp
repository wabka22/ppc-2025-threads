#include "seq/example/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool chistov_gauss_seq::TestTaskSequential::PreProcessingImpl() {
  kernel.assign(reinterpret_cast<double *>(task_data->inputs[1]), reinterpret_cast<double *>(task_data->inputs[1]) + 3);
  width = static_cast<size_t>(task_data->inputs_count[1]);
  height = static_cast<size_t>(task_data->inputs_count[2]);
  result_image = std::vector<double>(width * height, 0);
  return true;
}

bool chistov_gauss_seq::TestTaskSequential::ValidationImpl() {
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

bool chistov_gauss_seq::TestTaskSequential::RunImpl() {
  double sum = kernel[0] + kernel[1] + kernel[2];
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      double value = 0.0;
      for (ptrdiff_t k = -1; k <= 1; ++k) {
        ptrdiff_t tmp = static_cast<ptrdiff_t>(j) + k;

        if (tmp >= 0 && tmp < static_cast<ptrdiff_t>(width)) {
          value += image[(i * width) + tmp] * kernel[k + 1];
        }
      }

      result_image[(i * width) + j] = value / sum;
    }
  }

  return true;
}

bool chistov_gauss_seq::TestTaskSequential::PostProcessingImpl() {
  std::copy(result_image.begin(), result_image.end(), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
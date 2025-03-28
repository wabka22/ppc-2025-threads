#include "tbb/chistov_gauss_tbb/include/ops_tbb.hpp"
#include <tbb/tbb.h>

#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <vector>

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_group.h"


#include <omp.h>
#include <cmath>
#include <iostream>
#include <cstddef>
#include <vector>

bool chistov_gauss_tbb::TestTaskSequential::PreProcessingImpl() {
  kernel.assign(reinterpret_cast<double *>(task_data->inputs[1]), reinterpret_cast<double *>(task_data->inputs[1]) + 3);
  width = static_cast<size_t>(task_data->inputs_count[1]);
  height = static_cast<size_t>(task_data->inputs_count[2]);
  result_image = std::vector<double>(width * height, 0);
  return true;
}

bool chistov_gauss_tbb::TestTaskSequential::ValidationImpl() {
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

bool chistov_gauss_tbb::TestTaskSequential::RunImpl() {
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

bool chistov_gauss_tbb::TestTaskSequential::PostProcessingImpl() {
  std::copy(result_image.begin(), result_image.end(), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}

bool chistov_gauss_tbb::TestTaskOpenMP::PreProcessingImpl() {
  kernel.assign(reinterpret_cast<double *>(task_data->inputs[1]), reinterpret_cast<double *>(task_data->inputs[1]) + 3);
  width = static_cast<size_t>(task_data->inputs_count[1]);
  height = static_cast<size_t>(task_data->inputs_count[2]);
  result_image = std::vector<double>(width * height, 0);
  return true;
}

bool chistov_gauss_tbb::TestTaskOpenMP::ValidationImpl() {
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

bool chistov_gauss_tbb::TestTaskOpenMP::RunImpl() {
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

bool chistov_gauss_tbb::TestTaskOpenMP::PostProcessingImpl() {
  std::copy(result_image.begin(), result_image.end(), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}

bool chistov_gauss_tbb::TestTaskTBB::PreProcessingImpl() {
  kernel.assign(reinterpret_cast<double *>(task_data->inputs[1]), reinterpret_cast<double *>(task_data->inputs[1]) + 3);
  width = static_cast<size_t>(task_data->inputs_count[1]);
  height = static_cast<size_t>(task_data->inputs_count[2]);
  result_image = std::vector<double>(width * height, 0);
  return true;
}

bool chistov_gauss_tbb::TestTaskTBB::ValidationImpl() {
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

bool chistov_gauss_tbb::TestTaskTBB::RunImpl() {
  double sum_inv = 1.0 / (kernel[0] + kernel[1] + kernel[2]);
  int h = static_cast<int>(height);
  int w = static_cast<int>(width);

  tbb::parallel_for(tbb::blocked_range2d<int>(0, h, 0, w), [&](const tbb::blocked_range2d<int> &r) {
    for (int i = r.rows().begin(); i < r.rows().end(); ++i) {
      for (int j = r.cols().begin(); j < r.cols().end(); ++j) {
        double pixel_0 = (j > 0) ? image[i * width + (j - 1)] * kernel[0] : 0.0;
        double pixel_1 = image[i * width + j] * kernel[1];
        double pixel_2 = (j < width - 1) ? image[i * width + (j + 1)] * kernel[2] : 0.0;

        result_image[i * width + j] = (pixel_0 + pixel_1 + pixel_2) * sum_inv;
      }
    }
  });

  return true;
}

bool chistov_gauss_tbb::TestTaskTBB::PostProcessingImpl() {
  std::copy(result_image.begin(), result_image.end(), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}

#pragma once

#include <utility>
#include <vector>
#include <numeric>
#include <random>

#include "core/task/include/task.hpp"

namespace chistov_gauss_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> image, result_image;
  std::vector<double> kernel;
  size_t height, width;
};

}  // namespace chistov_gauss_seq
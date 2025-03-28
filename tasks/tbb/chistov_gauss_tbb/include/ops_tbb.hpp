#pragma once

#include <utility>
#include <vector>
#include <numeric>
#include <random>
#include <iostream>

#include "core/task/include/task.hpp"

namespace chistov_gauss_tbb {

class TestTaskTBB : public ppc::core::Task {
 public:
  explicit TestTaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> image, result_image;
  std::vector<double> kernel;
  size_t height, width;
};

class TestTaskOpenMP : public ppc::core::Task {
 public:
  explicit TestTaskOpenMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> image, result_image;
  std::vector<double> kernel;
  size_t height, width;
};

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

}  // namespace chistov_gauss_tbb
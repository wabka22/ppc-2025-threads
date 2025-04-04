#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace bessonov_e_radix_sort_simple_merging_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
};

}  // namespace bessonov_e_radix_sort_simple_merging_seq
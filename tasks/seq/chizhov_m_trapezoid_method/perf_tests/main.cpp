#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/chizhov_m_trapezoid_method/include/ops_seq.hpp"

TEST(chizhov_m_trapezoid_method_seq, test_pipeline_run) {
  int div = 300;
  int dim = 3;
  std::vector<double> limits = {0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) {
    return std::cos((f_val[0] * f_val[0]) + (f_val[1] * f_val[1]) + (f_val[2] * f_val[2]));
  };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_seq->inputs_count.emplace_back(sizeof(div));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs_count.emplace_back(sizeof(dim));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(limits.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size() * sizeof(double));

  // Create Task
  auto test_task_sequential = std::make_shared<chizhov_m_trapezoid_method_seq::TestTaskSequential>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_NEAR(57972, res[0], 0.3);
}

TEST(chizhov_m_trapezoid_method_seq, test_task_run) {
  int div = 300;
  int dim = 3;
  std::vector<double> limits = {0.0, 1000.0, 0.0, 1000.0, 0.0, 1000.0};

  std::vector<double> res(1, 0);
  auto f = [](const std::vector<double> &f_val) {
    return std::cos((f_val[0] * f_val[0]) + (f_val[1] * f_val[1]) + (f_val[2] * f_val[2]));
  };
  auto *f_object = new std::function<double(const std::vector<double> &)>(f);

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&div));
  task_data_seq->inputs_count.emplace_back(sizeof(div));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
  task_data_seq->inputs_count.emplace_back(sizeof(dim));

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(limits.data()));
  task_data_seq->inputs_count.emplace_back(limits.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data_seq->outputs_count.emplace_back(res.size() * sizeof(double));

  // Create Task
  auto test_task_sequential = std::make_shared<chizhov_m_trapezoid_method_seq::TestTaskSequential>(task_data_seq);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_NEAR(57972, res[0], 0.3);
}

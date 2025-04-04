#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/filateva_e_simpson/include/ops_seq.hpp"

TEST(filateva_e_simpson_seq, test_pipeline_run) {
  size_t mer = 1;
  size_t steps = 10000000;
  std::vector<double> a = {1};
  std::vector<double> b = {1000};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0];
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  auto test_task_sequential = std::make_shared<filateva_e_simpson_seq::Simpson>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  filateva_e_simpson_seq::Func integral_f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0] * x[0] / 3;
  };

  ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
}

TEST(filateva_e_simpson_seq, test_task_run) {
  size_t mer = 1;
  size_t steps = 10000000;
  std::vector<double> a = {1};
  std::vector<double> b = {1000};
  std::vector<double> res(1, 0);
  filateva_e_simpson_seq::Func f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0];
  };

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(f));
  task_data->inputs_count.emplace_back(mer);
  task_data->inputs_count.emplace_back(steps);

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  task_data->outputs_count.emplace_back(1);

  auto test_task_sequential = std::make_shared<filateva_e_simpson_seq::Simpson>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  filateva_e_simpson_seq::Func integral_f = [](std::vector<double> x) {
    if (x.empty()) {
      return 0.0;
    }
    return x[0] * x[0] * x[0] / 3;
  };

  ASSERT_NEAR(res[0], integral_f(b) - integral_f(a), 0.01);
}

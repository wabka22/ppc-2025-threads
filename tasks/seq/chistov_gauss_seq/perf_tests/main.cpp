#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/example/include/ops_seq.hpp"

TEST(chistov_gauss_seq, test_pipeline_run) {
  const size_t width = 5000;
  const size_t height = 5000;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width / 2; ++j) {
      in[(i * width) + j] = 0.0;
    }
    for (size_t j = width / 2; j < width; ++j) {
      in[(i * width) + j] = 1.0;
    }
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential =
      std::make_shared<chistov_gauss_seq::TestTaskSequential>(task_data_seq);

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
}

TEST(chistov_gauss_seq, test_task_run) {
  const size_t width = 5000;
  const size_t height = 5000;

  // Create data
  std::vector<double> in(width * height, 1);
  std::vector<double> out(width * height, 0);
  std::vector<double> ans(width * height, 1);

  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width / 2; ++j) {
      in[(i * width) + j] = 0.0;
    }
    for (size_t j = width / 2; j < width; ++j) {
      in[(i * width) + j] = 1.0;
    }
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_sequential = std::make_shared<chistov_gauss_seq::TestTaskSequential>(task_data_seq);

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
}

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/chistov_gauss_seq/include/ops_seq.hpp"

TEST(chistov_gauss_seq, test_pipeline_run) {
  const size_t width = 5000;
  const size_t height = 5000;

  std::vector<double> input_image(width * height, 255.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_image(width * height, 255.0);
  std::vector<double> kernel = {1, 2, 1};

  for (size_t i = 0; i < height; ++i) {
    expected_image[(i * width)] = 191.25;
    expected_image[((i + 1) * width) - 1] = 191.25;
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

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

  ASSERT_EQ(output_image, expected_image);
}

TEST(chistov_gauss_seq, test_task_run) {
  const size_t width = 5000;
  const size_t height = 5000;

  std::vector<double> input_image(width * height, 255.0);
  std::vector<double> output_image(width * height, 0.0);
  std::vector<double> expected_image(width * height, 255.0);
  std::vector<double> kernel = {1, 2, 1};

  for (size_t i = 0; i < height; ++i) {
    expected_image[(i * width)] = 191.25;
    expected_image[((i + 1) * width) - 1] = 191.25;
  }

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_seq->outputs_count.emplace_back(output_image.size());

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
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(output_image, expected_image);
}
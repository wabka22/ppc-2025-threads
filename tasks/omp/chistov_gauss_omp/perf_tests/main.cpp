#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "omp/chistov_gauss_omp/include/ops_omp.hpp"

namespace chistov_gauss_omp_test {

void CompareImages(const std::vector<double>& output_image, const std::vector<double>& expected_image, size_t width,
                   size_t height, double tolerance = 1e-6) {
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      ASSERT_NEAR(output_image[(i * width) + j], expected_image[(i * width) + j], tolerance);
    }
  }
}

std::vector<double> GenerateRandomImage(size_t width, size_t height, double min_val = 0.0, double max_val = 255.0) {
  std::vector<double> image(width * height);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min_val, max_val);

  for (size_t i = 0; i < width * height; ++i) {
    image[i] = dis(gen);
  }

  return image;
}

}  // namespace chistov_gauss_omp_test



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
  auto test_task_sequential = std::make_shared<chistov_gauss_omp::TestTaskSequential>(task_data_seq);

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
  
  chistov_gauss_omp_test::CompareImages(output_image, expected_image, width, height);
}

TEST(chistov_gauss_omp, test_task_run) {
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
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_omp->inputs_count.emplace_back(input_image.size());
  task_data_omp->inputs_count.emplace_back(width);
  task_data_omp->inputs_count.emplace_back(height);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image.data()));
  task_data_omp->outputs_count.emplace_back(output_image.size());

  // Create Task
  auto test_task = std::make_shared<chistov_gauss_omp::TestTaskOpenMP>(task_data_omp);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  chistov_gauss_omp_test::CompareImages(output_image, expected_image, width, height);
}

TEST(chistov_gauss_compare, test_task_run) {
  const size_t width = 5000;
  const size_t height = 5000;

  std::vector<double> input_image = chistov_gauss_omp_test::GenerateRandomImage(width, height);
  std::vector<double> output_image_seq(width * height, 0.0);
  std::vector<double> output_image_omp(width * height, 0.0);
  std::vector<double> kernel = {1, 2, 1};

  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_seq->inputs_count.emplace_back(input_image.size());
  task_data_seq->inputs_count.emplace_back(width);
  task_data_seq->inputs_count.emplace_back(height);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image_seq.data()));
  task_data_seq->outputs_count.emplace_back(output_image_seq.size());

  chistov_gauss_omp::TestTaskSequential image_filter_sequential(task_data_seq);
  ASSERT_EQ(image_filter_sequential.Validation(), true);
  image_filter_sequential.PreProcessing();
  image_filter_sequential.Run();
  image_filter_sequential.PostProcessing();

    // Create task_data
  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_image.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(kernel.data()));
  task_data_omp->inputs_count.emplace_back(input_image.size());
  task_data_omp->inputs_count.emplace_back(width);
  task_data_omp->inputs_count.emplace_back(height);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_image_omp.data()));
  task_data_omp->outputs_count.emplace_back(output_image_omp.size());

  // Create Task
  auto test_task = std::make_shared<chistov_gauss_omp::TestTaskOpenMP>(task_data_omp);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  chistov_gauss_omp_test::CompareImages(output_image_omp, output_image_seq, width, height);
}
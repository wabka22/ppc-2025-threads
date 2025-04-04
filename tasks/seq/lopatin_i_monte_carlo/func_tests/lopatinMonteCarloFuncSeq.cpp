#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/lopatin_i_monte_carlo/include/lopatinMonteCarloSeq.hpp"

namespace lopatin_i_monte_carlo_seq {

std::vector<double> GenerateBounds(double min_val, double max_val, int dimensions) {
  std::vector<double> bounds;
  for (int i = 0; i < dimensions; ++i) {
    bounds.push_back(min_val);
    bounds.push_back(max_val);
  }
  return bounds;
}
}  // namespace lopatin_i_monte_carlo_seq

TEST(lopatin_i_monte_carlo_seq, validationInvalidInputOddBoundsCount) {
  std::vector<double> bounds = {0.0, 1.0, 2.0};  // even num of bounds
  const int iterations = 10;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());  // incorrect num of inputs
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, [](const std::vector<double>&) { return 1.0; });
  ASSERT_FALSE(task.Validation());
}

TEST(lopatin_i_monte_carlo_seq, validationMissingOutputData) {
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(0.0, 1.0, 2);
  const int iterations = 10;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(4);
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, [](const std::vector<double>&) { return 1.0; });
  ASSERT_FALSE(task.Validation());
}

TEST(lopatin_i_monte_carlo_seq, highDimensionalIntegration) {
  const int dimensions = 7;
  const int iterations = 20000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(-1.0, 1.0, dimensions);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, [](const std::vector<double>&) {
    return 1.0;  // const
  });

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const double expected = std::pow(2.0, dimensions);  // hypercube
  const double tolerance = 0.05 * expected;
  EXPECT_NEAR(result, expected, tolerance);  // error 5%
}

TEST(lopatin_i_monte_carlo_seq, 1DConstantFunction) {
  const int dimensions = 1;
  const int iterations = 60000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(2.0, 5.0, dimensions);  // [2, 5]

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, [](const std::vector<double>& x) {
    return 1.0;  // f(x) = 1
  });

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const double expected = 3.0;
  const double tolerance = 0.05 * expected;
  EXPECT_NEAR(result, expected, tolerance);  // error 5%
}

TEST(lopatin_i_monte_carlo_seq, 3DExponentialFunction) {
  const int dimensions = 3;
  const int iterations = 60000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(0.0, 1.0, dimensions);  // [0,1]^3

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);
  lopatin_i_monte_carlo_seq::TestTaskSequential task(
      task_data, [](const std::vector<double>& x) { return std::pow(std::numbers::e, x[0] + x[1] + x[2]); });

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const double expected = std::pow(std::numbers::e - 1, 3);  // = 5.073
  const double tolerance = 0.05 * expected;
  EXPECT_NEAR(result, expected, tolerance);  // error 5%
}

TEST(lopatin_i_monte_carlo_seq, 2DLinearFunction) {
  const int dimensions = 2;
  const int iterations = 40000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(0.0, 1.0, dimensions);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  auto function = [](const std::vector<double>& x) {
    assert(x.size() == 2);
    return x[0] + x[1];
  };

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const double expected = 1.0;
  const double tolerance = 0.05 * expected;
  EXPECT_NEAR(result, expected, tolerance);  // error 5%
}

TEST(lopatin_i_monte_carlo_seq, 3DProductFunction) {
  const int dimensions = 3;
  const int iterations = 40000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(0.0, 1.0, dimensions);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data,
                                                     [](const std::vector<double>& x) { return x[0] * x[1] * x[2]; });

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  const double expected = 0.125;
  const double tolerance = 0.05 * expected;
  EXPECT_NEAR(result, expected, tolerance);  // error 5%
}

TEST(lopatin_i_monte_carlo_seq, 4DQuadraticFunction) {
  const int dimensions = 4;
  const int iterations = 60000;  // increase for 4D
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(0.0, 1.0, dimensions);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  // (x1 + x2 + x3 + x4)^2
  auto function = [](const std::vector<double>& x) {
    double sum = x[0] + x[1] + x[2] + x[3];
    return sum * sum;
  };

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // anal 13/3 = 4.33333
  const double expected = 13.0 / 3.0;
  const double tolerance = 0.05 * expected;
  EXPECT_NEAR(result, expected, tolerance);  // error 5%
}

TEST(lopatin_i_monte_carlo_seq, 2DCosineFunction) {
  const int dimensions = 2;
  const int iterations = 100000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(0.0, std::numbers::pi / 2, dimensions);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  // cos(x + y)
  auto function = [](const std::vector<double>& x) { return std::cos(x[0] + x[1]); };

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // analytical = 0
  const double expected = 0.0;
  const double tolerance = 0.03;
  EXPECT_NEAR(result, expected, tolerance);  // error 5%
}

TEST(lopatin_i_monte_carlo_seq, 2DSqrtFunction) {
  const int dimensions = 2;
  const int iterations = 30000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(0.0, 1.0, dimensions);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  // sqrt(x + y)
  auto function = [](const std::vector<double>& x) { return std::sqrt(x[0] + x[1]); };

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // analytical = 0.975
  const double expected = 0.975;
  const double tolerance = 0.01 * expected;
  EXPECT_NEAR(result, expected, tolerance);  // error 1%
}

TEST(lopatin_i_monte_carlo_seq, 3DSinFunction) {
  const int dimensions = 3;
  const int iterations = 40000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(0.0, std::numbers::pi / 6, dimensions);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  // sin(x + y + z)
  auto function = [](const std::vector<double>& x) { return std::sin(x[0] + x[1] + x[2]); };

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // analytical = 0.098
  const double expected = 0.098;
  const double tolerance = 0.05 * expected;
  EXPECT_NEAR(result, expected, tolerance);  // error 5%
}

TEST(lopatin_i_monte_carlo_seq, 4DLogFunction) {
  const int dimensions = 4;
  const int iterations = 50000;
  std::vector<double> bounds = lopatin_i_monte_carlo_seq::GenerateBounds(1.0, std::numbers::e, dimensions);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(bounds.data()));
  task_data->inputs_count.push_back(bounds.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&iterations)));
  task_data->inputs_count.push_back(1);

  double result = 0.0;
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(&result));
  task_data->outputs_count.push_back(1);

  // ln(x1 + x2 + x3 + x4)
  auto function = [](const std::vector<double>& x) { return std::log(x[0] + x[1] + x[2] + x[3]); };

  lopatin_i_monte_carlo_seq::TestTaskSequential task(task_data, function);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // analytical = 17.4108
  const double expected = 17.4108;
  const double tolerance = 0.05 * expected;
  EXPECT_NEAR(result, expected, tolerance);  // error 5%
}

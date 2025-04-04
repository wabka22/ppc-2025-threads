#include <gtest/gtest.h>

#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/plekhanov_d_dijkstra/include/ops_seq.hpp"

namespace plekhanov_d_dijkstra_seq {

static std::vector<int> CalculateExpectedResult(                       // NOLINT(misc-use-anonymous-namespace)
    const std::vector<std::vector<std::pair<size_t, int>>> &adj_list,  // NOLINT(misc-use-anonymous-namespace)
    size_t start_vertex) {                                             // NOLINT(misc-use-anonymous-namespace)
  size_t n = adj_list.size();
  const int inf = INT_MAX;
  std::vector<int> distances(n, inf);
  distances[start_vertex] = 0;

  using Pii = std::pair<int, size_t>;
  std::priority_queue<Pii, std::vector<Pii>, std::greater<>> pq;
  pq.emplace(0, start_vertex);

  while (!pq.empty()) {
    auto [d, u] = pq.top();
    pq.pop();

    if (d != distances[u]) {
      continue;
    }

    for (const auto &edge : adj_list[u]) {
      size_t v = edge.first;
      int weight = edge.second;
      if (distances[u] != inf && (distances[u] + weight < distances[v])) {
        distances[v] = distances[u] + weight;
        pq.emplace(distances[v], v);
      }
    }
  }
  return distances;
}

}  // namespace plekhanov_d_dijkstra_seq

TEST(plekhanov_d_dijkstra_seq, test_pipeline_run) {
  constexpr size_t kNumVertices = 6000;
  size_t start_vertex = 0;

  std::vector<std::vector<std::pair<size_t, int>>> adj_list(kNumVertices);
  for (size_t i = 0; i < kNumVertices; ++i) {
    for (size_t j = 0; j < kNumVertices; ++j) {
      if (i != j) {
        if (rand() % 3 == 0) {
          int weight = (rand() % 10) + 1;
          adj_list[i].emplace_back(j, weight);
        }
      }
    }
  }

  std::vector<int> graph_data;
  for (const auto &vertex_edges : adj_list) {
    for (const auto &edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }

  std::vector<int> distances(kNumVertices, INT_MAX);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph_data.data()));
  task_data_seq->inputs_count.emplace_back(graph_data.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&start_vertex));
  task_data_seq->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(distances.data()));
  task_data_seq->outputs_count.emplace_back(kNumVertices);

  auto test_task_sequential = std::make_shared<plekhanov_d_dijkstra_seq::TestTaskSequential>(task_data_seq);

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
  std::vector<int> expected = plekhanov_d_dijkstra_seq::CalculateExpectedResult(adj_list, start_vertex);
  EXPECT_EQ(distances, expected);
}

TEST(plekhanov_d_dijkstra_seq, test_task_run) {
  constexpr size_t kNumVertices = 6000;
  size_t start_vertex = 0;

  std::vector<std::vector<std::pair<size_t, int>>> adj_list(kNumVertices);
  for (size_t i = 0; i < kNumVertices; ++i) {
    for (size_t j = 0; j < kNumVertices; ++j) {
      if (i != j) {
        if (rand() % 3 == 0) {
          int weight = (rand() % 10) + 1;
          adj_list[i].emplace_back(j, weight);
        }
      }
    }
  }

  std::vector<int> graph_data;
  for (const auto &vertex_edges : adj_list) {
    for (const auto &edge : vertex_edges) {
      graph_data.push_back(static_cast<int>(edge.first));
      graph_data.push_back(edge.second);
    }
    graph_data.push_back(-1);
  }

  std::vector<int> distances(kNumVertices, INT_MAX);

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(graph_data.data()));
  task_data_seq->inputs_count.emplace_back(graph_data.size());

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&start_vertex));
  task_data_seq->inputs_count.emplace_back(sizeof(start_vertex));

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(distances.data()));
  task_data_seq->outputs_count.emplace_back(kNumVertices);

  auto test_task_sequential = std::make_shared<plekhanov_d_dijkstra_seq::TestTaskSequential>(task_data_seq);

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
  std::vector<int> expected = plekhanov_d_dijkstra_seq::CalculateExpectedResult(adj_list, start_vertex);
  EXPECT_EQ(distances, expected);
}
#include <format>
#include <iostream>

#include "micrograd/nn.h"

namespace micrograd {

namespace {
Value SumOfSquares(std::span<const Value> expected,
                   std::span<const Value> predicted) {
  Value loss = Value(0.0);
  for (size_t i = 0; i < expected.size(); ++i) {
    loss = loss.Add(predicted[i].Subtract(expected[i]).Pow(2));
  }
  return loss;
}
}  // namespace

void Run() {
  auto stdout = std::ostreambuf_iterator<char>(std::cout);
  std::vector<std::vector<Value>> inputs = {
      {Value(2), Value(3), Value(-1)},
      {Value(3), Value(-1), Value(0.5)},
      {Value(0.5), Value(1.0), Value(1.0)},
      {Value(1), Value(1), Value(-1)},
  };
  std::vector<Value> targets = {Value(1), Value(-1), Value(-1), Value(1)};
  std::array layers = std::to_array<size_t>({4, 4, 1});
  auto n = MLP(3, layers);
  for (size_t i = 0; i < 500; ++i) {
    // Update
    for (Value& p : n.Parameters()) {
      p.value(p.value() + (-0.005 * p.gradient()));
    }
    // Compute forward pass
    std::vector<Value> outputs;
    outputs.reserve(inputs.size());
    for (const auto& input : inputs) {
      auto output = n(input);
      outputs.push_back(std::move(output.front()));
    }
    Value loss = SumOfSquares(targets, outputs);
    for (Value& p : n.Parameters()) {
      p.gradient(0.0);
    }
    loss.Backward();
    std::format_to(stdout, "loss: {}\n", loss.value());
    for (const Value& output : outputs) {
      std::format_to(stdout, "output value: {}\n", output.value());
    }
    std::format_to(stdout, "\n\n");
  }
}

}  // namespace micrograd

int main() { micrograd::Run(); }

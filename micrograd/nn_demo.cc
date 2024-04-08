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
  std::vector<std::vector<Value>> inputs = {
      {Value(2), Value(3), Value(-1)},
      {Value(3), Value(-1), Value(0.5)},
      {Value(0.5), Value(1.0), Value(1.0)},
      {Value(1), Value(1), Value(-1)},
  };
  std::vector<Value> targets = {Value(1), Value(-1), Value(-1), Value(1)};
  std::array layers = std::to_array<size_t>({4, 4, 1});
  auto n = MLP(3, layers);
  std::vector<Value> outputs;
  outputs.reserve(inputs.size());
  for (const auto& input : inputs) {
    auto output = n(input);
    outputs.push_back(std::move(output.front()));
  }
  Value loss = SumOfSquares(targets, outputs);
  loss.Backward();
  std::format_to(std::ostreambuf_iterator<char>(std::cout), "loss: {}\n",
                 loss.value());
}

}  // namespace micrograd

int main() { micrograd::Run(); }

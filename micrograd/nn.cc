#include "micrograd/nn.h"

#include <algorithm>
#include <iterator>
#include <random>

namespace micrograd {

namespace {

float random_float(float start, float end) {
  thread_local static auto rng = [] {
    std::seed_seq seed{3, 2, 4, 1};
    return std::mt19937(seed);
  }();
  std::uniform_real_distribution<float> dist(start, end);
  return dist(rng);
}

}  // namespace

Neuron::Neuron(size_t number_of_inputs, bool nonlinear)
    : bias_(random_float(-1, 1)), nonlinear_(nonlinear) {
  weights_.reserve(number_of_inputs);
  std::generate_n(std::back_inserter(weights_), number_of_inputs,
                  [] { return Value(random_float(-1, 1)); });
}

Value Neuron::operator()(std::span<const Value> x) const {
  Value v = bias_;
  for (size_t i = 0; i < x.size(); ++i) {
    v = v.Add(weights_[i].Multiply(x[i]));
  }
  if (nonlinear_) {
    return v.Relu();
  }
  return v;
}

std::vector<Value> Neuron::Parameters() const {
  std::vector<Value> p;
  p.reserve(weights_.size() + 1);
  std::copy(weights_.begin(), weights_.end(), std::back_inserter(p));
  p.push_back(bias_);
  return p;
}

Layer::Layer(size_t number_of_inputs, size_t number_of_outputs,
             bool nonlinear) {
  neurons_.reserve(number_of_outputs);
  for (size_t i = 0; i < number_of_outputs; ++i) {
    neurons_.emplace_back(number_of_inputs, nonlinear);
  }
}

std::vector<Value> Layer::operator()(std::span<const Value> x) const {
  std::vector<Value> outs;
  outs.reserve(neurons_.size());
  for (const auto& n : neurons_) {
    outs.push_back(n(x));
  }
  return outs;
}

std::vector<Value> Layer::Parameters() const {
  std::vector<Value> output;
  for (const auto& n : neurons_) {
    for (const auto& p : n.Parameters()) {
      output.push_back(p);
    }
  }
  return output;
}

MLP::MLP(size_t number_of_inputs, std::span<size_t> number_of_outputs) {
  layers_.reserve(number_of_outputs.size() + 1);
  size_t prev = number_of_inputs;
  for (size_t i = 0; size_t output_size : number_of_outputs) {
    bool nonlinear = ++i != number_of_outputs.size();
    layers_.emplace_back(prev, output_size, nonlinear);
    prev = output_size;
  }
}

std::vector<Value> MLP::operator()(std::span<const Value> x) const {
  // Hold the memory in the current evaluation pass here,
  // to make sure x always points to valid memory.
  std::vector<Value> current;
  for (const auto& layer : layers_) {
    current = layer(x);
    x = current;
  }
  return current;
}

std::vector<Value> MLP::Parameters() const {
  std::vector<Value> outputs;
  for (const auto& layer : layers_) {
    for (const auto& p : layer.Parameters()) {
      outputs.push_back(p);
    }
  }
  return outputs;
}
}  // namespace micrograd

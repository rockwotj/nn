#pragma once

#include <span>
#include <vector>

#include "micrograd/micrograd.h"

namespace micrograd {

//
class Neuron {
 public:
  //
  explicit Neuron(size_t number_of_inputs, bool nonlinear = true);

  //
  Value operator()(std::span<const Value> x) const;

  std::vector<Value> Parameters() const;

 private:
  std::vector<Value> weights_;
  Value bias_;
  bool nonlinear_;
};

//
class Layer {
 public:
  //
  Layer(size_t number_of_inputs, size_t number_of_outputs,
        bool nonlinear = true);

  //
  std::vector<Value> operator()(std::span<const Value> x) const;

  std::vector<Value> Parameters() const;

 private:
  std::vector<Neuron> neurons_;
};

//
class MLP {
 public:
  //
  MLP(size_t number_of_inputs, std::span<size_t> number_of_outputs);
  MLP(size_t number_of_inputs, std::vector<size_t> number_of_outputs)
      : MLP(number_of_inputs, std::span(number_of_outputs)) {}

  //
  std::vector<Value> operator()(std::span<const Value> x) const;

  std::vector<Value> Parameters() const;

 private:
  std::vector<Layer> layers_;
};

}  // namespace micrograd

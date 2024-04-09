#pragma once

#include <span>
#include <vector>

#include "micrograd/micrograd.h"

namespace micrograd {

//
class Neuron {
 public:
  //
  explicit Neuron(size_t number_of_inputs);

  //
  Value operator()(std::span<const Value> x) const;

  std::vector<Value> Parameters() const;

 private:
  std::vector<Value> weights_;
  Value bias_;
};

//
class Layer {
 public:
  //
  Layer(size_t number_of_inputs, size_t number_of_outputs);

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

  //
  std::vector<Value> operator()(std::span<const Value> x) const;

  std::vector<Value> Parameters() const;

 private:
  std::vector<Layer> layers_;
};

}  // namespace micrograd

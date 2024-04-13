#pragma once

#include <span>
#include <vector>

#include "micrograd/micrograd.h"

namespace micrograd {

// An mathmatical model of a individual neuron.
class Neuron {
 public:
  //
  explicit Neuron(size_t number_of_inputs, bool nonlinear = true);

  // Compute the forward pass of this neuron using x as the input.
  //
  // It's length must match `number_of_inputs` from the constructor.
  Value operator()(std::span<const Value> x) const;

  // All the weights of this neuron.
  std::vector<Value> Parameters() const;

 private:
  std::vector<Value> weights_;
  Value bias_;
  bool nonlinear_;
};

// A layer of identical neurons in a neural network.
class Layer {
 public:
  // The size of this layer, interms of inputs and outputs.
  Layer(size_t number_of_inputs, size_t number_of_outputs,
        bool nonlinear = true);

  // Compute the forward pass of this neuron using x as the input.
  //
  // It's length must match `number_of_inputs` from the constructor.
  // The output vector will be of size `number_of_outputs`.
  std::vector<Value> operator()(std::span<const Value> x) const;

  // All the weights for all the neurons in this layer.
  std::vector<Value> Parameters() const;

 private:
  std::vector<Neuron> neurons_;
};

// A multi-layer precepticon.
class MLP {
 public:
  // Create the MLP with the inputs and number of outputs at each layer.
  MLP(size_t number_of_inputs, std::span<size_t> number_of_outputs);
  MLP(size_t number_of_inputs, std::vector<size_t> number_of_outputs)
      : MLP(number_of_inputs, std::span(number_of_outputs)) {}

  // Compute the forward pass of this neuron using x as the input.
  //
  // It's length must match `number_of_inputs` from the constructor.
  // The output vector will be of size last element in `number_of_outputs`.
  std::vector<Value> operator()(std::span<const Value> x) const;

  // all the weights for all layers in this MLP.
  std::vector<Value> Parameters() const;

 private:
  std::vector<Layer> layers_;
};

}  // namespace micrograd

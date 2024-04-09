#include "micrograd/micrograd.h"

#include <gtest/gtest.h>
#include <torch/nn.h>

namespace micrograd {

TEST(MicrogradValue, SimpleExpression) {
  {
    auto a = Value(2);
    auto b = Value(-3);
    auto c = Value(10);
    auto e = a.Multiply(b);
    auto d = e.Add(c);
    auto f = Value(-2);
    auto out = d.Multiply(f);
    EXPECT_FLOAT_EQ(out.value(), -8.0);
    out.Backward();
    EXPECT_FLOAT_EQ(a.gradient(), 6.0);
    EXPECT_FLOAT_EQ(b.gradient(), -4.0);
  }
  {
    auto a = torch::tensor({2.0f}, torch::requires_grad());
    auto b = torch::tensor({-3.0f}, torch::requires_grad());
    auto c = torch::tensor(10.0f);
    auto e = a.multiply(b);
    auto d = e.add(c);
    auto f = torch::tensor(-2.0f);
    auto out = d.multiply(f);
    EXPECT_FLOAT_EQ(out.data().item().toFloat(), -8.0);
    out.backward();
    EXPECT_FLOAT_EQ(a.grad().data().item().toFloat(), 6.0);
    EXPECT_FLOAT_EQ(b.grad().data().item().toFloat(), -4.0);
  }
}

TEST(MicrogradValue, Neuron) {
  {
    auto x1 = Value(2);
    auto x2 = Value(0);
    auto w1 = Value(-3);
    auto w2 = Value(1.0);
    auto b = Value(6.013735870195432);
    auto n = x1.Multiply(w1).Add(x2.Multiply(w2)).Add(b);
    auto out = n.Relu();
    EXPECT_FLOAT_EQ(out.value(), 0.013735771);
    out.Backward();
    EXPECT_FLOAT_EQ(x1.gradient(), -3);
    EXPECT_FLOAT_EQ(x2.gradient(), 1);
    EXPECT_FLOAT_EQ(w1.gradient(), 2);
    EXPECT_FLOAT_EQ(w2.gradient(), 0);
  }
  {
    auto x1 = torch::tensor({2.0f}, torch::requires_grad());
    auto x2 = torch::tensor({0.0f}, torch::requires_grad());
    auto w1 = torch::tensor({-3.0f}, torch::requires_grad());
    auto w2 = torch::tensor({1.0f}, torch::requires_grad());
    auto b = torch::tensor({6.013735870195432});
    auto n = x1.multiply(w1).add(x2.multiply(w2)).add(b);
    auto out = n.relu();
    EXPECT_FLOAT_EQ(out.data().item().toFloat(), 0.013735771);
    out.backward();
    EXPECT_FLOAT_EQ(x1.grad().item().toFloat(), -3);
    EXPECT_FLOAT_EQ(x2.grad().item().toFloat(), 1);
    EXPECT_FLOAT_EQ(w1.grad().item().toFloat(), 2);
    EXPECT_FLOAT_EQ(w2.grad().item().toFloat(), 0);
  }
}

TEST(MicrogradValue, AllOps) {
  {
    auto a = Value(-4);
    auto b = Value(2);
    auto c = a.Add(b);
    auto d = a.Multiply(b).Add(b.Pow(3));
    c = c.Add(c).Add(1);
    c = c.Add(Value(1).Add(c).Add(a.Negate()));
    d = d.Add(d.Multiply(2).Add(b.Add(a).Relu()));
    d = d.Add(Value(3).Multiply(d).Add(b.Subtract(a).Relu()));
    auto e = c.Subtract(d);
    auto f = e.Pow(2.0);
    auto g = f.Divide(2.0);
    g = g.Add(Value(10.0).Divide(f));
    g.Backward();
    EXPECT_FLOAT_EQ(g.value(), 24.704082);
    EXPECT_FLOAT_EQ(a.gradient(), 138.83382);
    EXPECT_FLOAT_EQ(b.gradient(), 645.5773);
  }
  {
    auto a = torch::tensor({-4.0f}, torch::requires_grad());
    auto b = torch::tensor({2.0f}, torch::requires_grad());
    auto c = a + b;
    auto d = a * b + b.pow(3.0f);
    c = c + c + 1.0f;
    c = c + 1.0f + c + (-a);
    d = d + d * 2.0f + (b + a).relu();
    d = d + 3.0f * d + (b - a).relu();
    auto e = c - d;
    auto f = e.pow(2.0f);
    auto g = f / 2.0f;
    g = g + 10.0 / f;
    g.backward();
    EXPECT_FLOAT_EQ(g.item().toFloat(), 24.704082);
    EXPECT_FLOAT_EQ(a.grad().item().toFloat(), 138.83382);
    EXPECT_FLOAT_EQ(b.grad().item().toFloat(), 645.57727);
  }
}

}  // namespace micrograd

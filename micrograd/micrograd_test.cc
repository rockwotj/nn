#include "micrograd/micrograd.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

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

}  // namespace micrograd

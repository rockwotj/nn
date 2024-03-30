#pragma once

#include <utility>

namespace micrograd {

class Value {
public:
  template <typename H> friend H AbslHashValue(H h, const Value &v) {
    return H::combine(std::move(h), v.value_);
  }

private:
  float value_;
};

} // namespace micrograd

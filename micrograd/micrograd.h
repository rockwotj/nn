#pragma once

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"

namespace micrograd {

class Value {
 public:
  explicit Value(float data);

  Value operator+(Value* other);
  Value operator*(Value* other);

  void backward();

  template <typename H>
  friend H AbslHashValue(H h, const Value& v) {
    return H::combine(std::move(h), v.value_, v.grad_, v.backward_,
                      v.children_);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Value& p) {
    absl::Format(&sink, "Value(value=%f, grad=%f, children=%v)", p.value_,
                 p.grad_, p.children_);
  }

  const absl::flat_hash_set<Value*>& children() const;

 private:
  Value(float data, absl::flat_hash_set<Value*> children);

  float value_;
  float grad_ = 0;
  absl::flat_hash_set<Value*> children_;
  absl::AnyInvocable<void()> backward_ = [] {};
};

}  // namespace micrograd

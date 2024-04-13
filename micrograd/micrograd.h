#pragma once

#include <memory>
#include <utility>

#include "absl/hash/hash.h"
#include "absl/strings/str_format.h"

namespace micrograd {

class ValueImpl;

/**
 * The underlying engine for creating mathmatical expressions that can back
 * propigate.
 *
 * This value class is a small wrapper over a shared pointer, so it is
 * copy-able, but the underlying value is still the same.
 */
class Value {
 public:
  explicit Value(float data);

  Value Add(const Value& other) const;
  Value Add(float other) const { return Add(Value(other)); }
  Value Subtract(const Value& other) const;
  Value Subtract(float other) const { return Subtract(Value(other)); }

  Value Multiply(const Value& other) const;
  Value Multiply(float other) const { return Multiply(Value(other)); }
  Value Divide(const Value& other) const;
  Value Divide(float other) const { return Divide(Value(other)); }

  Value Pow(float other) const;
  Value Negate() const;
  Value Relu() const;

  float value() const;
  void value(float);
  float gradient() const;
  void gradient(float);

  /**
   * Populate the gradient for this node and all it's children.
   */
  void Backward();

  template <typename H>
  friend H AbslHashValue(H h, const Value& v) {
    HashValue(absl::HashState::Create(&h), *v.impl_.get());
    return std::move(h);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Value& p) {
    absl::Format(&sink, "%s", DebugString(*p.impl_.get()));
  }

  bool operator==(const Value&) const = default;

 private:
  explicit Value(std::shared_ptr<ValueImpl> impl);

  static void HashValue(absl::HashState state, const ValueImpl&);
  static std::string DebugString(const ValueImpl&);

  std::shared_ptr<ValueImpl> impl_;
};

}  // namespace micrograd

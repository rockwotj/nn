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
 */
class Value {
 public:
  explicit Value(float data);

  Value Add(const Value& other);
  Value Add(float other) { return Add(Value(other)); }

  Value Multiply(const Value& other);
  Value Multiply(float other) { return Multiply(Value(other)); }

  float value() const;
  float gradient() const;

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

#include "micrograd/micrograd.h"

#include <memory>

#include "absl/container/flat_hash_set.h"

namespace micrograd {

namespace {
enum class Op : char {
  kNone = ' ',
  kAdd = '+',
  kMultiply = '*',
  kPow = '^',
  kReLU = '?',
};
}

class ValueImpl : public std::enable_shared_from_this<ValueImpl> {
  using ChildrenSet = absl::flat_hash_set<std::shared_ptr<ValueImpl>>;

 public:
  ValueImpl(float val) : value_(val) {}
  ValueImpl(float val, ChildrenSet children, Op op)
      : value_(val), children_(children), op_(op) {}

  std::shared_ptr<ValueImpl> Add(std::shared_ptr<ValueImpl> other) {
    auto out = std::make_shared<ValueImpl>(
        value_ + other->value_, ChildrenSet({shared_from_this(), other}),
        Op::kAdd);
    out->backward_ = [this, other = other.get(), out = out.get()] {
      this->grad_ += out->grad_;
      other->grad_ += out->grad_;
    };
    return out;
  }

  std::shared_ptr<ValueImpl> Multiply(std::shared_ptr<ValueImpl> other) {
    auto out = std::make_shared<ValueImpl>(
        value_ * other->value_, ChildrenSet({shared_from_this(), other}),
        Op::kMultiply);
    out->backward_ = [this, other = other.get(), out = out.get()] {
      this->grad_ += other->value_ * out->grad_;
      other->grad_ += this->value_ * out->grad_;
    };
    return out;
  }

  std::shared_ptr<ValueImpl> Pow(float other) {
    auto out = std::make_shared<ValueImpl>(
        std::pow(value_, other), ChildrenSet({shared_from_this()}), Op::kPow);
    out->backward_ = [this, other, out = out.get()] {
      this->grad_ += (other * std::pow(value_, other - 1)) * out->grad_;
    };
    return out;
  }

  std::shared_ptr<ValueImpl> Relu() {
    auto out = std::make_shared<ValueImpl>(
        value_ < 0 ? 0 : value_, ChildrenSet({shared_from_this()}), Op::kReLU);
    out->backward_ = [this, out = out.get()] {
      this->grad_ += out->value_ > 0 ? out->grad_ : 0;
    };
    return out;
  }

  void Backward() {
    std::vector<ValueImpl*> output;
    absl::flat_hash_set<ValueImpl*> visited;
    TopologicalSort(&output, &visited);
    grad_ = 1.0;
    for (ssize_t i = output.size() - 1; i >= 0; --i) {
      ValueImpl* v = output[i];
      v->backward_();
    }
  }

  float value() const { return value_; }
  void value(float v) { value_ = v; }
  float grad() const { return grad_; }
  void grad(float v) { grad_ = v; }

  std::string DebugString() const {
    std::string children_debug_string = "{";
    for (const auto& child : children_) {
      children_debug_string += child->DebugString();
    }
    children_debug_string += "}";
    return absl::StrFormat("Value(value=%f, grad=%f, op=%c, children=%s)",
                           value_, grad_, op_, children_debug_string);
  }

 private:
  void TopologicalSort(std::vector<ValueImpl*>* output,
                       absl::flat_hash_set<ValueImpl*>* visited) {
    auto [_, inserted] = visited->insert(this);
    if (!inserted) {
      return;
    }
    for (const auto& child : children_) {
      child->TopologicalSort(output, visited);
    }
    output->push_back(this);
  }

  float value_;
  float grad_ = 0.0;
  absl::flat_hash_set<std::shared_ptr<ValueImpl>> children_;
  absl::AnyInvocable<void()> backward_ = [] {};
  Op op_ = Op::kNone;
};

Value::Value(float data) : impl_(std::make_shared<ValueImpl>(data)) {}

Value Value::Add(const Value& other) const {
  return Value(impl_->Add(other.impl_));
}
Value Value::Subtract(const Value& other) const {
  return Value(impl_->Add(other.Negate().impl_));
}
Value Value::Multiply(const Value& other) const {
  return Value(impl_->Multiply(other.impl_));
}
Value Value::Divide(const Value& other) const {
  return Value(impl_->Multiply(other.Pow(-1).impl_));
}
Value Value::Pow(float other) const { return Value(impl_->Pow(other)); }
Value Value::Negate() const { return this->Multiply(-1); }
Value Value::Relu() const { return Value(impl_->Relu()); }
void Value::Backward() { impl_->Backward(); }
float Value::value() const { return impl_->value(); }
void Value::value(float v) { impl_->value(v); }
float Value::gradient() const { return impl_->grad(); }
void Value::gradient(float v) { impl_->grad(v); }

Value::Value(std::shared_ptr<ValueImpl> impl) : impl_(std::move(impl)) {}

void Value::HashValue(absl::HashState state, const ValueImpl& impl) {
  absl::HashState::combine(std::move(state), std::addressof(impl));
}

std::string Value::DebugString(const ValueImpl& impl) {
  return impl.DebugString();
}

}  // namespace micrograd

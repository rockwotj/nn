#include "micrograd/micrograd.h"

#include <memory>

#include "absl/container/flat_hash_set.h"

namespace micrograd {

namespace {
enum class Op : char {
  kNone = ' ',
  kAdd = '+',
  kMultiply = '*',
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
  float grad() const { return grad_; }

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

Value Value::Add(const Value& other) { return Value(impl_->Add(other.impl_)); }
Value Value::Multiply(const Value& other) {
  return Value(impl_->Multiply(other.impl_));
}
void Value::Backward() { impl_->Backward(); }
float Value::value() const { return impl_->value(); }
float Value::gradient() const { return impl_->grad(); }

Value::Value(std::shared_ptr<ValueImpl> impl) : impl_(std::move(impl)) {}

void Value::HashValue(absl::HashState state, const ValueImpl& impl) {
  absl::HashState::combine(std::move(state), std::addressof(impl));
}

std::string Value::DebugString(const ValueImpl& impl) {
  return impl.DebugString();
}

}  // namespace micrograd

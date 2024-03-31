#include "micrograd/micrograd.h"

namespace micrograd {

Value::Value(float data) : value_(data) {}

Value::Value(float data, absl::flat_hash_set<Value*> children)
    : value_(data), children_(std::move(children)) {}

Value Value::operator+(Value* other) {
  Value out = Value(value_ + other->value_, {this, other});
  out.backward_ = [this, other] {
    this->grad_ += other->grad_;
    other->grad_ += this->grad_;
  };
  return out;
}

Value Value::operator*(Value* other) {
  Value out = Value(value_ * other->value_, {this, other});
  out.backward_ = [this, other] {
    this->grad_ += other->value_ * other->grad_;
    other->grad_ += this->value_ * this->grad_;
  };
  return out;
}

namespace {
struct topological_visit {
  std::vector<Value*> topo;
  absl::flat_hash_set<Value*> visited;

  void build(Value* v) {
    auto [_, inserted] = visited.insert(v);
    if (!inserted) {
      return;
    }
    for (Value* child : v->children()) {
      build(child);
    }
    topo.push_back(v);
  };
};
}  // namespace

void Value::backward() {
  topological_visit visitor;
  visitor.build(this);
  grad_ = 1;
  for (size_t i = visitor.topo.size() - 1; i >= 0; --i) {
    Value* v = visitor.topo[i];
    v->backward();
  }
}

const absl::flat_hash_set<Value*>& Value::children() const { return children_; }

}  // namespace micrograd

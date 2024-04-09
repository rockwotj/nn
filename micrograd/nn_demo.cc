#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <plot/plot.hpp>
#include <stdexcept>

#include "micrograd/micrograd.h"
#include "micrograd/nn.h"

struct Dataset {
  std::vector<std::pair<float, float>> points;
  std::vector<float> classifications;

  static Dataset ParseFile(const std::filesystem::path& path) {
    std::ifstream f{path};
    nlohmann::json data = nlohmann::json::parse(f,
                                                /*cb=*/nullptr,
                                                /*allow_exceptions=*/true,
                                                /*ignore_comments=*/true);
    return Parse(data);
  }

  static Dataset Parse(const nlohmann::json& root) {
    Dataset ds;
    if (!root["data"].is_array()) {
      throw std::runtime_error("invalid points");
    }
    ds.points.reserve(root["data"].size());
    ds.classifications.reserve(root["data"].size());
    for (const auto& points : root["data"]) {
      if (!points.is_array() || points.size() != 2) {
        throw std::runtime_error("invalid point");
      }
      auto x = points[0].get<float>();
      auto y = points[1].get<float>();
      ds.points.emplace_back(x, y);
    }
    if (!root["classifications"].is_array()) {
      throw std::runtime_error("invalid classifications");
    }
    for (const auto& element : root["classifications"]) {
      if (!element.is_number()) {
        throw std::runtime_error("invalid classification");
      }
      ds.classifications.emplace_back(element.get<float>());
    }
    return ds;
  }
};

float EvaluatePoint(micrograd::MLP& model, plot::Pointf p) {
  using micrograd::Value;
  std::vector<Value> inputs = {Value(p.x), Value(-p.y)};
  Value score = model(inputs).front();
  return score.value();
}

void Draw(const Dataset& ds, micrograd::MLP& model) {
  using namespace plot;
  TerminalInfo term;
  term.detect();
  // Each Braille Canvas is made up of cells that are 2x4 points
  // Points are switched on and off individually, but color is stored
  // per cell.
  constexpr Coord canvasCellCols = 70;
  constexpr Coord canvasCellRows = 20;
  constexpr Size canvasCellSize(canvasCellCols, canvasCellRows);

  constexpr float aspectRatio =
      float(2 * canvasCellCols) / float(4 * canvasCellRows);

  constexpr Rectf realCanvasBounds({-2.0f, -2.0f}, {2.5f, 2.5f / aspectRatio});

  RealCanvas<BrailleCanvas> canvas(realCanvasBounds, canvasCellSize, term);

  for (size_t i = 0; i < ds.points.size(); ++i) {
    const auto& [x, y] = ds.points[i];
    auto p = Pointf(x, -y);
    float classification = ds.classifications[i];
    canvas.dot(classification > 0 ? palette::red : palette::blue, p);
  }
  canvas.line(palette::whitesmoke, {-3, 0}, {3, 0}, TerminalOp::ClipSrc);
  canvas.line(palette::whitesmoke, {0, -3}, {0, 3}, TerminalOp::ClipSrc);
  canvas.fill(
      palette::pink, canvas.bounds(),
      [&model](Pointf p) { return EvaluatePoint(model, p) > 0; },
      TerminalOp::ClipSrc);
  canvas.fill(
      palette::lightblue, canvas.bounds(),
      [&model](Pointf p) { return EvaluatePoint(model, p) <= 0; },
      TerminalOp::ClipSrc);

  std::cout << margin(frame(BorderStyle::Double, &canvas, term)) << std::flush;
}

int main() {
  auto training_data = Dataset::ParseFile("demo_input.json");
  using namespace micrograd;
  // 2 layer neural network
  auto model = MLP(2, std::vector<size_t>{16, 16, 1});
  std::cout << "number of parameters: " << model.Parameters().size() << "\n";

  constexpr size_t steps = 100;
  for (size_t k = 0; k < steps; ++k) {
    // Forward pass
    std::vector<Value> scores;
    scores.reserve(training_data.points.size());
    for (const auto& [x, y] : training_data.points) {
      std::vector<Value> inputs = {Value(x), Value(y)};
      Value score = model(inputs).front();
      scores.push_back(std::move(score));
    }
    // SVM "max-margin" loss
    std::vector<Value> losses;
    losses.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
      auto expected = Value(training_data.classifications[i]);
      Value loss = Value(1).Add(expected.Negate().Multiply(scores[i])).Relu();
      losses.push_back(loss);
    }
    Value data_loss = Value(0.0);
    for (const auto& loss : losses) {
      data_loss = loss.Add(data_loss);
    }
    data_loss = data_loss.Multiply(Value(1).Divide(Value(losses.size())));
    constexpr float alpha = 1e-4;
    Value reg_loss = Value(0.0);
    for (const auto& p : model.Parameters()) {
      reg_loss = reg_loss.Add(p.Multiply(p));
    }
    reg_loss = Value(alpha).Multiply(reg_loss);
    auto total_loss = data_loss.Add(reg_loss);

    // Accuracy
    float accuracy = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
      float score = scores[i].value();
      float expected = training_data.classifications[i];
      accuracy += (score > 0) == (expected > 0) ? 1.0 : 0.0;
    }
    accuracy = accuracy / scores.size();

    // Backward pass
    for (auto& p : model.Parameters()) {
      p.gradient(0);
    }
    total_loss.Backward();

    // Update
    float learning_rate = 1.0 - ((0.9 * k) / steps);
    for (auto& p : model.Parameters()) {
      p.value(p.value() - (learning_rate * p.gradient()));
    }
    std::cout << "step " << k << " loss " << total_loss.value() << " accuracy "
              << accuracy * 100 << "%\n";
  }
  Draw(training_data, model);
}

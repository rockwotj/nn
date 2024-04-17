#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/str_format.h"

constexpr char kSpecial = '`';

static_assert('a' - kSpecial == 1);

torch::Tensor MakeBigrams(std::filesystem::path path) {
  std::ifstream stream(path);
  std::string name;
  auto counts = torch::zeros({27, 27}, torch::dtype<int32_t>());
  while (std::getline(stream, name)) {
    for (ssize_t i = -1; i < ssize_t(name.size()); ++i) {
      char a = i >= 0 ? name[i] : kSpecial;
      int j = i + 1;
      char b = j < name.size() ? name[j] : kSpecial;
      counts[a - kSpecial][b - kSpecial] += 1;
    }
  }
  return counts;
}

void Sample(torch::Tensor n) {
  auto p = n[0].toType(torch::kFloat32);
  p = p / p.sum();
  absl::PrintF("%v\n", absl::FormatStreamed(p));
}

ABSL_FLAG(std::string, names_file, "makemore/names.txt", "input names file");

int main() {
  auto counts = MakeBigrams(absl::GetFlag(FLAGS_names_file));
  absl::PrintF("%v\n", counts['r' - kSpecial]['i' - kSpecial].item().toFloat());
  Sample(std::move(counts));
}

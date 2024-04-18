#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/str_format.h"

constexpr char kSpecial = '`';

static_assert('a' - kSpecial == 1);

std::vector<std::string> ReadNames(std::filesystem::path path) {
  std::ifstream stream(path);
  std::string name;
  std::vector<std::string> names;
  while (std::getline(stream, name)) {
    names.push_back(name);
  }
  return names;
}

torch::Tensor MakeBigrams(const std::vector<std::string>& words) {
  auto counts = torch::zeros({27, 27}, torch::dtype<int32_t>());
  for (const auto& word : words) {
    for (ssize_t i = -1; i < ssize_t(word.size()); ++i) {
      char a = i >= 0 ? word[i] : kSpecial;
      int j = i + 1;
      char b = j < word.size() ? word[j] : kSpecial;
      counts[a - kSpecial][b - kSpecial] += 1;
    }
  }
  return counts;
}

void Sample(const std::vector<std::string>& words, torch::Tensor N) {
  auto P = (N + 1).toType(torch::kF32);
  P /= P.sum(/*dim=*/1, /*keepdim=*/true);
  auto g = torch::make_generator<torch::CPUGeneratorImpl>();
  g.set_current_seed(2147483647);
  for (int i = 0; i < 50; ++i) {
    std::string out;
    int ix = 0;
    for (;;) {
      auto p = P[ix];
      ix = torch::multinomial(p,
                              /*num_samples=*/1,
                              /*replacement=*/true,
                              /*generator=*/g)
               .item()
               .toInt();
      if (ix == 0) {
        break;
      }
      out.push_back(kSpecial + ix);
    }
    absl::PrintF("%v\n", out);
  }

  auto log_likelihood = torch::zeros({1});
  float n = 0;
  for (const auto& word : words) {
    for (ssize_t i = -1; i < ssize_t(word.size()); ++i) {
      char a = i >= 0 ? word[i] : kSpecial;
      int j = i + 1;
      char b = j < word.size() ? word[j] : kSpecial;
      auto prob = P[a - kSpecial][b - kSpecial];
      auto logprob = torch::log(prob);
      log_likelihood += logprob;
      n += 1;
    }
  }
  absl::PrintF("%v\n", absl::FormatStreamed(log_likelihood));
  absl::PrintF("%v\n", absl::FormatStreamed(-log_likelihood));
  absl::PrintF("%v\n", absl::FormatStreamed((-log_likelihood) / n));
}

void TrainNN(std::span<std::string> words) {
  std::vector<int32_t> xs_vec;
  std::vector<int32_t> ys_vec;
  for (const auto& word : words) {
    for (ssize_t i = -1; i < ssize_t(word.size()); ++i) {
      char a = i >= 0 ? word[i] : kSpecial;
      int j = i + 1;
      char b = j < word.size() ? word[j] : kSpecial;
      xs_vec.push_back(a - kSpecial);
      ys_vec.push_back(b - kSpecial);
    }
  }
  auto g = torch::make_generator<torch::CPUGeneratorImpl>();
  g.set_current_seed(2147483647);

  auto xs = torch::tensor(xs_vec);
  auto ys = torch::tensor(ys_vec);
  auto num = xs.numel();
  absl::PrintF("number of examples: %d\n", num);
  auto W = torch::randn({27, 27}, g, torch::requires_grad());
  for (int k = 0; k < 50; ++k) {
    auto xenc = torch::one_hot(xs, /*num_classes=*/27).toType(torch::kF32);
    auto logits = xenc.matmul(W);
    auto counts = logits.exp();
    torch::Tensor probs = counts / counts.sum(1, /*keepdim=*/true);
    auto loss = -torch::index_select(
                     torch::index_select(probs, 0, torch::arange(num)), 1, ys)
                     .log()
                     .mean();
    absl::PrintF("loss: %v\n", absl::FormatStreamed(loss));

    W.mutable_grad().reset();
    loss.backward();
    W.data() += -5.0 * W.grad();
  }
}

ABSL_FLAG(std::string, names_file, "makemore/names.txt", "input names file");

int main() {
  auto names = ReadNames(absl::GetFlag(FLAGS_names_file));
  // auto counts = MakeBigrams(names);
  // Sample(names, std::move(counts));
  TrainNN(names);
}

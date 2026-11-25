// Text from a pretrained model in the Hugging Face format (Llama, Mistral, Qwen2 or GPT-2), or from a small one
// trained here when no folder is given.
//
//   ./generate path/to/Llama-3.2-1B "The capital of France is" --temperature 0.7 --top-p 0.9 --tokens 50
//   ./generate
//
// The folder needs config.json, tokenizer.json and model.safetensors (or its shards).

#include "dummy_core.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

static Device best_device() {
#ifdef USE_CUDA
    int gpus = 0;
    if (cudaGetDeviceCount(&gpus) == cudaSuccess && gpus > 0) return Device::CUDA;
#endif
    return Device::CPU;
}

template <typename Model>
static int run(Model& model, const Tokenizer& tok, const std::string& prompt, GenerateOptions o) {
    model.to(best_device());
    auto ids = tok.encode_tensor(prompt);
    // Words as they come: decoding everything so far keeps characters that take several tokens whole
    std::vector<int64_t> made;
    std::string shown;
    o.on_tokens = [&](const std::vector<int64_t>& t) {
        made.push_back(t[0]);
        std::string text = tok.decode(made);
        if (!text.empty() && (unsigned char)text.back() < 0x80) {
            std::cout << text.substr(shown.size()) << std::flush;
            shown = text;
        }
        return true;
    };
    std::cout << prompt;
    auto start = std::chrono::steady_clock::now();
    auto out = generate(model, ids, o);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << tok.decode(made).substr(shown.size()) << "\n\n"
              << made.size() << " tokens, " << (double)made.size() / seconds << " per second, " << model.num_parameters() / 1000000 << "M parameters\n";
    return made.empty() ? 1 : 0;
}

// A tiny model of a tiny text, to see the pieces work together
static int demo() {
    std::string text;
    for (int i = 0; i < 40; ++i) text += "One fish, two fish, red fish, blue fish. This one has a little star, this one has a little car. ";
    auto tok = Tokenizer::train(text, 320, {"<|end|>"});
    auto ids = tok.encode(text);
    TokenDataset data(std::vector<int32_t>(ids.begin(), ids.end()));
    manual_seed(0);
    ModernTransformer model((int)tok.vocab_size(), 64, 64, 4, 2, 2);
    Device device = best_device();
    model.to(device);
    auto params = model.parameters();
    Adam opt(3e-3f);
    TokenLoader loader(data, 8, 32, /*seed=*/0, device);
    float loss_value = 0.0f;
    for (int step = 0; step < 60; ++step) {
        Batch b = loader.next();
        opt.zero_grad(params);
        auto loss = fused_cross_entropy(model.forward(b.inputs), b.targets);
        loss->backward();
        opt.step(params);
        loss_value = loss->item();
    }
    std::cout << "trained a demo model of " << model.num_parameters() << " parameters, loss " << loss_value << "\n";
    GenerateOptions o;
    o.max_new_tokens = 24;
    o.temperature = 0.0f;
    return run(model, tok, "One fish,", o);
}

int main(int argc, char** argv) {
    std::string dir, prompt = "Once upon a time";
    GenerateOptions o;
    o.max_new_tokens = 100;
    o.temperature = 0.8f;
    o.top_p = 0.95f;
    int plain = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        bool value = i + 1 < argc;
        if (a == "--temperature" && value) o.temperature = (float)std::atof(argv[++i]);
        else if (a == "--top-p" && value) o.top_p = (float)std::atof(argv[++i]);
        else if (a == "--top-k" && value) o.top_k = std::atoll(argv[++i]);
        else if (a == "--tokens" && value) o.max_new_tokens = std::atoll(argv[++i]);
        else if (a == "--seed" && value) manual_seed((unsigned)std::atoi(argv[++i]));
        else if (plain++ == 0) dir = a;
        else prompt = a;
    }
    try {
        if (dir.empty()) return demo();
        auto config = json::parse_file(dir + "/config.json");
        auto tok = Tokenizer::from_file(dir + "/tokenizer.json");
        if (const auto* eos = config.find("eos_token_id")) {
            if (eos->is_array()) for (const auto& e : eos->array) o.eos.push_back(e.as_int());
            else if (!eos->is_null()) o.eos.push_back(eos->as_int());
        }
        if (config.get_string("model_type", "") == "gpt2") {
            auto model = load_gpt2(dir);
            return run(model, tok, prompt, o);
        }
        auto model = load_llama(dir, /*max_seq_len=*/4096);
        return run(model, tok, prompt, o);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

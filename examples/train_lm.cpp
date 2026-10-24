// A complete training run: a character-level language model with train/validation data,
// AdamW with warmup + cosine decay, gradient clipping, metric logging, periodic evaluation,
// and resumable checkpoints. Use it as a template for your own experiments.
//
//   ./train_lm                  train on the built-in corpus
//   ./train_lm my_text.txt      train on any text file (one token per byte)
//   ./train_lm --resume         continue from train_lm.ckpt.safetensors
//
// Metrics go to train_lm_metrics.csv (step,metric,value).

#include "dummy_core.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

static const char* kCorpus =
    "A small engine for deep learning can be read in an afternoon. Tensors hold numbers, ops turn "
    "tensors into new tensors, and autograd walks the graph backwards to find out how every number "
    "should change. The model is only a stack of layers, and each layer is only a few matrix "
    "multiplications and a handful of nonlinear functions. Training is a loop: take a batch, make a "
    "prediction, measure the error, compute the gradients, and nudge the weights a little. Then do it "
    "again, many thousands of times, until the error stops falling. "
    "Most bugs in such an engine are quiet. A transposed tensor read in the wrong order still produces "
    "numbers. A gradient sent to the wrong place still lets the loss move. A mask that looks at the "
    "future still trains, and even trains well, until the model is asked to write something on its own. "
    "This is why every op deserves a gradient check, and why the checks must run on every change. "
    "A quiet bug found today is a week of confusing experiments that never has to happen. "
    "Good research tools get out of the way. They let you write a new layer in the morning, check its "
    "gradients before lunch, and watch it train by the evening. They save checkpoints so a crash costs "
    "minutes instead of days, they log every number worth plotting, and they make the slow parts easy "
    "to find. Speed matters, but trust matters first: a fast answer you cannot believe is no answer. "
    "The loop below is the whole story. Data comes in as token windows, the model predicts the next "
    "token at every position, the cross entropy says how surprised it was, and the optimizer makes it "
    "a little less surprised next time. Validation data that the model never trains on tells you "
    "whether it learned the patterns or only memorized the pages. When the two losses part ways, the "
    "model has started to memorize, and it is time to stop, regularize, or find more data. "
    "Small models on small text learn the easy things first: which letters are common, which letters "
    "follow which, where spaces go, how words begin and end. Only later, with more data and more "
    "capacity, do they learn grammar, facts, and style. Watching that progression in your own engine, "
    "written line by line, is one of the quiet joys of building things from scratch. ";

int main(int argc, char** argv) {
    bool resume = false;
    std::string text_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--resume") resume = true;
        else text_path = arg;
    }

    // ---- Configuration ----
    const int64_t steps = 200, batch_size = 16, seq_len = 32;
    const int64_t log_every = 10, eval_every = 50;
    const float peak_lr = 3e-3f, min_lr = 3e-4f;
    const std::string ckpt_path = "train_lm.ckpt.safetensors";

    Device device = Device::CPU;
#ifdef USE_CUDA
    int gpus = 0;
    if (cudaGetDeviceCount(&gpus) == cudaSuccess && gpus > 0) device = Device::CUDA;
#endif

    // ---- Data ----
    TokenDataset data = text_path.empty() ? TokenDataset::from_string(kCorpus) : TokenDataset::from_bytes(text_path);
    // Not a structured binding: lambdas below capture these, which C++17 forbids for bindings
    auto parts = data.split(0.9);
    TokenDataset train = std::move(parts.first), val = std::move(parts.second);
    std::cout << "tokens: " << train.size() << " train, " << val.size() << " validation, vocab " << data.vocab_size() << "\n";

    // ---- Model, optimizer, schedule ----
    manual_seed(0);
    ModernTransformer model(data.vocab_size(), seq_len, /*d_model=*/64, /*n_heads=*/4, /*n_layers=*/2, /*n_kv_heads=*/2);
    model.to(device);
    auto params = model.parameters();  // after to(): moving replaces the tensors
    std::cout << "parameters: " << model.num_parameters() << " on " << (device == Device::CUDA ? "CUDA" : "CPU") << "\n";

    Adam optimizer(peak_lr, 0.9f, 0.95f, 1e-8f, /*weight_decay=*/0.1f);
    CosineAnnealingLR schedule(peak_lr, (int)steps, /*warmup_steps=*/20, min_lr);

    TrainingState state;
    if (resume) {
        state = load_checkpoint(ckpt_path, model, &optimizer);
        schedule.current_step = (int)state.step;
        std::cout << "resumed from step " << state.step << "\n";
    } else {
        std::remove(ckpt_path.c_str());
    }
    MetricsLogger logger("train_lm_metrics.csv", /*echo=*/true, /*append=*/resume);

    // Mean validation loss over the whole validation stream, without building graphs
    auto evaluate = [&]() {
        NoGradGuard no_grad;
        model.eval();
        double total = 0.0;
        int64_t batches = 0;
        for (int64_t w = 0; w < val.num_windows(seq_len); w += batch_size) {
            Batch b = val.windows(w, batch_size, seq_len, device);
            total += fused_cross_entropy(model.forward(b.inputs), b.targets)->cpu()->data_ptr<float>()[0];
            ++batches;
        }
        model.train();
        return total / (double)batches;
    };

    // ---- Training loop ----
    std::mt19937 rng((unsigned)(1234 + state.step));  // a different stream after resuming
    double first_val = evaluate();
    double last_val = first_val;
    logger.log(state.step, {{"val_loss", first_val}});

    for (int64_t step = state.step; step < steps; ++step) {
        optimizer.lr = schedule.get_lr();
        Batch b = train.sample(batch_size, seq_len, rng, device);

        optimizer.zero_grad(params);
        auto loss = fused_cross_entropy(model.forward(b.inputs), b.targets);
        loss->backward();
        float grad_norm = clip_grad_norm_(params, 1.0f);
        optimizer.step(params);
        schedule.step();

        if (step % log_every == 0) {
            float train_loss = loss->cpu()->data_ptr<float>()[0];
            logger.log(step, {{"train_loss", train_loss}, {"lr", optimizer.lr}, {"grad_norm", grad_norm}});
        }
        if ((step + 1) % eval_every == 0) {
            last_val = evaluate();
            logger.log(step + 1, {{"val_loss", last_val}});
            save_checkpoint(ckpt_path, model, &optimizer, TrainingState{step + 1, {{"val_loss", last_val}}});
        }
    }

    // ---- Sample from the model ----
    {
        NoGradGuard no_grad;
        model.eval();
        std::string text = "The model";
        for (int i = 0; i < 80; ++i) {
            int64_t n = std::min<int64_t>((int64_t)text.size(), seq_len);
            std::vector<float> ids;
            for (int64_t k = (int64_t)text.size() - n; k < (int64_t)text.size(); ++k) ids.push_back((unsigned char)text[k]);
            auto x = std::make_shared<Tensor>(std::vector<int64_t>{1, n}, Device::CPU, DType::Float32, false);
            x->copy_from_vector(ids);
            auto logits = model.forward(x->to(device))->cpu();
            const float* last = logits->data_ptr<float>() + (n - 1) * data.vocab_size();
            int best = (int)(std::max_element(last, last + data.vocab_size()) - last);  // greedy decoding
            text.push_back((char)best);
        }
        std::cout << "sample: " << text << "\n";
    }

    std::cout << "validation loss: " << first_val << " -> " << last_val << "\n";
    // Non-zero exit if the model failed to learn, so CTest can use this as a smoke test
    return last_val < 0.8 * first_val ? 0 : 1;
}

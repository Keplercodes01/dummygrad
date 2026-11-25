#pragma once
// Text generation with a KV cache, for any model with blocks, max_seq_len and forward_cached (GPT,
// ModernTransformer).
//
//   GenerateOptions o;
//   o.max_new_tokens = 100;
//   o.temperature = 0.8f;
//   o.top_p = 0.95f;
//   auto ids = generate(model, prompt, o);          // [batch, time + up to 100]
//
//   auto best = beam_search(model, prompt, {});     // the most likely continuations

#include "functional.h"
#include "kv_cache.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct GenerateOptions {
    int64_t max_new_tokens = 64;
    float temperature = 1.0f;         // 0: always the most likely token
    int64_t top_k = 0;                // draw from the k most likely tokens only (0: all)
    float top_p = 1.0f;               // draw from the fewest most likely tokens whose probabilities add up to top_p
    float min_p = 0.0f;               // drop tokens less than min_p times as likely as the most likely one
    float repetition_penalty = 1.0f;  // above 1: tokens the row already contains become less likely
    std::vector<int64_t> eos;         // a row ends with any of these tokens
    int64_t pad = 0;                  // fills rows after their end
    // Called with every step's tokens, one per row (pad for rows that have ended); false stops
    std::function<bool(const std::vector<int64_t>&)> on_tokens;
};

namespace generate_detail {

// The model in evaluation mode, without graphs, for as long as this lives
template <typename Model>
struct InferenceScope {
    Model& model;
    bool was_training;
    NoGradGuard no_grad;
    explicit InferenceScope(Model& m) : model(m), was_training(m.is_training()) { model.eval(); }
    ~InferenceScope() {
        if (was_training) model.train();
    }
};

inline std::vector<int64_t> host_ids(const TensorPtr& t) {
    std::vector<int64_t> ids;
    for (float f : t->to_vector()) ids.push_back((int64_t)f);
    return ids;
}

// [batch, time] from [time] or [batch, time]
inline TensorPtr as_batch(const TensorPtr& prompt, const char* who) {
    if (!prompt || prompt->ndim() < 1 || prompt->ndim() > 2 || prompt->size() == 0) {
        throw std::invalid_argument(std::string(who) + ": the prompt must be [time] or [batch, time] token ids, and not empty");
    }
    return prompt->ndim() == 1 ? reshape(prompt, {1, prompt->shape[0]}) : prompt;
}

template <typename Model>
int64_t room_for(const Model& model, int64_t prompt, int64_t wanted, const char* who) {
    if (prompt > model.max_seq_len) {
        throw std::invalid_argument(std::string(who) + ": the prompt has " + std::to_string(prompt) + " tokens, the model takes " + std::to_string(model.max_seq_len));
    }
    // The last token needs no forward pass of its own, so it may be one past the model's length
    return std::min<int64_t>(wanted, model.max_seq_len - prompt + 1);
}

inline TensorPtr id_column(const std::vector<int64_t>& ids, Device device) {
    std::vector<float> f(ids.begin(), ids.end());
    return op_detail::from_values(f, {(int64_t)ids.size(), 1}, device);
}

}  // namespace generate_detail

// Draws a token from one row of logits. u is uniform on [0, 1); seen lists the tokens to penalize.
inline int64_t sample_logits(const float* logits, int64_t vocab, const GenerateOptions& o, const std::vector<int64_t>& seen, double u) {
    if (o.temperature < 0.0f || o.repetition_penalty <= 0.0f) throw std::invalid_argument("generate: temperature must be >= 0 and repetition_penalty > 0");
    std::vector<float> l(logits, logits + vocab);
    if (o.repetition_penalty != 1.0f) {
        std::vector<char> done((size_t)vocab, 0);
        for (int64_t t : seen) {
            if (t < 0 || t >= vocab || done[(size_t)t]) continue;
            done[(size_t)t] = 1;
            l[(size_t)t] = l[(size_t)t] > 0.0f ? l[(size_t)t] / o.repetition_penalty : l[(size_t)t] * o.repetition_penalty;
        }
    }
    if (o.temperature == 0.0f || o.top_k == 1) return (int64_t)(std::max_element(l.begin(), l.end()) - l.begin());

    // Candidates: (probability, token), the most likely ones when top_k asks for them
    std::vector<std::pair<double, int64_t>> c((size_t)vocab);
    for (int64_t t = 0; t < vocab; ++t) c[(size_t)t] = {(double)l[(size_t)t], t};
    auto better = [](const std::pair<double, int64_t>& a, const std::pair<double, int64_t>& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; };
    if (o.top_k > 0 && o.top_k < vocab) {
        std::nth_element(c.begin(), c.begin() + o.top_k, c.end(), better);
        c.resize((size_t)o.top_k);
    }
    double top = -std::numeric_limits<double>::infinity();
    for (const auto& p : c) top = std::max(top, p.first);
    if (!std::isfinite(top)) throw std::runtime_error("generate: the logits are not finite");
    double total = 0.0;
    for (auto& p : c) total += p.first = std::exp((p.first - top) / (double)o.temperature);
    if (o.min_p > 0.0f) {
        c.erase(std::remove_if(c.begin(), c.end(), [&](const auto& p) { return p.first < (double)o.min_p; }), c.end());
        total = 0.0;
        for (const auto& p : c) total += p.first;
    }
    if (o.top_p < 1.0f) {
        // Most likely first, until enough probability is covered
        auto worse = [&](const auto& a, const auto& b) { return better(b, a); };
        std::make_heap(c.begin(), c.end(), worse);
        auto rest = c.end();
        double kept = 0.0;
        do {
            std::pop_heap(c.begin(), rest, worse);
            --rest;
            kept += rest->first;
        } while (rest != c.begin() && kept < (double)o.top_p * total);
        c.erase(c.begin(), rest);
        total = 0.0;
        for (const auto& p : c) total += p.first;
    }
    // Candidates in token order, so the draw does not depend on how they were found
    std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    double target = u * total, sum = 0.0;
    for (const auto& p : c) {
        sum += p.first;
        if (target < sum) return p.second;
    }
    return c.back().second;
}

// prompt ([time] or [batch, time] ids) followed by tokens drawn from the model. Every row has the prompt's
// length; generation stops at max_new_tokens, at the model's max_seq_len, or when every row has ended.
// Draws follow manual_seed.
template <typename Model>
std::shared_ptr<Tensor> generate(Model& model, const std::shared_ptr<Tensor>& prompt_in, const GenerateOptions& o = {}) {
    auto prompt = generate_detail::as_batch(prompt_in, "generate");
    generate_detail::InferenceScope<Model> scope(model);
    int64_t B = prompt->shape[0], T = prompt->shape[1];
    int64_t steps = generate_detail::room_for(model, T, o.max_new_tokens, "generate");
    auto flat = generate_detail::host_ids(prompt);
    std::vector<std::vector<int64_t>> rows((size_t)B);
    for (int64_t b = 0; b < B; ++b) rows[(size_t)b].assign(flat.begin() + b * T, flat.begin() + (b + 1) * T);
    std::vector<char> ended((size_t)B, 0);

    std::vector<KVCache> caches(model.blocks.size());
    for (auto& c : caches) c.reserve = T + steps;
    Device device = model.token_emb.weight->device;
    auto next = prompt;
    for (int64_t step = 0; step < steps; ++step) {
        auto logits = model.forward_cached(next, caches)->to_vector();  // [B, 1, vocab]
        int64_t vocab = (int64_t)logits.size() / B;
        auto u = rand({B})->to_vector();
        std::vector<int64_t> tokens((size_t)B, o.pad);
        for (int64_t b = 0; b < B; ++b) {
            if (ended[(size_t)b]) continue;
            int64_t t = sample_logits(logits.data() + b * vocab, vocab, o, rows[(size_t)b], (double)u[(size_t)b]);
            tokens[(size_t)b] = t;
            ended[(size_t)b] = std::find(o.eos.begin(), o.eos.end(), t) != o.eos.end();
        }
        for (int64_t b = 0; b < B; ++b) rows[(size_t)b].push_back(tokens[(size_t)b]);
        bool more = !o.on_tokens || o.on_tokens(tokens);
        if (!more || std::all_of(ended.begin(), ended.end(), [](char e) { return e != 0; })) break;
        if (step + 1 < steps) next = generate_detail::id_column(tokens, device);
    }
    std::vector<float> out;
    for (const auto& r : rows) out.insert(out.end(), r.begin(), r.end());
    return op_detail::from_values(out, {B, (int64_t)rows[0].size()}, prompt->device);
}

struct BeamSearchOptions {
    int64_t max_new_tokens = 64;
    int num_beams = 4;
    float length_penalty = 1.0f;  // a sequence's score is its log probability / (new tokens ^ length_penalty)
    std::vector<int64_t> eos;
};
struct BeamResult {
    std::vector<int64_t> tokens;  // the prompt and what follows it, with the eos that ended it
    float score;
};

// The num_beams best continuations of one prompt ([time] ids), best first
template <typename Model>
std::vector<BeamResult> beam_search(Model& model, const std::shared_ptr<Tensor>& prompt_in, const BeamSearchOptions& o = {}) {
    auto prompt = generate_detail::as_batch(prompt_in, "beam_search");
    if (prompt->shape[0] != 1) throw std::invalid_argument("beam_search: one prompt at a time");
    if (o.num_beams < 1) throw std::invalid_argument("beam_search: num_beams must be positive");
    generate_detail::InferenceScope<Model> scope(model);
    int64_t T = prompt->shape[1];
    int64_t steps = generate_detail::room_for(model, T, o.max_new_tokens, "beam_search");
    Device device = model.token_emb.weight->device;

    struct Beam {
        std::vector<int64_t> tokens;
        double log_prob;
    };
    auto score_of = [&](double log_prob, int64_t new_tokens) { return (float)(log_prob / std::pow((double)std::max<int64_t>(1, new_tokens), (double)o.length_penalty)); };
    std::vector<Beam> beams{{generate_detail::host_ids(prompt), 0.0}};
    std::vector<BeamResult> finished;
    auto by_score = [](const BeamResult& a, const BeamResult& b) { return a.score > b.score; };

    std::vector<KVCache> caches(model.blocks.size());
    for (auto& c : caches) c.reserve = T + steps;
    auto next = prompt;
    for (int64_t step = 0; step < steps && !beams.empty(); ++step) {
        auto logits = model.forward_cached(next, caches)->to_vector();
        int64_t vocab = (int64_t)(logits.size() / beams.size());
        // Enough candidates that num_beams survive even if every eos token is among them
        struct Candidate {
            double log_prob;
            int64_t beam, token;
        };
        auto better = [](const Candidate& a, const Candidate& b) {
            if (a.log_prob != b.log_prob) return a.log_prob > b.log_prob;
            return a.beam != b.beam ? a.beam < b.beam : a.token < b.token;
        };
        std::vector<Candidate> all;
        size_t keep = (size_t)o.num_beams * (1 + o.eos.size());
        for (size_t b = 0; b < beams.size(); ++b) {
            const float* row = logits.data() + (int64_t)b * vocab;
            double top = (double)*std::max_element(row, row + vocab), sum = 0.0;
            for (int64_t t = 0; t < vocab; ++t) sum += std::exp((double)row[t] - top);
            double log_total = top + std::log(sum);
            std::vector<Candidate> mine((size_t)vocab);
            for (int64_t t = 0; t < vocab; ++t) mine[(size_t)t] = {beams[b].log_prob + (double)row[t] - log_total, (int64_t)b, t};
            size_t take = std::min(keep, mine.size());
            std::partial_sort(mine.begin(), mine.begin() + (std::ptrdiff_t)take, mine.end(), better);
            all.insert(all.end(), mine.begin(), mine.begin() + (std::ptrdiff_t)take);
        }
        std::sort(all.begin(), all.end(), better);
        std::vector<Beam> grown;
        std::vector<int64_t> from, tokens;
        for (const auto& c : all) {
            if ((int)grown.size() == o.num_beams) break;
            auto seq = beams[(size_t)c.beam].tokens;
            seq.push_back(c.token);
            if (std::find(o.eos.begin(), o.eos.end(), c.token) != o.eos.end()) {
                finished.push_back({std::move(seq), score_of(c.log_prob, step + 1)});
                continue;
            }
            grown.push_back({std::move(seq), c.log_prob});
            from.push_back(c.beam);
            tokens.push_back(c.token);
        }
        beams = std::move(grown);
        std::sort(finished.begin(), finished.end(), by_score);
        if ((int)finished.size() > o.num_beams) finished.resize((size_t)o.num_beams);
        if ((int)finished.size() == o.num_beams && !beams.empty() && o.length_penalty >= 0.0f) {
            // Longer sequences only lose probability: with the longest length's divisor, can any beat the worst kept?
            if (score_of(beams[0].log_prob, steps) <= finished.back().score) break;
        }
        if (beams.empty() || step + 1 == steps) break;
        for (auto& c : caches) c.select_rows(from);
        next = generate_detail::id_column(tokens, device);
    }
    for (const auto& b : beams) finished.push_back({b.tokens, score_of(b.log_prob, (int64_t)b.tokens.size() - T)});
    std::sort(finished.begin(), finished.end(), by_score);
    if ((int)finished.size() > o.num_beams) finished.resize((size_t)o.num_beams);
    return finished;
}

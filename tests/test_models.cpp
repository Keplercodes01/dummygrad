// Tests of operators and indexing, KV caches and generation, pretrained checkpoints, adapters, experts, data
// loading and memory. Models are checked against plain-loop implementations written here.
//
// Exit code is 0 only if every test passes. An argument runs only tests whose name contains it.

#include "test_util.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <set>

using Ids = std::vector<int64_t>;

// A folder that disappears with the test
struct TempDir {
    std::string path;
    explicit TempDir(const std::string& name) {
        path = (std::filesystem::temp_directory_path() / ("dummygrad_" + name)).string();
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

static TensorPtr id_tensor(const Ids& ids, const std::vector<int64_t>& shape) { return tensor(std::vector<float>(ids.begin(), ids.end()), shape); }

static std::vector<Device> devices() {
    std::vector<Device> d{Device::CPU};
    if (has_cuda()) d.push_back(Device::CUDA);
    return d;
}

// ---------------------------------------------------------------------------------------------
// Operators, indexing, printing
// ---------------------------------------------------------------------------------------------

static std::string operators() {
    auto a = tensor({1, 2, 3}, {3}), b = tensor({4, 5, 6}, {3});
    CHECK(near(a + b, {5, 7, 9}));
    CHECK(near(a - b, {-3, -3, -3}));
    CHECK(near(a * b, {4, 10, 18}));
    CHECK(near(b / a, {4, 2.5f, 2}));
    CHECK(near(2.0f * a + 1, {3, 5, 7}));
    CHECK(near(1.0f - a, {0, -1, -2}));
    CHECK(near(6.0f / a, {6, 3, 2}));
    CHECK(near(a / 2.0f - 1.0f, {-0.5f, 0, 0.5f}));
    CHECK(near(-a, {-1, -2, -3}));
    CHECK(near(reshape(a, {3, 1}) * tensor({10, 1}, {2}), {10, 1, 20, 2, 30, 3}));
    // Assignment forms rebind: the other name keeps its tensor
    auto c = a;
    c += b;
    c *= 2.0f;
    c -= a;
    c /= 3.0f;
    c += 1.0f;
    CHECK(near(c, {4, 5, 6}));
    CHECK(near(a, {1, 2, 3}));
    auto x = tensor({2, 3}, {2}, Device::CPU, true);
    sum((x * x - 4.0f) / (x + 1.0f))->backward();
    CHECK(near(x->grad, {12.0f / 9.0f, 19.0f / 16.0f}));
    // Comparisons still compare pointers
    TensorPtr null;
    if (a == b || a != a || null != nullptr || !a) return "pointer comparisons changed";
    return "";
}

// Python's slice.indices, written out
static std::vector<int> python_slice(int n, std::optional<int64_t> start, std::optional<int64_t> stop, int64_t step) {
    std::vector<int> out;
    auto fix = [n, step](int64_t p) {
        if (p < 0) {
            p += n;
            if (p < 0) p = step < 0 ? -1 : 0;
        } else if (p >= n) {
            p = step < 0 ? n - 1 : n;
        }
        return p;
    };
    int64_t i = start ? fix(*start) : (step < 0 ? n - 1 : 0), end = stop ? fix(*stop) : (step < 0 ? -1 : n);
    for (; step > 0 ? i < end : i > end; i += step) out.push_back((int)i);
    return out;
}

static std::string indexing() {
    auto x = reshape(arange(0, 24), {2, 3, 4});
    CHECK(shape_is(x->index({}), {2, 3, 4}));
    CHECK(shape_is(x->index({1}), {3, 4}));
    CHECK(scalar(x->index({1, -1, 2}), 22));
    CHECK(near(x->index({Ellipsis, -1}), {3, 7, 11, 15, 19, 23}));
    CHECK(near(x->index({Slice(), Slice(1, 3), Slice(std::nullopt, std::nullopt, 2)}), {4, 6, 8, 10, 16, 18, 20, 22}));
    CHECK(shape_is(x->index({Slice(), NewAxis, 0, Ellipsis, NewAxis}), {2, 1, 4, 1}));
    CHECK(near(x->index({0, Slice(std::nullopt, std::nullopt, -1), 0}), {8, 4, 0}));
    // Every slice of short vectors
    for (int n = 0; n <= 4; ++n) {
        auto v = n > 0 ? arange(0, n) : op_detail::empty({0}, Device::CPU);
        for (int64_t start = -6; start <= 6; ++start) {
            for (int64_t stop = -6; stop <= 6; ++stop) {
                for (int64_t step : {-3, -2, -1, 1, 2, 3}) {
                    std::optional<int64_t> s0 = start == 6 ? std::nullopt : std::optional<int64_t>(start);
                    std::optional<int64_t> s1 = stop == 6 ? std::nullopt : std::optional<int64_t>(stop);
                    auto want = python_slice(n, s0, s1, step);
                    auto got = v->index({Slice(s0, s1, step)});
                    if (got->to_vector() != std::vector<float>(want.begin(), want.end())) {
                        return "slice " + std::to_string(start) + ":" + std::to_string(stop) + ":" + std::to_string(step) + " of " + std::to_string(n);
                    }
                }
            }
        }
    }
    // Integers and slices are views
    auto view = x->index({1, Slice(0, 2)});
    if (view->storage != x->storage) return "a slice copied";

    // Index tensors
    CHECK(near(x->index({tensor({1, 0}, {2}), tensor({2, -1}, {2})}), {20, 21, 22, 23, 8, 9, 10, 11}));
    CHECK(near(x->index({tensor({0, 1}, {2}), Slice(), tensor({3, 0}, {2})}), {3, 7, 11, 12, 16, 20}));  // dims first
    CHECK(near(x->index({0, Slice(), tensor({1, 2}, {2})}), {1, 2, 5, 6, 9, 10}));
    auto y = reshape(arange(0, 120), {2, 3, 4, 5});
    auto t1 = tensor({2, 0}, {2, 1}), t2 = tensor({3, 1, 0}, {3});
    auto z = y->index({Slice(), t1, t2, Slice(1, 3)});
    CHECK(shape_is(z, {2, 2, 3, 2}));
    auto zv = z->to_vector();
    size_t k = 0;
    for (int a = 0; a < 2; ++a)
        for (int i : {2, 0})
            for (int j : {3, 1, 0})
                for (int b = 1; b < 3; ++b)
                    if (zv[k++] != (float)(((a * 3 + i) * 4 + j) * 5 + b)) return "index tensors in the middle: element " + std::to_string(k - 1);
    // Past 2^24 combined positions, where one float index cannot count them
    auto wide = op_detail::filled({5000, 4000}, Device::CPU, 0.0f);
    wide->data_ptr<float>()[4999 * 4000 + 3998] = 7.0f;
    wide->data_ptr<float>()[4999 * 4000 + 3999] = 8.0f;
    CHECK(near(wide->index({tensor({4999, 4999, 0}, {3}), tensor({3999, 3998, 1}, {3})}), {8, 7, 0}));

    // Masks
    CHECK(near(x->index({Mask(gt(x, 20.5f))}), {21, 22, 23}));
    CHECK(near(x->index({Mask(tensor({0, 1, 0, 0, 0, 1}, {2, 3}))}), {4, 5, 6, 7, 20, 21, 22, 23}));
    CHECK(shape_is(x->index({Slice(), Mask(tensor({1, 0, 1}, {3}))}), {2, 2, 4}));
    CHECK(shape_is(x->index({Mask(zeros({2}))}), {0, 3, 4}));

    CHECK(expect_throw([&] { x->index({0, 0, 0, 0}); }, "too many indices"));
    CHECK(expect_throw([&] { x->index({2}); }, "an integer out of range"));
    CHECK(expect_throw([&] { x->index({tensor({2}, {1})}); }, "an index out of range"));
    CHECK(expect_throw([&] { x->index({tensor({-3}, {1})}); }, "a negative index out of range"));
    CHECK(expect_throw([&] { x->index({Ellipsis, Ellipsis}); }, "two ellipses"));
    CHECK(expect_throw([&] { x->index({Mask(zeros({3}))}); }, "a mask of another size"));
    CHECK(expect_throw([&] { x->index({Slice(0, 1, 0)}); }, "a step of 0"));
    return "";
}

static std::string index_assignment() {
    auto x = reshape(arange(0, 6), {2, 3});
    CHECK(near(index_put(x, {0}, 9.0f), {9, 9, 9, 3, 4, 5}));
    CHECK(near(x->index_put({Slice(), Slice(1, 3)}, tensor({-1, -2}, {2})), {0, -1, -2, 3, -1, -2}));
    CHECK(near(index_put(x, {Ellipsis, Slice(std::nullopt, std::nullopt, -2)}, tensor({8, 7}, {2})), {7, 1, 8, 7, 4, 8}));
    CHECK(near(x->index_put({tensor({1, 0}, {2}), tensor({2, 0}, {2})}, tensor({50, 60}, {2})), {60, 1, 2, 3, 4, 50}));
    CHECK(near(index_put(x, {Mask(gt(x, 3.5f))}, 0.0f), {0, 1, 2, 3, 0, 0}));
    CHECK(near(x->index_put({1, Slice(0, 2)}, tensor({1}, {1}), true), {0, 1, 2, 4, 5, 5}));
    CHECK(near(tensor({0, 0, 0}, {3})->index_put({tensor({2, 2, 0}, {3})}, tensor({1, 2, 4}, {3}), true), {4, 0, 3}));
    CHECK(near(x, {0, 1, 2, 3, 4, 5}));  // the input is left alone
    CHECK(expect_throw([&] { x->index_put({0}, zeros({4})); }, "a value of another shape"));
    return "";
}

static std::string printing() {
    std::ostringstream os;
    os << tensor({1, 2, 3, 4}, {2, 2}) << " " << scalar_tensor(0.5f) << " " << TensorPtr();
    if (os.str() != "[[1, 2],\n [3, 4]], shape=[2, 2] 0.5, shape=[] tensor(null)") return "printed " + os.str();
    auto big = tensor_string(arange(0, 2000));
    if (big.rfind("[0, 1, 2, ..., 1997, 1998, 1999]", 0) != 0) return "printed " + big;
    return "";
}

// ---------------------------------------------------------------------------------------------
// KV cache and generation
// ---------------------------------------------------------------------------------------------

template <typename Model>
static std::string cache_matches(Model& model, int vocab) {
    NoGradGuard no_grad;
    model.eval();
    for (Device d : devices()) {
        model.to(d);
        manual_seed(3);
        auto ids = randint(0, vocab, {2, 7})->to(d);
        auto full = model.forward(ids);
        std::vector<KVCache> caches;
        TensorList parts;
        for (auto [start, n] : {std::pair<int, int>{0, 3}, {3, 1}, {4, 2}, {6, 1}}) {
            parts.push_back(model.forward_cached(contiguous(narrow(ids, 1, start, n)), caches, /*last_only=*/false));
        }
        CHECK(near(cat(parts, 1), full, 1e-4f, 1e-4f));
        // Dropping tokens and feeding them again
        for (auto& c : caches) c.truncate(4);
        CHECK(near(model.forward_cached(contiguous(narrow(ids, 1, 4, 3)), caches), contiguous(narrow(full, 1, 6, 1)), 1e-4f, 1e-4f));
        // Rows can be rearranged
        for (auto& c : caches) {
            c.truncate(6);
            c.select_rows({1, 0, 1});
        }
        auto order = tensor({1, 0, 1}, {3}, d);
        auto last = model.forward_cached(index_select(narrow(ids, 1, 6, 1), 0, order), caches);
        CHECK(near(last, index_select(narrow(full, 1, 6, 1), 0, order), 1e-4f, 1e-4f));
        // A reset cache starts over, with another batch size
        for (auto& c : caches) c.reset();
        auto one = contiguous(narrow(ids, 0, 1, 1));
        CHECK(near(model.forward_cached(one, caches, false), contiguous(narrow(full, 0, 1, 1)), 1e-4f, 1e-4f));
    }
    model.to(Device::CPU);
    GradModeGuard grad(true);
    std::vector<KVCache> caches;
    return expect_throw([&] { model.forward_cached(zeros({1, 2}), caches); }, "a cache with gradients on");
}

static ModernTransformerConfig small_config() {
    ModernTransformerConfig c;
    c.vocab_size = 13;
    c.max_seq_len = 16;
    c.d_model = 16;
    c.n_heads = 4;
    c.n_kv_heads = 2;
    c.n_layers = 2;
    c.hidden_dim = 24;
    c.qkv_bias = c.out_bias = c.ffn_bias = false;
    return c;
}

static std::string kv_cache() {
    manual_seed(1);
    GPT gpt(13, 16, 16, 4, 2);
    auto err = cache_matches(gpt, 13);
    if (!err.empty()) return "GPT: " + err;
    ModernTransformer modern(13, 16, 16, 4, 2, 1);
    err = cache_matches(modern, 13);
    if (!err.empty()) return "ModernTransformer: " + err;
    auto c = small_config();
    c.head_dim = 6;  // not d_model / n_heads
    c.tie_embeddings = false;
    ModernTransformer untied(c);
    err = cache_matches(untied, 13);
    return err.empty() ? "" : "untied: " + err;
}

static std::string generation() {
    manual_seed(2);
    ModernTransformer model(small_config());
    NoGradGuard no_grad;
    for (Device d : devices()) {
        model.to(d);
        GenerateOptions o;
        o.max_new_tokens = 6;
        o.temperature = 0.0f;
        auto prompt = id_tensor({1, 5, 2, 7, 7, 3}, {2, 3})->to(d);
        auto out = generate(model, prompt, o);
        CHECK(shape_is(out, {2, 9}));
        if (out->device != d) return "generate moved the ids to another device";
        // Greedy decoding without a cache
        auto naive = prompt;
        for (int i = 0; i < 6; ++i) {
            auto logits = model.forward(naive);
            naive = cat({naive, reshape(argmax(contiguous(narrow(logits, 1, naive->shape[1] - 1, 1)), -1), {2, 1})}, 1);
        }
        CHECK(near(out, naive));
        o.temperature = 1.0f;
        o.top_k = 1;
        CHECK(near(generate(model, prompt, o), naive));
        // Draws follow the seed
        o.top_k = 0;
        manual_seed(5);
        auto first = generate(model, prompt, o);
        auto second = generate(model, prompt, o);
        manual_seed(5);
        CHECK(near(generate(model, prompt, o), first));
        if (near(second, first).empty()) return "two generations drew the same tokens";
        // A row ends at its eos token and is padded; the other row goes on
        auto greedy = naive->to_vector();
        o.temperature = 0.0f;
        o.eos = {(int64_t)greedy[4]};
        o.pad = 12;
        auto ended = generate(model, prompt, o)->to_vector();
        int64_t width = (int64_t)ended.size() / 2;
        bool after = false;
        for (int64_t t = 3; t < width; ++t) {
            if (after && ended[(size_t)t] != 12.0f) return "tokens after the end of a row";
            if (!after && ended[(size_t)t] != greedy[(size_t)t]) return "a row changed before its end";
            after = after || ended[(size_t)t] == greedy[4];
        }
        // Stopping from the callback, and one unbatched prompt
        o.eos.clear();
        int calls = 0;
        o.on_tokens = [&](const Ids& t) { return t.size() == 1 && ++calls < 3; };
        CHECK(shape_is(generate(model, id_tensor({4, 4}, {2}), o), {1, 5}));
    }
    model.to(Device::CPU);
    // Up to the model's length, and one token more, which needs no position of its own
    GenerateOptions o;
    o.max_new_tokens = 100;
    CHECK(shape_is(generate(model, id_tensor({1, 2}, {2}), o), {1, 17}));
    CHECK(expect_throw([&] { generate(model, zeros({1, 17}), o); }, "a prompt longer than the model"));
    if (!model.is_training()) return "generate left the model in eval mode";
    return "";
}

static std::string sampling() {
    const std::vector<float> p = {0.15f, 0.5f, 0.05f, 0.3f};
    std::vector<float> logits;
    for (float v : p) logits.push_back(std::log(v) + 2.0f);
    // The share of evenly spaced draws that picked each token
    auto shares = [&](const GenerateOptions& o, const std::vector<float>& l, const Ids& seen = {}) {
        std::vector<float> s(l.size(), 0.0f);
        for (int i = 0; i < 2000; ++i) s[(size_t)sample_logits(l.data(), (int64_t)l.size(), o, seen, (i + 0.5) / 2000)] += 1.0f / 2000;
        return s;
    };
    GenerateOptions o;
    CHECK(near(shares(o, logits), p, 1e-3f));
    o.temperature = 0.5f;
    CHECK(near(shares(o, logits), {0.0225f / 0.365f, 0.25f / 0.365f, 0.0025f / 0.365f, 0.09f / 0.365f}, 1e-3f));
    o.temperature = 1.0f;
    o.top_k = 2;
    CHECK(near(shares(o, logits), {0, 0.625f, 0, 0.375f}, 1e-3f));
    o.top_k = 0;
    o.top_p = 0.6f;
    CHECK(near(shares(o, logits), {0, 0.625f, 0, 0.375f}, 1e-3f));
    o.top_p = 0.9f;
    CHECK(near(shares(o, logits), {0.15f / 0.95f, 0.5f / 0.95f, 0, 0.3f / 0.95f}, 1e-3f));
    o.top_p = 0.0f;
    CHECK(near(shares(o, logits), {0, 1, 0, 0}, 1e-3f));
    o.top_p = 1.0f;
    o.min_p = 0.25f;
    CHECK(near(shares(o, logits), {0.15f / 0.95f, 0.5f / 0.95f, 0, 0.3f / 0.95f}, 1e-3f));
    o.min_p = 0.0f;
    // Penalized tokens: positive logits shrink, negative ones grow
    o.repetition_penalty = 2.0f;
    auto q = shares(o, {1.0f, -1.0f, 0.0f}, {0, 1, 1});
    double z = std::exp(0.5) + std::exp(-2.0) + 1.0;
    CHECK(near(q, {(float)(std::exp(0.5) / z), (float)(std::exp(-2.0) / z), (float)(1.0 / z)}, 1e-3f));
    o.repetition_penalty = 1.0f;
    const float inf[2] = {INFINITY, 0.0f};
    return expect_throw([&] { sample_logits(inf, 2, o, {}, 0.5); }, "infinite logits");
}

// Against every possible continuation
static std::string beams() {
    manual_seed(4);
    GPT model(4, 8, 8, 2, 1);
    NoGradGuard no_grad;
    const Ids prompt = {1, 2};
    auto log_probs = [&](const Ids& seq) {
        auto lp = log_softmax(model.forward(id_tensor(seq, {1, (int64_t)seq.size()})), -1)->to_vector();
        return std::vector<float>(lp.end() - 4, lp.end());
    };
    for (bool with_eos : {false, true}) {
        std::vector<std::pair<double, Ids>> all;
        std::function<void(const Ids&, double)> grow = [&](const Ids& seq, double lp) {
            size_t n = seq.size() - prompt.size();
            bool ended = with_eos && n > 0 && seq.back() == 3;
            if (ended || n == 3) {
                all.push_back({lp / std::pow((double)n, 0.7), seq});
                return;
            }
            auto next = log_probs(seq);
            for (int64_t t = 0; t < 4; ++t) {
                auto longer = seq;
                longer.push_back(t);
                grow(longer, lp + next[(size_t)t]);
            }
        };
        grow(prompt, 0.0);
        std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        BeamSearchOptions o;
        o.max_new_tokens = 3;
        o.num_beams = 64;
        o.length_penalty = 0.7f;
        if (with_eos) o.eos = {3};
        auto got = beam_search(model, id_tensor(prompt, {2}), o);
        if (got.size() < 3) return "too few beams";
        for (size_t i = 0; i < 3; ++i) {
            if (got[i].tokens != all[i].second) return "beam " + std::to_string(i) + (with_eos ? " (eos)" : "") + " is another sequence";
            if (std::fabs(got[i].score - all[i].first) > 1e-4) return "beam score";
        }
    }
    // One beam is greedy decoding
    BeamSearchOptions one;
    one.max_new_tokens = 5;
    one.num_beams = 1;
    GenerateOptions greedy;
    greedy.max_new_tokens = 5;
    greedy.temperature = 0.0f;
    auto a = beam_search(model, id_tensor(prompt, {2}), one)[0].tokens;
    return near(std::vector<float>(a.begin(), a.end()), generate(model, id_tensor(prompt, {2}), greedy)->to_vector(), 0.0f);
}

// ---------------------------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------------------------

static std::string json_values() {
    auto v = json::parse(" {\"a\": [1, -2.5e2, true, null], \"b\": {\"c\": \"x\\n\\u00e9\\ud83d\\ude00\\\"\"}, \"d\": {}, \"e\": []} ");
    if (v.at("a").array.size() != 4 || v.at("a").array[1].as_number() != -250.0 || !v.at("a").array[2].as_bool() || !v.at("a").array[3].is_null()) return "array";
    if (v.at("b").at("c").as_string() != "x\n\xc3\xa9\xf0\x9f\x98\x80\"") return "string escapes";
    if (v.get("missing", 3.0) != 3.0 || v.get_string("missing", "s") != "s" || v.at("a").array[0].as_int() != 1) return "defaults";
    if (json::parse(json::quote("a\"\\\n\x01")).as_string() != "a\"\\\n\x01") return "quoting";
    for (const char* bad : {"", "{", "[1,]", "{\"a\" 1}", "tru", "\"abc", "1 2", "{\"a\":1,}", "01x", "\"\\q\"", "[1", "-"}) {
        CHECK(expect_throw([&] { json::parse(bad); }, std::string("json ") + bad));
    }
    return expect_throw([] { json::parse(std::string(100000, '[')); }, "deep nesting");
}

using Weights = std::map<std::string, std::pair<std::vector<int64_t>, std::vector<float>>>;

static void add_weight(Weights& w, const std::string& name, const std::vector<int64_t>& shape, float scale = 0.2f, float shift = 0.0f) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    std::normal_distribution<float> dist(shift, scale);
    std::vector<float> v((size_t)n);
    for (auto& x : v) x = dist(g_rng);
    w[name] = {shape, v};
}
static void save_weights(const Weights& w, const std::string& path, const std::function<bool(const std::string&)>& wanted = nullptr) {
    std::unordered_map<std::string, TensorPtr> t;
    for (const auto& [name, v] : w) if (!wanted || wanted(name)) t[name] = tensor(v.second, v.first);
    io::save_safetensors(path, t);
}
static void write_file(const std::string& path, const std::string& text) { std::ofstream(path) << text; }

using Rows = std::vector<std::vector<double>>;  // [time][features]
// x M^T + bias for a matrix stored [out, in] (torch Linear)
static Rows torch_linear(const Weights& w, const std::string& name, const Rows& x) {
    const auto& [shape, m] = w.at(name + ".weight");
    auto bias = w.find(name + ".bias");
    Rows y(x.size(), std::vector<double>((size_t)shape[0], 0.0));
    for (size_t t = 0; t < x.size(); ++t)
        for (int64_t o = 0; o < shape[0]; ++o) {
            for (int64_t i = 0; i < shape[1]; ++i) y[t][(size_t)o] += (double)m[(size_t)(o * shape[1] + i)] * x[t][(size_t)i];
            if (bias != w.end()) y[t][(size_t)o] += (double)bias->second.second[(size_t)o];
        }
    return y;
}
// Causal attention; keys and values have kv heads
static Rows plain_attention(const Rows& q, const Rows& k, const Rows& v, int heads, int kv, int hd) {
    size_t T = q.size();
    Rows out(T, std::vector<double>((size_t)(heads * hd), 0.0));
    for (int h = 0; h < heads; ++h) {
        int g = h / (heads / kv);
        for (size_t t = 0; t < T; ++t) {
            std::vector<double> p(t + 1);
            double top = -1e300, z = 0.0;
            for (size_t s = 0; s <= t; ++s) {
                p[s] = 0.0;
                for (int d = 0; d < hd; ++d) p[s] += q[t][(size_t)(h * hd + d)] * k[s][(size_t)(g * hd + d)];
                p[s] /= std::sqrt((double)hd);
                top = std::max(top, p[s]);
            }
            for (auto& x : p) z += x = std::exp(x - top);
            for (size_t s = 0; s <= t; ++s)
                for (int d = 0; d < hd; ++d) out[t][(size_t)(h * hd + d)] += p[s] / z * v[s][(size_t)(g * hd + d)];
        }
    }
    return out;
}

struct LlamaShape {
    int V = 11, D = 12, H = 4, KV = 2, HD = 4, F = 20, L = 2;
    double theta = 5000.0, eps = 1e-5;
    bool tied = false, qkv_bias = false;
    std::function<double(double)> scale_freq = [](double f) { return f; };
};
// The forward pass of transformers' LlamaForCausalLM on its own weight layout: logits [time * vocab]
static std::vector<float> plain_llama(const Weights& w, const LlamaShape& c, const Ids& ids) {
    const auto& emb = w.at("model.embed_tokens.weight").second;
    Rows x;
    for (int64_t id : ids) x.emplace_back(emb.begin() + id * c.D, emb.begin() + (id + 1) * c.D);
    auto norm = [&](const std::string& name, Rows r) {
        const auto& g = w.at(name).second;
        for (auto& row : r) {
            double ms = 0.0;
            for (double v : row) ms += v * v / (double)row.size();
            for (size_t i = 0; i < row.size(); ++i) row[i] *= (double)g[i] / std::sqrt(ms + c.eps);
        }
        return r;
    };
    // q cos + rotate_half(q) sin, the halves HD / 2 apart
    auto rotate = [&](Rows r, int heads) {
        for (size_t t = 0; t < r.size(); ++t)
            for (int h = 0; h < heads; ++h)
                for (int j = 0; j < c.HD / 2; ++j) {
                    double angle = (double)t * c.scale_freq(std::pow(c.theta, -2.0 * j / c.HD));
                    double& a = r[t][(size_t)(h * c.HD + j)];
                    double& b = r[t][(size_t)(h * c.HD + c.HD / 2 + j)];
                    double a0 = a, b0 = b;
                    a = a0 * std::cos(angle) - b0 * std::sin(angle);
                    b = b0 * std::cos(angle) + a0 * std::sin(angle);
                }
        return r;
    };
    auto add_rows = [](Rows& a, const Rows& b) {
        for (size_t t = 0; t < a.size(); ++t) for (size_t i = 0; i < a[t].size(); ++i) a[t][i] += b[t][i];
    };
    for (int l = 0; l < c.L; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        auto h = norm(p + "input_layernorm.weight", x);
        auto q = rotate(torch_linear(w, p + "self_attn.q_proj", h), c.H), k = rotate(torch_linear(w, p + "self_attn.k_proj", h), c.KV);
        add_rows(x, torch_linear(w, p + "self_attn.o_proj", plain_attention(q, k, torch_linear(w, p + "self_attn.v_proj", h), c.H, c.KV, c.HD)));
        h = norm(p + "post_attention_layernorm.weight", x);
        auto gate = torch_linear(w, p + "mlp.gate_proj", h), up = torch_linear(w, p + "mlp.up_proj", h);
        for (size_t t = 0; t < gate.size(); ++t) for (size_t i = 0; i < gate[t].size(); ++i) gate[t][i] = gate[t][i] / (1.0 + std::exp(-gate[t][i])) * up[t][i];
        add_rows(x, torch_linear(w, p + "mlp.down_proj", gate));
    }
    x = norm("model.norm.weight", x);
    const auto& head = c.tied ? emb : w.at("lm_head.weight").second;
    std::vector<float> logits;
    for (const auto& row : x)
        for (int v = 0; v < c.V; ++v) {
            double s = 0.0;
            for (int i = 0; i < c.D; ++i) s += row[(size_t)i] * (double)head[(size_t)(v * c.D + i)];
            logits.push_back((float)s);
        }
    return logits;
}
static Weights llama_weights(const LlamaShape& c) {
    Weights w;
    add_weight(w, "model.embed_tokens.weight", {c.V, c.D}, 1.0f);
    add_weight(w, "model.norm.weight", {c.D}, 0.1f, 1.0f);
    if (!c.tied) add_weight(w, "lm_head.weight", {c.V, c.D});
    for (int l = 0; l < c.L; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        add_weight(w, p + "input_layernorm.weight", {c.D}, 0.1f, 1.0f);
        add_weight(w, p + "post_attention_layernorm.weight", {c.D}, 0.1f, 1.0f);
        add_weight(w, p + "self_attn.q_proj.weight", {c.H * c.HD, c.D}, 0.4f);
        add_weight(w, p + "self_attn.k_proj.weight", {c.KV * c.HD, c.D}, 0.4f);
        add_weight(w, p + "self_attn.v_proj.weight", {c.KV * c.HD, c.D}, 0.4f);
        add_weight(w, p + "self_attn.o_proj.weight", {c.D, c.H * c.HD}, 0.4f);
        for (const char* n : {"q_proj", "k_proj", "v_proj"}) if (c.qkv_bias) add_weight(w, p + "self_attn." + n + ".bias", {(std::string(n) == "q_proj" ? c.H : c.KV) * c.HD});
        add_weight(w, p + "mlp.gate_proj.weight", {c.F, c.D});
        add_weight(w, p + "mlp.up_proj.weight", {c.F, c.D});
        add_weight(w, p + "mlp.down_proj.weight", {c.D, c.F});
    }
    return w;
}

static std::string llama_checkpoints() {
    const Ids ids = {3, 1, 4, 1, 5, 9, 2, 6};
    auto input_ids = id_tensor(ids, {1, 8});
    NoGradGuard no_grad;
    {
        // Two shards, grouped queries, a head width of its own, an untied output
        TempDir dir("llama");
        LlamaShape c;
        auto w = llama_weights(c);
        auto first = [](const std::string& n) { return n.find("layers.0") != std::string::npos; };
        save_weights(w, dir.path + "/model-00001-of-00002.safetensors", first);
        save_weights(w, dir.path + "/model-00002-of-00002.safetensors", [&](const std::string& n) { return !first(n); });
        std::string index = "{\"metadata\": {\"total_size\": 1}, \"weight_map\": {";
        bool comma = false;
        for (const auto& [name, v] : w) {
            index += std::string(comma ? ", " : "") + json::quote(name) + ": \"model-0000" + (first(name) ? "1" : "2") + "-of-00002.safetensors\"";
            comma = true;
        }
        write_file(dir.path + "/model.safetensors.index.json", index + "}}");
        write_file(dir.path + "/config.json",
                   "{\"model_type\": \"llama\", \"hidden_act\": \"silu\", \"vocab_size\": 11, \"hidden_size\": 12, \"intermediate_size\": 20, \"num_hidden_layers\": 2,"
                   " \"num_attention_heads\": 4, \"num_key_value_heads\": 2, \"head_dim\": 4, \"rms_norm_eps\": 1e-05, \"rope_theta\": 5000.0, \"rope_scaling\": null,"
                   " \"max_position_embeddings\": 32, \"tie_word_embeddings\": false, \"attention_bias\": false}");
        auto model = load_llama(dir.path);
        CHECK(near(model.forward(input_ids)->to_vector(), plain_llama(w, c, ids), 1e-4f, 1e-4f));
        if (load_llama(dir.path, 8).max_seq_len != 8) return "max_seq_len";
        // Saved and loaded again, and the saved file holds the original layout
        TempDir saved("llama_saved");
        save_llama(model, saved.path);
        auto tensors = io::load_safetensors(saved.path + "/model.safetensors");
        if (tensors.size() != w.size()) return "save_llama wrote " + std::to_string(tensors.size()) + " tensors";
        for (const auto& [name, v] : w) {
            auto err = near(tensors.at(name), v.second, 0.0f, 0.0f);
            if (!err.empty() || tensors.at(name)->shape != v.first) return "save_llama: " + name + " " + err;
        }
        auto again = load_llama(saved.path);
        CHECK(near(again.forward(input_ids), model.forward(input_ids), 0.0f, 0.0f));
        for (Device d : devices()) {
            again.to(d);
            CHECK(near(again.forward(input_ids->to(d))->to_vector(), plain_llama(w, c, ids), 1e-3f, 1e-3f));
        }
    }
    {
        // Qwen2: biases on the projections, a tied output, scaled RoPE
        TempDir dir("qwen");
        LlamaShape c;
        c.tied = c.qkv_bias = true;
        c.HD = 3 * 2;
        c.D = 24;
        c.scale_freq = [](double f) { return f / 4.0; };
        auto w = llama_weights(c);
        save_weights(w, dir.path + "/model.safetensors");
        write_file(dir.path + "/config.json",
                   "{\"model_type\": \"qwen2\", \"vocab_size\": 11, \"hidden_size\": 24, \"intermediate_size\": 20, \"num_hidden_layers\": 2, \"num_attention_heads\": 4,"
                   " \"num_key_value_heads\": 2, \"rms_norm_eps\": 1e-05, \"rope_theta\": 5000.0, \"rope_scaling\": {\"type\": \"linear\", \"factor\": 4.0},"
                   " \"max_position_embeddings\": 64, \"tie_word_embeddings\": true, \"use_sliding_window\": false, \"sliding_window\": 4}");
        auto model = load_llama(dir.path);
        if (model.max_seq_len != 64) return "an unused sliding window limited the length";
        CHECK(near(model.forward(input_ids)->to_vector(), plain_llama(w, c, ids), 1e-4f, 1e-4f));
        // A checkpoint that does not fit fails with the tensor's name
        w.erase("model.layers.1.mlp.up_proj.weight");
        save_weights(w, dir.path + "/model.safetensors");
        try {
            load_llama(dir.path);
            return "a missing tensor did not throw";
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("mlp.up_proj") == std::string::npos) return std::string("unclear error: ") + e.what();
        }
        write_file(dir.path + "/config.json", "{\"model_type\": \"mamba\"}");
        CHECK(expect_throw([&] { load_llama(dir.path); }, "an unknown model type"));
    }
    // Llama 3's scaling keeps short wavelengths, divides long ones and is continuous in between
    RopeScaling s;
    s.type = RopeScaling::Type::Llama3;
    s.factor = 8.0f;
    const float two_pi = 6.283185307f;
    for (auto [wavelength, want] : {std::pair<float, float>{100.0f, 1.0f}, {2047.0f, 1.0f}, {2049.0f, 0.9996f}, {8190.0f, 0.1252f}, {8200.0f, 0.125f}, {1e6f, 0.125f}}) {
        float f = two_pi / wavelength;
        if (std::fabs(s.apply(f) / f - want) > 2e-3f) return "llama3 rope scaling at wavelength " + std::to_string(wavelength);
    }
    return "";
}

static std::string gpt2_checkpoints() {
    TempDir dir("gpt2");
    const int V = 11, P = 16, D = 12, H = 3, L = 2;
    Weights w;
    add_weight(w, "transformer.wte.weight", {V, D}, 1.0f);
    add_weight(w, "transformer.wpe.weight", {P, D}, 0.5f);
    auto norm_weights = [&](const std::string& n) {
        add_weight(w, n + ".weight", {D}, 0.1f, 1.0f);
        add_weight(w, n + ".bias", {D}, 0.1f);
    };
    auto conv1d = [&](const std::string& n, int in, int out) {
        add_weight(w, n + ".weight", {in, out}, 0.4f);
        add_weight(w, n + ".bias", {out}, 0.1f);
    };
    norm_weights("transformer.ln_f");
    for (int l = 0; l < L; ++l) {
        const std::string p = "transformer.h." + std::to_string(l) + ".";
        norm_weights(p + "ln_1");
        norm_weights(p + "ln_2");
        conv1d(p + "attn.c_attn", D, 3 * D);
        conv1d(p + "attn.c_proj", D, D);
        conv1d(p + "mlp.c_fc", D, 4 * D);
        conv1d(p + "mlp.c_proj", 4 * D, D);
    }
    save_weights(w, dir.path + "/model.safetensors");
    write_file(dir.path + "/config.json",
               "{\"model_type\": \"gpt2\", \"activation_function\": \"gelu_new\", \"vocab_size\": 11, \"n_positions\": 16, \"n_embd\": 12, \"n_head\": 3, \"n_layer\": 2,"
               " \"layer_norm_epsilon\": 1e-05, \"n_inner\": null}");

    // The forward pass in plain loops
    const Ids ids = {3, 1, 4, 1, 5, 9};
    auto apply = [&](const std::string& n, const Rows& x) {
        const auto& [shape, m] = w.at(n + ".weight");
        const auto& b = w.at(n + ".bias").second;
        Rows y(x.size(), std::vector<double>((size_t)shape[1]));
        for (size_t t = 0; t < x.size(); ++t)
            for (int64_t o = 0; o < shape[1]; ++o) {
                y[t][(size_t)o] = (double)b[(size_t)o];
                for (int64_t i = 0; i < shape[0]; ++i) y[t][(size_t)o] += x[t][(size_t)i] * (double)m[(size_t)(i * shape[1] + o)];
            }
        return y;
    };
    auto norm = [&](const std::string& n, Rows x) {
        const auto& g = w.at(n + ".weight").second;
        const auto& b = w.at(n + ".bias").second;
        for (auto& row : x) {
            double mu = std::accumulate(row.begin(), row.end(), 0.0) / (double)row.size(), var = 0.0;
            for (double v : row) var += (v - mu) * (v - mu) / (double)row.size();
            for (size_t i = 0; i < row.size(); ++i) row[i] = (row[i] - mu) / std::sqrt(var + 1e-5) * (double)g[i] + (double)b[i];
        }
        return x;
    };
    auto add_rows = [](Rows& a, const Rows& b) {
        for (size_t t = 0; t < a.size(); ++t) for (size_t i = 0; i < a[t].size(); ++i) a[t][i] += b[t][i];
    };
    Rows x;
    const auto& wte = w.at("transformer.wte.weight").second;
    const auto& wpe = w.at("transformer.wpe.weight").second;
    for (size_t t = 0; t < ids.size(); ++t) {
        x.emplace_back();
        for (int i = 0; i < D; ++i) x.back().push_back((double)wte[(size_t)(ids[t] * D + i)] + (double)wpe[t * (size_t)D + (size_t)i]);
    }
    for (int l = 0; l < L; ++l) {
        const std::string p = "transformer.h." + std::to_string(l) + ".";
        auto qkv = apply(p + "attn.c_attn", norm(p + "ln_1", x));
        Rows parts[3];
        for (const auto& row : qkv) for (int k = 0; k < 3; ++k) parts[k].emplace_back(row.begin() + k * D, row.begin() + (k + 1) * D);
        add_rows(x, apply(p + "attn.c_proj", plain_attention(parts[0], parts[1], parts[2], H, H, D / H)));
        auto h = apply(p + "mlp.c_fc", norm(p + "ln_2", x));
        for (auto& row : h) for (auto& v : row) v = 0.5 * v * (1.0 + std::tanh(std::sqrt(2.0 / 3.14159265358979323846) * (v + 0.044715 * v * v * v)));
        add_rows(x, apply(p + "mlp.c_proj", h));
    }
    x = norm("transformer.ln_f", x);
    std::vector<float> want;
    for (const auto& row : x)
        for (int v = 0; v < V; ++v) {
            double s = 0.0;
            for (int i = 0; i < D; ++i) s += row[(size_t)i] * (double)wte[(size_t)(v * D + i)];
            want.push_back((float)s);
        }
    NoGradGuard no_grad;
    auto model = load_gpt2(dir.path);
    CHECK(near(model.forward(id_tensor(ids, {1, 6}))->to_vector(), want, 1e-4f, 1e-4f));
    // The other naming, without a prefix
    Weights bare;
    for (const auto& [name, v] : w) bare[name.substr(12)] = v;
    save_weights(bare, dir.path + "/model.safetensors");
    return near(load_gpt2(dir.path).forward(id_tensor(ids, {1, 6}))->to_vector(), want, 1e-4f, 1e-4f);
}

// ---------------------------------------------------------------------------------------------
// Adapters and experts
// ---------------------------------------------------------------------------------------------

static std::string lora() {
    manual_seed(11);
    auto c = small_config();
    ModernTransformer model(c);
    model.eval();
    auto ids = id_tensor({1, 2, 3, 4, 5, 6}, {2, 3});
    auto before = model.forward(ids)->detach();
    int n = add_lora(model, 2, 4.0f, {"W_q", "W_v"});
    if (n != 4) return "add_lora adapted " + std::to_string(n) + " layers";
    CHECK(near(model.forward(ids), before, 1e-6f));  // B starts at zero
    auto params = trainable_parameters(model);
    if (params.size() != 8) return "trainable parameters: " + std::to_string(params.size());
    std::vector<std::vector<float>> frozen;
    for (auto& p : model.parameters()) if (!p->requires_grad) frozen.push_back(p->to_vector());

    Adam opt(0.05f);
    float first = 0.0f, last = 0.0f;
    for (int step = 0; step < 10; ++step) {
        model.zero_grad();
        auto loss = fused_cross_entropy(model.forward(ids), id_tensor({2, 3, 4, 5, 6, 7}, {2, 3}));
        loss->backward();
        (step == 0 ? first : last) = loss->item();
        opt.step(params);
    }
    if (!(last < first)) return "adapters did not learn";
    size_t i = 0;
    for (auto& p : model.parameters()) {
        if (p->requires_grad) continue;
        if (p->grad) return "a frozen parameter got a gradient";
        CHECK(near(p, frozen[i++], 0.0f, 0.0f));
    }
    auto tuned = model.forward(ids)->detach();
    if (near(tuned, before, 1e-4f).empty()) return "training changed nothing";

    TempDir dir("lora");
    save_lora(model, dir.path + "/a.safetensors");
    manual_seed(11);
    ModernTransformer fresh(c);
    fresh.eval();
    load_lora(fresh, dir.path + "/a.safetensors");
    CHECK(near(fresh.forward(ids), tuned, 1e-5f));
    merge_lora(model);
    CHECK(near(model.forward(ids), tuned, 1e-4f, 1e-4f));
    if (!trainable_parameters(model).empty()) return "adapters left after merging";
    CHECK(expect_throw([&] { add_lora(model, 2, 1.0f, {"no_such_layer"}); }, "a target that matches nothing"));

    // Dropout touches the adapter only in training
    Linear l(4, 3);
    l.enable_lora(2, 2.0f, 0.5f);
    l.lora_B->fill_(1.0f);
    auto x = ones({1, 4});
    l.eval();
    CHECK(near(l.forward(x), l.forward(x), 0.0f, 0.0f));
    l.train();
    bool differs = false;
    auto y = l.forward(x);
    for (int k = 0; k < 20; ++k) differs = differs || !near(l.forward(x), y, 1e-6f).empty();
    return differs ? "" : "LoRA dropout never dropped";
}

static std::string no_bias_linear() {
    Linear l(3, 2, 0.0f, /*use_bias=*/false);
    if (l.b || l.parameters().size() != 1) return "a bias-free layer has a bias";
    for (Device d : devices()) {
        l.to(d);
        auto x = input({2, 5, 3})->to(d);
        CHECK(near(l.forward(x), matmul(x, l.W), 1e-6f));
        sum(square(l.forward(x)))->backward();
        if (!l.W->grad) return "no gradient";
        l.zero_grad();
    }
    Linear with(3, 2, 0.5f);
    return near(with.b, {0.5f, 0.5f});
}

static std::string experts() {
    manual_seed(21);
    const int D = 6, E = 4;
    MixtureOfExperts moe(D, 8, E, 2);
    for (Device d : devices()) {
        moe.to(d);
        auto x = input({2, 5, D})->to(d);
        auto y = moe.forward(x);
        CHECK(shape_is(y, {2, 5, D}));
        // Token by token
        auto probs = softmax(moe.router.forward(reshape(x, {10, D})))->to_vector();
        std::vector<float> want;
        Ids used((size_t)E, 0);
        {
            NoGradGuard no_grad;
            for (int t = 0; t < 10; ++t) {
                const float* p = probs.data() + t * E;
                std::vector<int> order(E);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
                auto xt = contiguous(narrow(reshape(x, {10, D}), 0, t, 1));
                auto total = zeros({1, D}, d);
                for (int k = 0; k < 2; ++k) {
                    total = total + moe.experts[(size_t)order[(size_t)k]]->forward(xt) * (p[order[(size_t)k]] / (p[order[0]] + p[order[1]]));
                    ++used[(size_t)order[(size_t)k]];
                }
                auto v = total->to_vector();
                want.insert(want.end(), v.begin(), v.end());
            }
        }
        CHECK(near(y->to_vector(), want, 1e-4f, 1e-4f));
        if (used != moe.tokens_per_expert) return "tokens_per_expert";
        sum(square(y) + moe.aux_loss)->backward();
        if (!moe.router.W->grad) return "the router got no gradient";
        for (int e = 0; e < E; ++e) if ((moe.experts[(size_t)e]->w_up.W->grad != nullptr) != (used[(size_t)e] > 0)) return "expert gradients";
        moe.zero_grad();
    }
    moe.to(Device::CPU);
    // Gradients, away from where the choice of experts changes
    auto x = input({1, 3, D});
    auto err = dummygrad::gradcheck([&](const TensorList& in) { return moe.forward(in[0]); }, {x});
    if (!err.empty()) return err;
    // The balance loss is 1 when the router is indifferent, and the number of experts when one takes everything
    moe.router.W->fill_(0.0f);
    moe.forward(x);
    CHECK(scalar(moe.aux_loss, 1.0f, 1e-5f));
    MixtureOfExperts one(D, 8, E, 1);
    one.router.W->fill_(0.0f);
    for (int i = 0; i < D; ++i) one.router.W->data_ptr<float>()[i * E + 2] = 30.0f;
    one.forward(ones({4, D}));
    CHECK(scalar(one.aux_loss, (float)E, 1e-4f));
    return expect_throw([] { MixtureOfExperts(4, 4, 2, 3); }, "top_k above the number of experts");
}

int main(int argc, char** argv) {
    if (argc > 1) g_filter = argv[1];
    manual_seed(0);
    std::cout << "dummygrad model test suite (CUDA " << (has_cuda() ? "available" : "not available") << ")\n";
    run("operators", operators);
    run("indexing", indexing);
    run("index_assignment", index_assignment);
    run("printing", printing);
    run("kv_cache", kv_cache);
    run("generation", generation);
    run("sampling", sampling);
    run("beam_search", beams);
    run("json", json_values);
    run("llama_checkpoints", llama_checkpoints);
    run("gpt2_checkpoints", gpt2_checkpoints);
    run("no_bias_linear", no_bias_linear);
    run("lora", lora);
    run("experts", experts);
    return report();
}

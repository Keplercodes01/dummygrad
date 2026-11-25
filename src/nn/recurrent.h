#pragma once
// Recurrent layers with PyTorch's equations: RNN, LSTM and GRU cells and multi-layer, optionally
// bidirectional sequence modules. Sequences are [batch, time, features].

#include "layers.h"

// One matrix [hidden, input] per gate group, stored as [input, gates * hidden] to multiply from the right
namespace recurrent_detail {
inline std::shared_ptr<Tensor> make_weight(int64_t in, int64_t out, int64_t hidden) {
    auto w = std::make_shared<Tensor>(std::vector<int64_t>{in, out}, true);
    float bound = 1.0f / std::sqrt((float)hidden);
    init::uniform_(w, -bound, bound);
    return w;
}
inline std::shared_ptr<Tensor> make_bias(int64_t out) { return zeros({out}, Device::CPU, DType::Float32, true); }
}  // namespace recurrent_detail

class RNNCell : public Module {
public:
    std::shared_ptr<Tensor> weight_ih, weight_hh, bias;
    int64_t input_size, hidden_size;
    bool use_relu;
    RNNCell(int64_t input_size, int64_t hidden_size, bool use_relu = false) : input_size(input_size), hidden_size(hidden_size), use_relu(use_relu) {
        weight_ih = recurrent_detail::make_weight(input_size, hidden_size, hidden_size);
        weight_hh = recurrent_detail::make_weight(hidden_size, hidden_size, hidden_size);
        bias = recurrent_detail::make_bias(hidden_size);
    }
    // h' = tanh(x W_ih + h W_hh + b); x [batch, input], h [batch, hidden] (null: zeros)
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& h_in = nullptr) {
        return step(linear(x, weight_ih, bias), h_in ? h_in : full({x->shape[0], hidden_size}, 0.0f, x->device));
    }
    // From the input's projection, so a sequence can project every step at once
    std::shared_ptr<Tensor> step(const std::shared_ptr<Tensor>& xp, const std::shared_ptr<Tensor>& h) {
        auto a = add(xp, matmul(h, weight_hh));
        return use_relu ? relu(a) : tanh(a);
    }
    void visit(ModuleVisitor& v) override {
        v.parameter("weight_ih", weight_ih);
        v.parameter("weight_hh", weight_hh);
        v.parameter("bias", bias);
    }
};

class LSTMCell : public Module {
public:
    using State = std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>;  // hidden, cell
    std::shared_ptr<Tensor> weight_ih, weight_hh, bias;
    int64_t input_size, hidden_size;
    LSTMCell(int64_t input_size, int64_t hidden_size) : input_size(input_size), hidden_size(hidden_size) {
        weight_ih = recurrent_detail::make_weight(input_size, 4 * hidden_size, hidden_size);
        weight_hh = recurrent_detail::make_weight(hidden_size, 4 * hidden_size, hidden_size);
        bias = recurrent_detail::make_bias(4 * hidden_size);
    }
    State zero_state(int64_t batch, Device device) const { return {full({batch, hidden_size}, 0.0f, device), full({batch, hidden_size}, 0.0f, device)}; }
    State forward(const std::shared_ptr<Tensor>& x, const State& s_in = {}) {
        return step(linear(x, weight_ih, bias), s_in.first ? s_in : zero_state(x->shape[0], x->device));
    }
    State step(const std::shared_ptr<Tensor>& xp, const State& s) {
        // Gates in PyTorch's order: input, forget, cell, output
        auto gates = chunk(add(xp, matmul(s.first, weight_hh)), 4, 1);
        auto i = sigmoid(gates[0]), f = sigmoid(gates[1]), g = tanh(gates[2]), o = sigmoid(gates[3]);
        auto c = add(mul(f, s.second), mul(i, g));
        return {mul(o, tanh(c)), c};
    }
    void visit(ModuleVisitor& v) override {
        v.parameter("weight_ih", weight_ih);
        v.parameter("weight_hh", weight_hh);
        v.parameter("bias", bias);
    }
};

class GRUCell : public Module {
public:
    std::shared_ptr<Tensor> weight_ih, weight_hh, bias_ih, bias_hh;
    int64_t input_size, hidden_size;
    GRUCell(int64_t input_size, int64_t hidden_size) : input_size(input_size), hidden_size(hidden_size) {
        weight_ih = recurrent_detail::make_weight(input_size, 3 * hidden_size, hidden_size);
        weight_hh = recurrent_detail::make_weight(hidden_size, 3 * hidden_size, hidden_size);
        bias_ih = recurrent_detail::make_bias(3 * hidden_size);
        bias_hh = recurrent_detail::make_bias(3 * hidden_size);
    }
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x, const std::shared_ptr<Tensor>& h_in = nullptr) {
        return step(linear(x, weight_ih, bias_ih), h_in ? h_in : full({x->shape[0], hidden_size}, 0.0f, x->device));
    }
    std::shared_ptr<Tensor> step(const std::shared_ptr<Tensor>& xp, const std::shared_ptr<Tensor>& h) {
        // Gates in PyTorch's order: reset, update, new; the reset gate scales the recurrent term
        auto xg = chunk(xp, 3, 1), hg = chunk(linear(h, weight_hh, bias_hh), 3, 1);
        auto r = sigmoid(add(xg[0], hg[0])), z = sigmoid(add(xg[1], hg[1]));
        auto n = tanh(add(xg[2], mul(r, hg[2])));
        return add(n, mul(z, sub(h, n)));
    }
    void visit(ModuleVisitor& v) override {
        v.parameter("weight_ih", weight_ih);
        v.parameter("weight_hh", weight_hh);
        v.parameter("bias_ih", bias_ih);
        v.parameter("bias_hh", bias_hh);
    }
};

struct RecurrentOptions {
    int64_t num_layers = 1;
    float dropout = 0.0f;  // between layers, in train mode
    bool bidirectional = false;
};

// Output [batch, time, directions * hidden] and each layer and direction's final hidden state
struct RecurrentOutput {
    std::shared_ptr<Tensor> output;
    std::shared_ptr<Tensor> h;  // [layers * directions, batch, hidden]
    std::shared_ptr<Tensor> c;  // the LSTM's cell state, else null
};

// Shared by RNN, LSTM and GRU. The input of every step is projected in one matrix product per layer.
template <typename Cell>
class RecurrentBase : public Module {
public:
    std::vector<std::shared_ptr<Cell>> cells;
    RecurrentOptions options;

    template <typename... Args>
    RecurrentBase(int64_t input_size, int64_t hidden_size, RecurrentOptions opt, Args... args) : options(opt) {
        int64_t directions = options.bidirectional ? 2 : 1;
        for (int64_t l = 0; l < options.num_layers; ++l) {
            int64_t in = l == 0 ? input_size : directions * hidden_size;
            for (int64_t d = 0; d < directions; ++d) cells.push_back(std::make_shared<Cell>(in, hidden_size, args...));
        }
    }

    // x [batch, time, features]; h0 and c0 [layers * directions, batch, hidden], null for zeros
    RecurrentOutput forward(const std::shared_ptr<Tensor>& x_in, const std::shared_ptr<Tensor>& h0 = nullptr, const std::shared_ptr<Tensor>& c0 = nullptr) {
        DG_PROFILE_OP("recurrent");
        if (x_in->ndim() != 3) throw std::invalid_argument("recurrent layer: input must be [batch, time, features]");
        int64_t directions = options.bidirectional ? 2 : 1, T = x_in->shape[1];
        auto x = x_in;
        TensorList final_h, final_c;
        for (int64_t l = 0; l < options.num_layers; ++l) {
            if (l > 0) x = ::dropout(x, options.dropout, training_);
            TensorList sides;
            for (int64_t d = 0; d < directions; ++d) {
                int64_t index = l * directions + d;
                auto& cell = *cells[(size_t)index];
                auto xp = project(cell, x);
                auto state = initial(cell, x, h0 ? select(h0, 0, index) : nullptr, c0 ? select(c0, 0, index) : nullptr);
                TensorList outs((size_t)T);
                for (int64_t i = 0; i < T; ++i) {
                    int64_t t = d == 0 ? i : T - 1 - i;
                    state = cell.step(select(xp, 1, t), state);
                    outs[(size_t)t] = hidden_of(state);
                }
                sides.push_back(stack(outs, 1));
                final_h.push_back(hidden_of(state));
                if (auto c = cell_of(state)) final_c.push_back(c);
            }
            x = directions == 1 ? sides[0] : cat(sides, 2);
        }
        return {x, stack(final_h, 0), final_c.empty() ? nullptr : stack(final_c, 0)};
    }

    void visit(ModuleVisitor& v) override {
        for (size_t i = 0; i < cells.size(); ++i) v.module("cells." + std::to_string(i), *cells[i]);
    }

private:
    using TensorP = std::shared_ptr<Tensor>;
    static TensorP project(RNNCell& c, const TensorP& x) { return linear(x, c.weight_ih, c.bias); }
    static TensorP project(LSTMCell& c, const TensorP& x) { return linear(x, c.weight_ih, c.bias); }
    static TensorP project(GRUCell& c, const TensorP& x) { return linear(x, c.weight_ih, c.bias_ih); }
    static LSTMCell::State initial(LSTMCell& c, const TensorP& x, const TensorP& h, const TensorP& cs) {
        auto zero = c.zero_state(x->shape[0], x->device);
        return {h ? h : zero.first, cs ? cs : zero.second};
    }
    template <typename C>
    static TensorP initial(C& c, const TensorP& x, const TensorP& h, const TensorP&) {
        return h ? h : full({x->shape[0], c.hidden_size}, 0.0f, x->device);
    }
    static TensorP hidden_of(const TensorP& s) { return s; }
    static TensorP hidden_of(const LSTMCell::State& s) { return s.first; }
    static TensorP cell_of(const TensorP&) { return nullptr; }
    static TensorP cell_of(const LSTMCell::State& s) { return s.second; }
};

class RNN : public RecurrentBase<RNNCell> {
public:
    RNN(int64_t input_size, int64_t hidden_size, RecurrentOptions opt = {}, bool use_relu = false) : RecurrentBase(input_size, hidden_size, opt, use_relu) {}
};
class LSTM : public RecurrentBase<LSTMCell> {
public:
    LSTM(int64_t input_size, int64_t hidden_size, RecurrentOptions opt = {}) : RecurrentBase(input_size, hidden_size, opt) {}
};
class GRU : public RecurrentBase<GRUCell> {
public:
    GRU(int64_t input_size, int64_t hidden_size, RecurrentOptions opt = {}) : RecurrentBase(input_size, hidden_size, opt) {}
};

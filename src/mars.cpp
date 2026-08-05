#include "engine.h"
#include "linear.h"
#include "embedding.h"
#include "transformer.h"
#include "broadcasting.h"
#include "loss.h"
#include "optimizers.h"
#include <iostream>
#include <fstream>
#include <vector>

// Custom Slice operation for taking the last element in a sequence
// Equivalent to PyTorch's x[:, -1, :]
struct SliceLastBackward : public Node {
    std::vector<int> in_shape;
    SliceLastBackward(std::vector<int> shape) : in_shape(std::move(shape)) {}

    std::vector<std::shared_ptr<Tensor>> apply(const std::vector<std::shared_ptr<Tensor>>& grads) override {
        auto out_grad = std::make_shared<Tensor>(in_shape, false);
        int B = in_shape[0];
        int S = in_shape[1];
        int D = in_shape[2];
        const float* g_ptr = grads[0]->data_ptr();
        float* out_ptr = out_grad->data_ptr();
        for (int b = 0; b < B; b++) {
            for (int d = 0; d < D; d++) {
                out_ptr[b * S * D + (S - 1) * D + d] = g_ptr[b * D + d];
            }
        }
        return {out_grad};
    }
};

inline std::shared_ptr<Tensor> slice_last(const std::shared_ptr<Tensor>& x) {
    if (x->shape.size() != 3) {
        throw std::runtime_error("slice_last: expected a 3D tensor [B, S, D]");
    }
    int B = x->shape[0];
    int S = x->shape[1];
    int D = x->shape[2];
    auto out = std::make_shared<Tensor>(std::vector<int>{B, D}, x->requires_grad);
    
    const float* x_ptr = x->data_ptr();
    float* out_ptr = out->data_ptr();
    
    for (int b = 0; b < B; b++) {
        for (int d = 0; d < D; d++) {
            out_ptr[b * D + d] = x_ptr[b * S * D + (S - 1) * D + d];
        }
    }
    
    if (x->requires_grad) {
        auto grad_fn = std::make_shared<SliceLastBackward>(x->shape);
        grad_fn->add_next_edge(get_grad_edge(x).function, 0);
        out->grad_fn = grad_fn;
    }
    return out;
}

class ActionEncoder {
public:
    Linear proj;
    ActionEncoder(int action_dim, int d_model) : proj(action_dim, d_model) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        return proj.forward(x);
    }
    std::vector<std::shared_ptr<Tensor>> parameters() const {
        return proj.parameters();
    }
};

class ActionDecoder {
public:
    Linear proj;
    ActionDecoder(int d_model, int action_dim) : proj(d_model, action_dim) {}
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        return proj.forward(x);
    }
    std::vector<std::shared_ptr<Tensor>> parameters() const {
        return proj.parameters();
    }
};

class Mars {
public:
    ActionEncoder encoder;
    Embedding pos_emb;
    std::vector<std::shared_ptr<TransformerBlock>> blocks;
    ActionDecoder decoder;

    int action_dim;
    int d_model;
    int num_heads;
    int max_seq;

    Mars(int action_dim = 156, int d_model = 256, int num_heads = 8, int max_seq = 128, int num_layers = 1)
        : encoder(action_dim, d_model),
          pos_emb(max_seq, d_model),
          decoder(d_model, action_dim),
          action_dim(action_dim),
          d_model(d_model),
          num_heads(num_heads),
          max_seq(max_seq) {
        for (int i = 0; i < num_layers; i++) {
            blocks.push_back(std::make_shared<TransformerBlock>(d_model, num_heads, true));
        }
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        auto enc_x = encoder.forward(x); // [B, S, d_model]
        int S = enc_x->shape[1];

        // Create positions tensor [S]
        auto positions = std::make_shared<Tensor>(std::vector<int>{S}, false);
        for (int i = 0; i < S; i++) {
            positions->data_ptr()[i] = (float)i;
        }

        auto pos_x = pos_emb.forward(positions); // [S, d_model]
        
        // Add embeddings using broadcasting addition
        auto h = cast_n_add(enc_x, pos_x);

        for (auto& block : blocks) {
            h = block->forward(h);
        }

        // slice last element: h[:, -1, :]
        auto last_h = slice_last(h);

        return decoder.forward(last_h);
    }

    std::vector<std::shared_ptr<Tensor>> parameters() const {
        auto p = encoder.parameters();
        auto pp = pos_emb.parameters();
        p.insert(p.end(), pp.begin(), pp.end());
        for (auto& block : blocks) {
            auto pb = block->parameters();
            p.insert(p.end(), pb.begin(), pb.end());
        }
        auto pd = decoder.parameters();
        p.insert(p.end(), pd.begin(), pd.end());
        return p;
    }
};

int main() {
    std::cout << "🚀 TRAINING MARS ON AMASS MOCAP DATA 🚀" << std::endl;

    // 1. Load AMASS Dataset
    std::ifstream file("amass_poses.bin", std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open amass_poses.bin!" << std::endl;
        return 1;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    int num_floats = size / sizeof(float);
    int total_frames = num_floats / 156;
    
    // Limit to 100 frames as requested for fast testing
    total_frames = std::min(total_frames, 100);
    
    std::vector<float> data(num_floats);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        std::cerr << "Failed to read data!" << std::endl;
        return 1;
    }
    std::cout << "Loaded " << total_frames << " frames of 156-dim AMASS data." << std::endl;

    // Normalize data (Z-score) to prevent NaNs
    double sum = 0.0;
    for (float v : data) sum += v;
    double mean = sum / data.size();
    
    double sq_sum = 0.0;
    for (float v : data) sq_sum += (v - mean) * (v - mean);
    double std_dev = std::sqrt(sq_sum / data.size()) + 1e-8;
    
    for (float& v : data) {
        v = (float)((v - mean) / std_dev);
    }
    std::cout << "Normalized data to mean 0, std 1." << std::endl;

    // 2. Hyperparameters
    int action_dim = 156;
    int max_seq = 2; // Reduced for debugging
    int batch_size = 1;
    int d_model = 128; // Smaller for speed
    int num_heads = 4;
    int num_layers = 1;

    // 3. Instantiate Model and Optimizer
    Mars model(action_dim, d_model, num_heads, max_seq, num_layers);
    Adam optimizer(0.001f);
    auto params = model.parameters();
    std::cout << "Model parameters: " << params.size() << " tensors." << std::endl;

    // 4. Training Loop
    int num_steps = 20;
    std::cout << "\nStarting " << num_steps << " training steps..." << std::endl;

    try {
        for (int step = 1; step <= num_steps; step++) {
            // Sample a random batch
            auto input = std::make_shared<Tensor>(std::vector<int>{batch_size, max_seq, action_dim}, false);
            auto target = std::make_shared<Tensor>(std::vector<int>{batch_size, action_dim}, false);
            
            float* in_ptr = input->data_ptr();
            float* tgt_ptr = target->data_ptr();

            for (int b = 0; b < batch_size; b++) {
                // Pick a random starting frame (ensuring we have enough frames for max_seq + 1)
                int start_idx = rand() % (total_frames - max_seq - 1);
                
                // Copy [max_seq] frames for input
                for (int s = 0; s < max_seq; s++) {
                    for (int d = 0; d < action_dim; d++) {
                        in_ptr[b * max_seq * action_dim + s * action_dim + d] = data[(start_idx + s) * action_dim + d];
                    }
                }
                // Copy [1] frame (the very next one) for target
                for (int d = 0; d < action_dim; d++) {
                    tgt_ptr[b * action_dim + d] = data[(start_idx + max_seq) * action_dim + d];
                }
            }

            // Zero gradients
            for (auto& p : params) p->zero_grad();

            // Forward pass
            auto pred = model.forward(input);

            // Compute MSE Loss
            auto loss = mse(pred, target);
            float current_loss = loss->data_at(0);

            std::cout << "Step " << step << "/" << num_steps << " | MSE Loss: " << current_loss << std::endl;

            // Backward pass
            loss->backward();

            // Gradient clipping
            for (auto& p : params) {
                if (p->grad) {
                    float* g_ptr = p->grad->data_ptr();
                    for (int i = 0; i < p->size(); i++) {
                        g_ptr[i] = std::max(-1.0f, std::min(1.0f, g_ptr[i]));
                    }
                }
            }

            // Optimizer step
            for (auto& p : params) optimizer.step(p);
        }
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION CAUGHT: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "UNKNOWN EXCEPTION CAUGHT" << std::endl;
    }

    std::cout << "✅ Mars training step completed successfully!" << std::endl;
    return 0;
}

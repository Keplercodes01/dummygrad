// Kernel declarations, included once per backend namespace (see kernels.h)

// Dense elementwise: y[i] = op(x[i]; p, q). x and y may be the same buffer.
void unary(Unary op, const float* x, float* y, int64_t n, float p, float q);

// Strided elementwise. Operands 0, 1 (and 2) are inputs, the last one the output.
void binary(Binary op, const float* a, const float* b, float* out, const Grid& g);
void where(const float* cond, const float* a, const float* b, float* out, const Grid& g);
void copy(const float* src, float* dst, const Grid& g);

// Dense x seen as [outer, K, inner], reduced over K into out [outer, inner]
void reduce(Reduce op, const float* x, float* out, int64_t outer, int64_t K, int64_t inner);
// Index (as a float) of the first largest or smallest element; NaN counts as both
void arg_reduce(bool largest, const float* x, float* out, int64_t outer, int64_t K, int64_t inner);
// Running reduction along K. For Max and Min, index (optional) receives the running winner's position.
void cumulate(Cumulate op, const float* x, float* out, float* index, int64_t outer, int64_t K, int64_t inner);

// out[o, m, i] = x[o, index[m], i]; indices must be in range (checked by the caller)
void index_select(const float* x, const float* index, float* out, int64_t outer, int64_t K, int64_t inner, int64_t M);
// out[o, k, i] += sum of src[o, m, i] over the m listed for k: positions[starts[k] .. starts[k + 1])
void index_add(float* out, const float* src, const std::vector<int>& positions, const std::vector<int>& starts,
               int64_t outer, int64_t K, int64_t inner, int64_t M);

void gather(const float* x, const float* index, float* out, const GatherPlan& g);
// out (x's shape) += src scattered to the positions gather reads. Each output scans the index line
// through it, so the cost is x's size times the index length along dim: for short index dimensions.
void scatter_add_scan(float* out, const float* index, const float* src, const GatherPlan& g);

// Sliding windows. col is [planes, kernel_size, out_size]; fold writes (does not accumulate into) x.
void unfold(const float* x, float* col, const Windows& w);
void fold(const float* col, float* x, const Windows& w);
// argmax is the flat input index within the plane (as a float), -1 for an all-NaN window
void max_pool(const float* x, float* out, float* argmax, const Windows& w);
void max_pool_backward(const float* g, const float* argmax, float* gx, const Windows& w);
void avg_pool(const float* x, float* out, const Windows& w);
void avg_pool_backward(const float* g, float* gx, const Windows& w);

// Counter-based random numbers (Philox 4x32-10): element i depends only on (seed, stream, i), so the
// same seed gives the same values on every device and thread count
void random_uniform(float* out, int64_t n, uint64_t seed, uint64_t stream);  // [0, 1)
void random_normal(float* out, int64_t n, uint64_t seed, uint64_t stream);
// y = x / keep where a uniform draw is below keep, else 0
void dropout(const float* x, float* y, int64_t n, float keep, uint64_t seed, uint64_t stream);

// Softmax or log-softmax over K of [outer, K, inner]
void softmax(const float* x, float* out, int64_t outer, int64_t K, int64_t inner, bool log);

// Optimizer updates over n elements. Null state pointers disable the feature that needs them.
void sgd_step(float* p, const float* g, float* buf, int64_t n, float lr, float momentum, float dampening,
              float weight_decay, bool nesterov, bool first);
void adam_step(float* p, const float* g, float* m, float* v, float* vmax, int64_t n, float lr, float beta1, float beta2,
               float eps, float weight_decay, bool decoupled, float bias1, float bias2);
void nadam_step(float* p, const float* g, float* m, float* v, int64_t n, float lr, float beta1, float beta2, float eps,
                float weight_decay, float mu_grad, float mu_avg, float bias2);
void radam_step(float* p, const float* g, float* m, float* v, int64_t n, float lr, float beta1, float beta2, float eps,
                float weight_decay, float bias1, float bias2, float rect);
void adamax_step(float* p, const float* g, float* m, float* u, int64_t n, float lr, float beta1, float beta2, float eps,
                 float weight_decay, float bias1);
void adagrad_step(float* p, const float* g, float* sum, int64_t n, float lr, float eps, float weight_decay);
void adadelta_step(float* p, const float* g, float* sq, float* acc, int64_t n, float lr, float rho, float eps,
                   float weight_decay);
void rmsprop_step(float* p, const float* g, float* sq, float* avg, float* buf, int64_t n, float lr, float alpha, float eps,
                  float weight_decay, float momentum);
void rprop_step(float* p, const float* g, float* prev, float* step, int64_t n, float eta_minus, float eta_plus,
                float step_min, float step_max);
void lion_step(float* p, const float* g, float* m, int64_t n, float lr, float beta1, float beta2, float weight_decay);
// avg = decay * avg + (1 - decay) * p
void ema_step(float* avg, const float* p, int64_t n, float decay);

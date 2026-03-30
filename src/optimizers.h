#include"engine.h"

//SGD
inline void SGD(const std::shared_ptr<Tensor>& param, const float& lr) {
    for(int i = 0; i<param->size(); i++) {
        param->data_at(i) -= lr * param->grad_at(i);
    }
}

//Adam the great 
class Adam {
    std::vector<float> m, v;
    int t = 0; 
    float lr, b1, b2, E;

    public:
        Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float E = 1e-8f)
            : lr(lr), b1(b1), b2(b2), E(E) {}

        void step(const std::shared_ptr<Tensor>& param) {
            if(m.empty()) {
                m.resize(param->size(), 0.0f);
                v.resize(param->size(), 0.0f);
            }
            t++;
            for(int i = 0; i<param->size(); i++) {
                m[i] = b1*m[i] + (1-b1)*param->grad_at(i);
                v[i] = b2*v[i] + (1-b2)*param->grad_at(i)*param->grad_at(i);

                float m_hat = m[i] / (1 - std::pow(b1, t)); 
                float v_hat = v[i] / (1 - std::pow(b2, t)); 

                //update
                param->data_at(i) -= lr * m_hat / (std::sqrt(v_hat) + E);
            }
        }
};

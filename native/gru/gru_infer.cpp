#include "gru_infer.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>

namespace gruexample {

static uint32_t read_u32(std::ifstream& f) {
    uint32_t x = 0;
    f.read(reinterpret_cast<char*>(&x), sizeof(x));
    return x;
}

static std::vector<float> read_float_block(std::ifstream& f) {
    uint32_t n = read_u32(f);
    std::vector<float> v(n);
    if (n > 0) f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    return v;
}

static float dot_product(const std::vector<float>& mat, uint32_t row, uint32_t width, const std::vector<float>& x) {
    float s = 0.0f;
    uint32_t base = row * width;
    for (uint32_t i = 0; i < width; ++i) s += mat[base + i] * x[i];
    return s;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

GruModel load_binary_model(const std::string& file) {
    std::ifstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + file);

    char magic[4];
    f.read(magic, 4); 
    read_u32(f); // version

    GruModel m;
    m.in_size = read_u32(f);
    m.hidden = read_u32(f);
    // Weights & Biases
    m.w_ih  = read_float_block(f);
    m.w_hh  = read_float_block(f);
    m.b_ih  = read_float_block(f);
    m.b_hh  = read_float_block(f);
    m.out_w = read_float_block(f);
    
    
    auto out_b_vec = read_float_block(f);
    if (!out_b_vec.empty()) m.out_b = out_b_vec[0];
    return m;
}

float inference(const GruModel& m, const std::vector<std::vector<float>>& sequences) {
    uint32_t H = m.hidden;
    uint32_t I = m.in_size;
    std::vector<float> h(H, 0.0f);
    std::vector<float> next(H, 0.0f);

    for (const auto& x : sequences) {
        for (uint32_t j = 0; j < H; ++j) {
            uint32_t r = j;
            uint32_t z = H + j;
            uint32_t n = 2 * H + j;
            
            // Reset gate
            float rt = sigmoid(dot_product(m.w_ih, r, I, x) + m.b_ih[r] + dot_product(m.w_hh, r, H, h) + m.b_hh[r]);
            
            // Update gate
            float zt = sigmoid(dot_product(m.w_ih, z, I, x) + m.b_ih[z] + dot_product(m.w_hh, z, H, h) + m.b_hh[z]);
            
            // New gate
            float nt = std::tanh(dot_product(m.w_ih, n, I, x) + m.b_ih[n] + rt * (dot_product(m.w_hh, n, H, h) + m.b_hh[n]));
            
            next[j] = (1.0f - zt) * nt + zt * h[j];
        }
        h.swap(next);
    }

    float y = m.out_b;
    for (uint32_t i = 0; i < H; ++i) y += m.out_w[i] * h[i];
    return y;
}

} // namespace gruexample
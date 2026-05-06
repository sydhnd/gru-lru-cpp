#include "lru_infer.h"
#include <iostream>
#include <fstream>
#include <vector>

namespace lruexample {

// Optimized: Zero-copy read directly into complex structs
LruModel loadFromBinaryToMemory(const std::string& path) {
    LruModel m;
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
        return m;
    }

    // 1. Read Header
    float header[3];
    file.read(reinterpret_cast<char*>(header), 3 * sizeof(float));
    m.feature  = static_cast<uint32_t>(header[0]);
    m.hidden   = static_cast<uint32_t>(header[1]);
    m.sequence = static_cast<uint32_t>(header[2]);
    
    std::cout << "from binary " << m.feature << "," << m.hidden << "," << m.sequence << std::endl;

    // 2. Read Lambda (Directly into std::vector<Complex>)
    m.lam.resize(m.hidden);
    file.read(reinterpret_cast<char*>(m.lam.data()), m.hidden * sizeof(Complex));

    // 3. Read B Matrix
    uint32_t b_size = m.hidden * m.feature;
    m.B.resize(b_size);
    file.read(reinterpret_cast<char*>(m.B.data()), b_size * sizeof(Complex));

    // 4. Read C Matrix
    uint32_t c_size = m.hidden;
    m.C.resize(c_size);
    file.read(reinterpret_cast<char*>(m.C.data()), c_size * sizeof(Complex));
    
    std::cout << "Model Loaded Successfully (Zero-Copy)!" << std::endl;
    return m;
}

// Legacy: Reconstructs complex numbers from grouped Real/Imag blocks
LruModel loadFromBinary(const std::string& path) {
    LruModel m;
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file.is_open()) return m;

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> buffer(size / sizeof(float));
    file.read(reinterpret_cast<char*>(buffer.data()), size);

    uint32_t current_pos = 0;
    m.feature  = static_cast<uint32_t>(buffer[current_pos++]);
    m.hidden   = static_cast<uint32_t>(buffer[current_pos++]);
    m.sequence = static_cast<uint32_t>(buffer[current_pos++]);

    std::cout << m.feature << "," << m.hidden << "," << m.sequence << std::endl;

    // --- Lambda Loading ---
    m.lam.resize(m.hidden);
    uint32_t offset_real = current_pos;
    uint32_t offset_imag = offset_real + m.hidden;
    for (uint32_t i = 0; i < m.hidden; ++i) {
        m.lam[i] = {buffer[offset_real + i], buffer[offset_imag + i]};
    }

    // --- B Matrix Loading ---
    uint32_t b_size = m.hidden * m.feature;
    m.B.resize(b_size);
    offset_real = offset_imag + m.hidden; 
    offset_imag = offset_real + b_size;
    for (uint32_t i = 0; i < b_size; ++i) {
        m.B[i] = {buffer[offset_real + i], buffer[offset_imag + i]};
    }

    // --- C Matrix Loading ---
    uint32_t c_size = m.hidden; 
    m.C.resize(c_size);
    offset_real = offset_imag + b_size;
    offset_imag = offset_real + c_size;
    for (uint32_t i = 0; i < c_size; ++i) {
        m.C[i] = {buffer[offset_real + i], buffer[offset_imag + i]};
    }
    
    std::cout << "Model Loaded Successfully (Manual Reconstruction)!" << std::endl;
    return m;
}

float inference(const LruModel& m, const std::vector<std::vector<float>>& window) {
    std::vector<Complex> state(m.hidden, {0.0f, 0.0f});

    for (const auto& row : window) {
        for (uint32_t h = 0; h < m.hidden; ++h) {
            Complex input_effect = {0.0f, 0.0f};
            for (uint32_t i = 0; i < m.feature; ++i) {
                input_effect += m.B[h * m.feature + i] * row[i];
            }
            
            // h_t = (lambda * h_{t-1}) + (B * u_t)
            state[h] = (m.lam[h] * state[h]) + input_effect;
        }
    }

    float logit = 0.0f;
    for (uint32_t h = 0; h < m.hidden; ++h) {
        logit += (m.C[h] * state[h]).real();
    }
    
    return logit;
}

} // namespace lruexample
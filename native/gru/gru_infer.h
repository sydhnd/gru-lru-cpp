#ifndef GRU_INFER_H
#define GRU_INFER_H

#include <vector>
#include <string>
#include <cstdint>

namespace gruexample {

struct GruModel {
    uint32_t in_size = 0;
    uint32_t hidden = 0;
    
    std::vector<float> w_ih;
    std::vector<float> w_hh;
    std::vector<float> b_ih;
    std::vector<float> b_hh;
    std::vector<float> out_w;
    
    float out_b = 0.0f;
};

GruModel load_binary_model(const std::string& file);
float inference(const GruModel& model, const std::vector<std::vector<float>>& sequence);
float sigmoid(float x);

} 

#endif 
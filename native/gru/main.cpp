#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <thread>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include "gru_infer.h"

float get_sensor_reading(int sensor_id) {
    return static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX / 100.0f));
}

int main(int argc, char** argv) {
    std::string model_path = "gru_model.bin";
    
    if (argc > 1) {
        model_path = argv[1];
    }

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    gruexample::GruModel model = gruexample::load_binary_model(model_path);
    
    uint32_t seq_len = 300;
    std::vector<std::vector<float>> input_seq(seq_len, std::vector<float>(model.in_size));

    while (true) {
        // Prepare input buffer
        for (uint32_t t = 0; t < seq_len; ++t) {
            for (uint32_t s = 0; s < model.in_size; ++s) {
                input_seq[t][s] = get_sensor_reading(s); 
            }
        }

        
        float logit = gruexample::inference(model, input_seq);
        float probability = gruexample::sigmoid(logit);
      
        if (probability > 0.5f) {
            std::cout << "[!] ANOMALY DETECTED: (Score: " << std::fixed << std::setprecision(4) << logit << ")" << std::endl;
        } else {
            std::cout << "[.] System Normal:    (Score: " << std::fixed << std::setprecision(4) << logit << ")" << std::endl;
        }

        // Loop delay (Frequency control)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}
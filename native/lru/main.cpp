#include <iostream>
#include <vector>
#include <fstream>
#include <complex>
#include <unistd.h>
#include <ctime>   
#include <cstdlib> 
#include "lru_infer.h"
#include <thread>
#include <chrono>

// generate random data 
float get_sensor_reading(int sensor_id) {
    return static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX / 100.0f));
}

int main(int argc, char** argv) {
    std::string model_path = "gru_model.bin";
    float threshold = -1.5f; 

    if (argc >= 2) {
        model_path = argv[1];
    }
    if (argc >= 3) {
        threshold = static_cast<float>(std::atof(argv[2]));
    }

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Load model using the loadFromBinaryToMemory  zero-copy version
    // or Load model using the reconstruction loader (loadFromBinary)
    // Also need to use the right python export format
    lruexample::LruModel lru_model = lruexample::loadFromBinaryToMemory(model_path);

    if (lru_model.sequence == 0 || lru_model.feature == 0) {
        return 1;
    }

    std::vector<std::vector<float>> sensor_window(
        lru_model.sequence, 
        std::vector<float>(lru_model.feature)
    );

    while (true) {
        for (int t = 0; t < lru_model.sequence; ++t) {
            for (int s = 0; s < lru_model.feature; ++s) {
                sensor_window[t][s] = get_sensor_reading(s); 
            }
        }

        float logit = lruexample::inference(lru_model, sensor_window);
        int is_anomaly = (logit > threshold) ? 1 : 0;

        if (is_anomaly) {
            std::cout << "[!] ANOMALY DETECTED: (Score:" << logit << ")" << std::endl;
        } else {
            std::cout << "[.] System Normal: (Score:"  << logit << ")" << std::endl;
        }

        //siumulate 1 hz data rate
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    return 0;
}

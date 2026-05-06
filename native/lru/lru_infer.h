#include <vector>
#include <complex>
#include <cstdint>

typedef std::complex<float> Complex;

namespace lruexample {
    struct LruModel {
        uint32_t feature = 0;
        uint32_t hidden = 0;
        uint32_t sequence = 0; 
        std::vector<Complex> lam; 
        std::vector<Complex> B;
        std::vector<Complex> C;
    };

    LruModel loadFromBinaryToMemory(const std::string& path);
    LruModel loadFromBinary(const std::string& path);
    float inference(const LruModel& m, const std::vector<std::vector<float>>& window);
}

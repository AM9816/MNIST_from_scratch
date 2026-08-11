#pragma once
#include <random>

template<typename T>
class RandomGen {
private:
    std::uniform_real_distribution<T> floatDist;
	std::normal_distribution<T> normDist;
    //std::uniform_int_distribution<int> intDist;
    bool normalUsed = false;
public:
    std::mt19937 engine;
    //RandomGen() { };
    RandomGen(unsigned int seed = 0) {
        engine = std::mt19937(seed);
        floatDist = std::uniform_real_distribution<T>(0, 1);
    }
    T get() {
        return floatDist(engine);
    }
	void normalSet(T mean, T dev) {
		normDist = std::normal_distribution<T>(mean, dev);
		normalUsed = true;
	}
    T normalGet() {
        if (not normalUsed)
            this->normalSet(0, 1);
        return normDist(engine);
    }
    int randint(int start, int end) {
        return std::uniform_int_distribution<int>(start, end)(engine);
    }
};
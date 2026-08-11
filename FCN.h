#pragma once
#include "datadefinitions.h"
#include "Eigen/Dense"
#include <vector>
#include <mutex>
#include "random.h"

template <typename T> using vectorList = nn::vectorList<T>;
template <typename T> void print(const T& x) { 
    std::cout << x << std::endl; }

// shortcuts
using Matrix = nn::Matrix;
using Vectorxd = nn::Vectorxd;
using rVectorxd = nn::rVectorxd;


struct FCN_Layer {
    FCN_Layer(int, int, float, nn::Activation);
    int inSize, outSize;
    float dropOut;
    nn::Activation activation;
};



struct FCN {

    FCN(
        vectorList<int>&, 
        fpoint, 
        vectorList<nn::Activation>&,
        bool);
    //~FCN();

    // core
    Matrix forward(Matrix&);
    vectorList<vectorList<Matrix>> backward(Matrix&, Matrix&, bool);
    void train(int, fpoint, fpoint, fpoint, fpoint*, int, int, bool);
    vectorList<FCN_Layer> arch; // architecture
    std::mutex paramConcurrentLock;

    // data 
    nn::DataSet<Matrix> trainData;
    nn::DataSet<Matrix> testData;
    void display_dataset(bool);
    void shuffle_dataset(int);
    vectorList<fpoint> test_against_unseen(int);
    int datapointLength = -1;
    
    // params
    void display_architecture();
    void serialize(fpoint*);
    void load_params(fpoint*);
    vectorList<Matrix> weights;
    vectorList<Matrix> bias;


};

int dll_sanity_check(int);
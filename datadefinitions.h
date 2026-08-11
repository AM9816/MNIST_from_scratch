#pragma once
#include "Eigen/Dense"
#include <vector>


namespace nn {

    // to allow changing precision easily 
    //#define fpoint double
    using fpoint = float;



    // eigen shortcuts
    using Matrix = Eigen::Matrix<fpoint, Eigen::Dynamic, Eigen::Dynamic>;
    using Vectorxd = Eigen::Matrix<fpoint, Eigen::Dynamic, 1>;
    using rVectorxd = Eigen::Matrix<fpoint, 1, Eigen::Dynamic>;

    template <typename T> using vectorList = std::vector<T>;

    template<typename T>
    struct DataSet {
        vectorList<T> Xs;
        vectorList<T> Ys;
    };

    // TODO
    //enum class LayerType {
    //    FULLY_CONNECTED_LAYER,
    //    CONVOLUTIONAL_LAYER,
    //    POOLING_LAYER,
    //};

    enum class Activation {
        RELU,
        LEAKY_RELU,
        SIGMOID,
        IDENTITY
    };


    constexpr int THREADS_TO_USE = 16;
    constexpr fpoint LEAKY_RELU_GRAD = .01;
    constexpr fpoint EPSILON = 1e-8;
    constexpr auto rSeed = 2;


    template<typename T>
    void printVector(vectorList<T>& l);
    void printShape(Matrix const& m);

}

using fpoint = nn::fpoint;

























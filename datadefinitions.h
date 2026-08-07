#pragma once
#include "Eigen/Dense"
#include <vector>
#define vectorList std::vector

// to allow changing precision easily 
#define fpoint double

using Eigen::Matrix;
using Eigen::Vector;
using Eigen::Dynamic;
//using Eigen::MatrixXd;
#define Matrixd Matrix<fpoint, -1, -1> 
#define Vectorxd Vector<fpoint, -1>
#define rVectorxd Eigen::RowVectorXd


template<typename T>
struct DataSet {
    vectorList<T> Xs;
    vectorList<T> Ys;
};


#define CONVOLUTIONAL_LAYER     1
#define FULLY_CONNECTED_LAYER   0
#define POOLING_LAYER           2

#define RELU        0
#define LEAKY_RELU  1
#define SIGMOID     2























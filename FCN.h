#pragma once
#include "datadefinitions.h"
#include "Eigen/Dense"
#include <vector>
#include <mutex>
#include "random.h"
#define vectorList std::vector

#define fpoint double

using Eigen::Matrix;
using Eigen::Vector;
using Eigen::Dynamic;
//using Eigen::MatrixXd;
#define Matrixd Matrix<fpoint, -1, -1> 




struct FCN_Layer {
    FCN_Layer(int, int, float, int);
    int inSize, outSize;
    float dropOut;
    int activation;

};



struct FCN {

    FCN(vectorList<int>&, fpoint, vectorList<int>&,
        fpoint, fpoint, bool);
    ~FCN();


    void display();
    void load_data(int*, int, bool, bool);
    void display_dataset(bool);
    Matrixd forward(Matrixd&);
    vectorList<vectorList<Matrixd>> backward(Matrixd&, Matrixd&);
    void train(int, fpoint, fpoint, fpoint, fpoint*, int, int);
    vectorList<fpoint> test_against_unseen(int);
    void clear_grad();
    void shuffle_dataset(int);

    void serialize(fpoint*);
    void load_params(fpoint*);

    DataSet<Matrixd> trainData;
    DataSet<Matrixd> testData;

    int datapointLength = -1;
    bool pause_parameter_edit = false;

    std::mutex paramMutex;

    vectorList<FCN_Layer> arch;

    vectorList<
        Matrix<fpoint, Dynamic, Dynamic>
    >   weights;

    vectorList<
        Matrix<fpoint, Dynamic, Dynamic>
    >   bias;

    // required for interacting with python
    FCN* self = nullptr;


};

int dll_sanity_check(int);
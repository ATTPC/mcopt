#include "mcopt.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <armadillo>
#include <string>
#include <H5Cpp.h>
#include <cmath>
#include <numeric>

std::vector<double> readEloss(const std::string& path)
{
    H5::H5File file (path.c_str(), H5F_ACC_RDONLY);
    H5::DataSet ds = file.openDataSet("eloss");

    H5::DataSpace filespace = ds.getSpace();
    int ndims = filespace.getSimpleExtentNdims();
    assert(ndims == 1);
    hsize_t dim;
    filespace.getSimpleExtentDims(&dim);

    H5::DataSpace memspace (ndims, &dim);

    std::vector<double> res (dim);

    ds.read(res.data(), H5::PredType::NATIVE_DOUBLE, memspace, filespace);

    filespace.close();
    memspace.close();
    ds.close();
    file.close();
    return res;
}

int main(const int argc, const char** argv)
{
    if (argc < 2) {
        std::cerr << "Provide a path to the eloss.h5 file as the first argument." << std::endl;
        return 1;
    }
    std::string elossPath (argv[1]);

    auto eloss = readEloss(elossPath);
    unsigned massNum = 1;
    unsigned chargeNum = 1;

    arma::vec efield {0., -940.756, 8950.697};
    arma::vec bfield {0, 0, 1.75};

    mcopt::Tracker tracker (massNum, chargeNum, eloss, efield, bfield);

    double x0 = 0;
    double y0 = 0.005;
    double z0 = 0.71;
    double enu0 = 2.68;
    double azi0 = 178 * M_PI / 180;
    double pol0 = (180 - 64.6) * M_PI / 180;

    std::vector<std::chrono::high_resolution_clock::duration> durations;

    size_t numIters = 20;

    for (size_t i = 0; i < numIters; i++) {
        auto begin = std::chrono::high_resolution_clock::now();
        mcopt::Track tr = tracker.trackParticle(x0, y0, z0, enu0, azi0, pol0);
        auto end = std::chrono::high_resolution_clock::now();

        durations.push_back(end - begin);
    }

    auto totalDuration =
        std::accumulate(durations.begin(), durations.end(), std::chrono::high_resolution_clock::duration::zero());

    double meanTime = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(totalDuration).count()) / numIters;


    std::cout << "Mean tracking time: " << meanTime << " µs" << std::endl;

    return 0;
}

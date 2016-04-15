#include <chrono>
#include <iostream>
#include <iomanip>
#include <armadillo>
#include <string>
#include <cmath>
#include <numeric>

#include "mcopt.h"
#include "utils.h"

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

    size_t numIters = 50000;

    for (size_t i = 0; i < numIters; i++) {
        auto begin = std::chrono::high_resolution_clock::now();
        mcopt::Track tr = tracker.trackParticle(x0, y0, z0, enu0, azi0, pol0);
        auto end = std::chrono::high_resolution_clock::now();

        durations.push_back(end - begin);
    }

    std::cout << "Ran " << durations.size() << " times" << std::endl;

    auto minDuration =
        std::min_element(durations.begin(), durations.end());

    double meanTime = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(*minDuration).count());


    std::cout << "Min tracking time: " << meanTime << " µs" << std::endl;

    return 0;
}

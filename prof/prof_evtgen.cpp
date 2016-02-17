#include <chrono>
#include <iostream>
#include <iomanip>
#include <armadillo>
#include <string>
#include <cmath>
#include <numeric>
#include <map>

#include "mcopt.h"
#include "utils.h"

int main(const int argc, const char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: prof_evtgen ELOSS_PATH LUT_PATH" << std::endl;
        return 1;
    }
    std::string elossPath (argv[1]);
    std::string lutPath (argv[2]);

    auto eloss = readEloss(elossPath);
    arma::Mat<uint16_t> lut = readLUT(lutPath);

    unsigned massNum = 1;
    unsigned chargeNum = 1;
    double clock = 12.5e6;
    double shape = 280e-9;
    double ioniz = 10;
    double umegasGain = 1000;
    double tilt = 6 * M_PI / 180;
    arma::vec beamCtr {0, 0, 0};

    arma::vec efield {0., -940.756, 8950.697};
    arma::vec bfield {0, 0, 1.75};
    arma::vec vd {-0.05307544, -0.53536606, -5.14373076};

    mcopt::Tracker tracker (massNum, chargeNum, eloss, efield, bfield);
    mcopt::PadPlane pads (lut, -0.280, 0.0001, -0.280, 0.0001, -108 * M_PI / 180);
    mcopt::EventGenerator evtgen(pads, vd, clock, shape, massNum, ioniz, umegasGain, tilt, beamCtr);

    double x0 = 0;
    double y0 = 0.005;
    double z0 = 0.71;
    double enu0 = 2.68;
    double azi0 = 178 * M_PI / 180;
    double pol0 = (180 - 64.6) * M_PI / 180;

    mcopt::Track tr = tracker.trackParticle(x0, y0, z0, enu0, azi0, pol0);
    arma::mat trPos = tr.getPositionMatrix();
    arma::mat trEn = tr.getEnergyVector();

    std::vector<std::chrono::high_resolution_clock::duration> durations;

    size_t numIters = 100;

    for (size_t i = 0; i < numIters; i++) {
        auto begin = std::chrono::high_resolution_clock::now();
        std::map<mcopt::pad_t, arma::vec> evt = evtgen.makeEvent(trPos, trEn);
        auto end = std::chrono::high_resolution_clock::now();

        durations.push_back(end - begin);
    }

    auto totalDuration =
        std::accumulate(durations.begin(), durations.end(), std::chrono::high_resolution_clock::duration::zero());

    double meanTime = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(totalDuration).count()) / numIters;

    std::cout << "Mean makeEvent time: " << meanTime << " µs" << std::endl;

    return 0;
}

#include <iostream>
#include <fstream>
#include "arma_include.h"
#include "mcopt.h"
#include <chrono>

int main(const int argc, const char** argv)
{
    if (argc < 2) {
        std::cerr << "Need more arguments" << std::endl;
        abort();
    }

    std::ifstream elossFile(argv[1]);
    if (!elossFile.good()) {
        std::cerr << "File was bad" << std::endl;
        abort();
    }

    std::vector<double> eloss;
    while (elossFile.good()) {
        double v;
        elossFile >> v;
        eloss.push_back(v);
    }

    elossFile.close();

    std::cout << "Got " << eloss.size() << " eloss values" << std::endl;

    double x0 = 0.;
    double y0 = 0.;
    double z0 = 0.49;
    double enu0 = 1.36;
    double azi0 = -3.110857;
    double pol0 = 1.951943;
    int massNum = 1;
    int chargeNum = 1;

    arma::vec::fixed<3> efield = {0., 0., 9.0e3};
    arma::vec::fixed<3> bfield = {0., 0., 2.0};

    Conditions cond;
    cond.massNum = massNum;
    cond.chargeNum = chargeNum;
    cond.eloss = eloss;
    cond.efield = efield;
    cond.bfield = bfield;

    Track tr;
    auto begin = std::chrono::high_resolution_clock::now();
    int N = 1000;
    for (int i = 0; i < N; i++) {
        tr = trackParticle(x0, y0, z0, enu0, azi0, pol0, cond);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::microseconds>(end-begin);
    std::cout << "Took " << delta.count() << " usec per track" << std::endl;

    arma::vec ctr0 = {x0, y0, z0, enu0, azi0, pol0, 2.0};
    arma::vec sig0 = {0., 0., 0.01, 0.1, 0.1, 0.1, 0.1};
    arma::mat trueValues (500, 3, arma::fill::zeros);
    trueValues.col(2) = arma::linspace(0, z0, 500);

    begin = std::chrono::high_resolution_clock::now();
    MCminimizeResult ctr = MCminimize(ctr0, sig0, trueValues, cond, 10, 200, 0.8);
    end = std::chrono::high_resolution_clock::now();
    auto delta2 = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);

    std::cout << "Fit took " << delta2.count() << " msec" << std::endl;

    // std::cout << std::get<1>(ctr) << std::endl;

    return 0;
}

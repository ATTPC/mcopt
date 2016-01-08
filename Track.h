#ifndef MCOPT_TRACK_H
#define MCOPT_TRACK_H

#include <vector>
#include <armadillo>
#include <cassert>

namespace mcopt {

    class Track
    {
    public:
        void append(const double x, const double y, const double z, const double time,
                    const double enu, const double azi, const double pol);

        arma::mat getMatrix() const;
        arma::vec getEnergyVector() const { return arma::vec(enu); }
        size_t numPts() const;
        void unTiltAndRecenter(const arma::vec beamCtr, const double tilt);

    private:
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> z;
        std::vector<double> time;
        std::vector<double> enu;
        std::vector<double> azi;
        std::vector<double> pol;
    };

}

#endif /* end of include guard: MCOPT_TRACK_H */

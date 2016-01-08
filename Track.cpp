#include "Track.h"

namespace mcopt
{
    void Track::append(const double xi, const double yi, const double zi, const double timei,
                       const double enui, const double azii, const double poli)
    {
        x.push_back(xi);
        y.push_back(yi);
        z.push_back(zi);
        time.push_back(timei);
        enu.push_back(enui);
        azi.push_back(azii);
        pol.push_back(poli);
    }

    arma::mat Track::getMatrix() const
    {
        arma::mat result (numPts(), 7);
        result.col(0) = arma::vec(x);
        result.col(1) = arma::vec(y);
        result.col(2) = arma::vec(z);
        result.col(3) = arma::vec(time);
        result.col(4) = arma::vec(enu);
        result.col(5) = arma::vec(azi);
        result.col(6) = arma::vec(pol);

        return result;
    }

    void Track::unTiltAndRecenter(const arma::vec beamCtr, const double tilt)
    {
        arma::mat data (numPts(), 3);
        data.col(0) = arma::vec(x) + beamCtr(0);
        data.col(1) = arma::vec(y) + beamCtr(1);
        data.col(2) = arma::vec(z);

        arma::mat tmat {{1, 0, 0},
                        {0, cos(-tilt), -sin(-tilt)},
                        {0, sin(-tilt), cos(-tilt)}};

        data = (tmat * data.t()).t();
        data.col(2) += beamCtr(2);

        x = arma::conv_to<std::vector<double>>::from(data.col(0));
        y = arma::conv_to<std::vector<double>>::from(data.col(1));
        z = arma::conv_to<std::vector<double>>::from(data.col(2));
        return;
    }

    size_t Track::numPts() const
    {
        size_t N = x.size();

        assert(y.size() == N);
        assert(z.size() == N);
        assert(time.size() == N);
        assert(enu.size() == N);
        assert(azi.size() == N);
        assert(pol.size() == N);

        return N;
    }
}

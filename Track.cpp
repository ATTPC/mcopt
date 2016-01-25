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

    arma::mat Track::getPositionMatrix() const
    {
        arma::mat result (numPts(), 3);
        result.col(0) = arma::vec(x);
        result.col(1) = arma::vec(y);
        result.col(2) = arma::vec(z);

        return result;
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

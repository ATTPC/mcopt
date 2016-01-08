#include "EventGen.h"

namespace mcopt
{
    arma::mat calibrate(const Track& tr, const arma::vec& vd, const double clock)
    {
        arma::mat trMat = tr.getMatrix();
        arma::mat pos = trMat.cols(0, 2);
        arma::mat result = pos + pos.col(2) * -vd.t() / clock * 10;
        result.col(2) -= pos.col(2);

        return result;
    }

    arma::mat uncalibrate(const Track& tr, const arma::vec& vd, const double clock, const int offset)
    {
        arma::mat trMat = tr.getMatrix();
        arma::mat pos = trMat.cols(0, 2);

        arma::vec tbs = pos.col(2) * clock / (10 * -vd(2)) + offset;

        arma::mat result = pos - tbs * -vd.t() / clock * 10;
        result.col(2) = tbs;

        return result;
    }

    std::map<uint16_t, unsigned long> makePeaksFromSimulation(const PadPlane& pads, const Track& tr, const arma::vec& vd,
                                                              const double clock, const int massNum, const double ioniz)
    {
        arma::mat uncal = uncalibrate(tr, vd, clock);
        arma::vec en = tr.getEnergyVector() * 1e6 * massNum;
        assert(en.n_rows == uncal.n_rows);
        arma::vec ne = arma::floor(-arma::diff(en) / ioniz);

        std::map<uint16_t, unsigned long> result;

        for (arma::uword i = 0; i < uncal.n_rows - 1; i++) {
            uint16_t pad = pads.getPadNumberFromCoordinates(uncal(i, 0), uncal(i, 1));
            if (pad != 20000) {
                result.emplace(pad, ne(i));
            }
        }

        return result;
    }
}

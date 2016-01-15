#include "EventGen.h"

static const double E_CHG = 1.602176565e-19;

namespace mcopt
{
    arma::mat calibrate(const Track& tr, const arma::vec& vd, const double clock)
    {
        arma::mat trMat = tr.getMatrix();
        arma::mat pos = trMat.cols(0, 2);
        arma::mat result = pos + pos.col(2) * -vd.t() / (clock * 100);
        result.col(2) -= pos.col(2);

        return result;
    }

    arma::mat uncalibrate(const Track& tr, const arma::vec& vd, const double clock, const int offset)
    {
        arma::mat trMat = tr.getMatrix();
        arma::mat pos = trMat.cols(0, 2);

        arma::vec tbs = pos.col(2) * clock * 100 / (-vd(2)) + offset;

        arma::mat result = pos - tbs * -vd.t() / (clock * 100);
        result.col(2) = tbs;

        return result;
    }

    arma::vec squareWave(const arma::uword size, const arma::uword leftEdge,
                         const arma::uword width, const double height)
    {
        arma::vec res (size, arma::fill::zeros);
        for (arma::uword i = leftEdge; i < leftEdge + width && i < size; i++) {
            res(i) = height;
        }
        return res;
    }

    std::map<uint16_t, Peak> makePeaksFromSimulation(const PadPlane& pads, const Track& tr, const arma::vec& vd,
                                                     const double clock, const int massNum, const double ioniz,
                                                     const unsigned gain)
    {
        arma::mat uncal = uncalibrate(tr, vd, clock);
        arma::vec en = tr.getEnergyVector() * 1e6 * massNum;
        assert(en.n_rows == uncal.n_rows);
        arma::uvec ne = arma::conv_to<arma::uvec>::from(arma::floor(-arma::diff(en) / ioniz)) * gain;
        arma::Col<unsigned> tbs = arma::conv_to<arma::Col<unsigned>>::from(arma::floor(uncal.col(2)));

        std::map<uint16_t, Peak> result;

        for (arma::uword i = 0; i < uncal.n_rows - 1; i++) {
            uint16_t pad = pads.getPadNumberFromCoordinates(uncal(i, 0), uncal(i, 1));
            if (pad != 20000) {
                auto& pk = result[pad];
                arma::uword ampl = ne(i);
                if (pk.amplitude < ampl) {
                    pk.timeBucket = tbs(i);
                    pk.amplitude = ampl;
                }
            }
        }

        return result;
    }

    Trigger::Trigger(const unsigned int padThreshMSB, const unsigned int padThreshLSB, const double trigWidth,
            const unsigned long multThresh, const unsigned long multWindow, const double writeCk,
            const double gain, const double discrFrac, const PadMap& padmap)
        : multThresh(multThresh), writeCk(writeCk), padmap(padmap)
    {
        double pt = ((padThreshMSB << 4) + padThreshLSB);
        double discrMax = discrFrac * 4096;  // in data ADC bins
        double elecPerBin = gain / 4096 / E_CHG;
        padThresh = (pt / 128) * discrMax * elecPerBin;

        this->trigWidth = std::lround(trigWidth * writeCk);
        this->multWindow = std::lround(multWindow / masterCk * writeCk);
    }

    arma::mat Trigger::findTriggerSignals(const std::map<pad_t, Peak>& peaks) const
    {
        arma::mat res (10, 512, arma::fill::zeros);
        for (const auto& pair : peaks) {
            const auto& pad = pair.first;
            const auto& peak = pair.second;

            if (peak.amplitude < padThresh) continue;

            auto cobo = padmap.reverseFind(pad).cobo;
            assert(cobo != padmap.missingValue);

            arma::vec sig = squareWave(512, peak.timeBucket, trigWidth, trigHeight);
            res.row(cobo) += sig.t();
        }
        return res;
    }

    arma::mat Trigger::applyMultiplicityWindow(const arma::mat& trigs) const
    {
        arma::mat res (arma::size(trigs), arma::fill::zeros);
        for (unsigned long j = 0; j < trigs.n_cols; j++) {
            unsigned long min = j < multWindow ? 0 : j - multWindow;
            unsigned long max = j;
            res.col(j) = arma::sum(trigs.cols(min, max), 1);
        }
        return res;
    }

    bool Trigger::didTrigger(const std::map<pad_t, Peak>& peaks) const
    {
        arma::mat sigs = applyMultiplicityWindow(findTriggerSignals(peaks));
        arma::vec maxes = arma::max(sigs, 1);
        return arma::any(maxes > multThresh);
    }
}

#include "EventGen.h"

static const double E_CHG = 1.602176565e-19;

namespace mcopt
{
    arma::mat calibrate(const Track& tr, const arma::vec& vd, const double clock)
    {
        // Assume tr has units of meters, vd in cm/us, clock in Hz.
        arma::mat trMat = tr.getMatrix();
        arma::mat pos = trMat.cols(0, 2);
        arma::mat result = pos + pos.col(2) * -vd.t() / (clock * 1e-4);
        result.col(2) -= pos.col(2);

        return result;
    }

    arma::mat uncalibrate(const Track& tr, const arma::vec& vd, const double clock, const int offset)
    {
        // Assume tr has units of meters, vd in cm/us, clock in Hz.
        arma::mat trMat = tr.getMatrix();
        arma::mat pos = trMat.cols(0, 2);

        arma::vec tbs = pos.col(2) * clock * 1e-4 / (-vd(2)) + offset;

        arma::mat result = pos - tbs * -vd.t() / (clock * 1e-4);
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

    arma::vec elecPulse(const double amplitude, const double shape, const double clock, const arma::uword offset)
    {
        arma::vec res (512, arma::fill::zeros);

        double s = shape * clock;  // IMPORTANT: shape and clock must have compatible units, e.g. MHz and us.

        for (arma::uword i = offset; i < res.n_rows; i++) {
            double t = (i - offset) / s;
            res(i) = amplitude * std::exp(-3*t) * std::sin(t) * std::pow(t, 3);
        }
        return res;
    }

    std::map<pad_t, arma::vec> EventGenerator::makeEvent(const Track& tr) const
    {
        arma::mat uncal = uncalibrate(tr, vd, clock);
        arma::vec en = tr.getEnergyVector() * 1e6 * massNum;
        assert(en.n_rows == uncal.n_rows);
        arma::uvec ne = arma::conv_to<arma::uvec>::from(arma::floor(-arma::diff(en) / ioniz)) * gain;
        arma::Col<unsigned> tbs = arma::conv_to<arma::Col<unsigned>>::from(arma::floor(uncal.col(2)));

        std::map<pad_t, arma::vec> result;

        for (arma::uword i = 0; i < uncal.n_rows - 1; i++) {
            pad_t pad = pads.getPadNumberFromCoordinates(uncal(i, 0), uncal(i, 1));
            if (pad != 20000) {
                arma::vec& padSignal = result[pad];
                if (padSignal.is_empty()) {
                    // This means that the signal was just default-constructed by std::map::operator[]
                    padSignal.zeros(512);
                }
                unsigned offset = tbs(i);

                // Use a precalculated pulse shape that just needs to be scaled and shifted. This is much faster.
                arma::vec pulse (512, arma::fill::zeros);
                pulse(arma::span(offset, 511)) = gain * ne(i) * pulseTemplate(arma::span(0, 511-offset));
                padSignal += pulse;
            }
        }

        return result;
    }

    std::map<pad_t, Peak> EventGenerator::makePeaksFromSimulation(const Track& tr) const
    {
        std::map<pad_t, arma::vec> evt = makeEvent(tr);

        std::map<pad_t, Peak> res;
        for (const auto& pair : evt) {
            arma::uword maxTB;
            unsigned maxVal = std::floor(pair.second.max(maxTB));  // This stores the location of the max in its argument
            res.emplace(pair.first, Peak{static_cast<unsigned>(maxTB), maxVal});
        }
        return res;
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

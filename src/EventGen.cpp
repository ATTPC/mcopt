#include "EventGen.h"

static const double E_CHG = 1.602176565e-19;

namespace mcopt
{
    arma::mat calibrate(const Track& tr, const arma::vec& vd, const double clock)
    {
        arma::mat pos = tr.getPositionMatrix();
        return calibrate(pos, vd, clock);
    }

    arma::mat calibrate(const arma::mat& pos, const arma::vec& vd, const double clock)
    {
        // Assume pos has units of meters, vd in cm/us, clock in Hz.
        arma::mat result = pos + pos.col(2) * -vd.t() / (clock * 1e-4);
        result.col(2) -= pos.col(2);

        return result;
    }

    arma::mat uncalibrate(const Track& tr, const arma::vec& vd, const double clock, const int offset)
    {
        arma::mat pos = tr.getPositionMatrix();
        return uncalibrate(pos, vd, clock, offset);
    }

    arma::mat uncalibrate(const arma::mat& pos, const arma::vec& vd, const double clock, const int offset)
    {
        // Assume tr has units of meters, vd in cm/us, clock in Hz.
        arma::vec tbs = pos.col(2) * clock * 1e-4 / (-vd(2)) + offset;

        arma::mat result = pos - tbs * -vd.t() / (clock * 1e-4);
        result.col(2) = tbs;

        return result;
    }

    arma::mat unTiltAndRecenter(const arma::mat& pos, const arma::vec& beamCtr, const double tilt)
    {
        arma::mat res (arma::size(pos));
        res.col(0) = pos.col(0) + beamCtr(0);
        res.col(1) = pos.col(1) + beamCtr(1);
        res.col(2) = pos.col(2);

        arma::mat tmat {{1, 0, 0},
                        {0, cos(-tilt), -sin(-tilt)},
                        {0, sin(-tilt), cos(-tilt)}};

        res = (tmat * res.t()).t();
        res.col(2) += beamCtr(2);
        return res;
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

    static inline double approxSin(const double t)
    {
        return t - t*t*t / 6 + t*t*t*t*t / 120 - t*t*t*t*t*t*t / 5040;
    }

    arma::vec elecPulse(const double amplitude, const double shape, const double clock, const double offset)
    {
        arma::vec res (512, arma::fill::zeros);

        double s = shape * clock;  // IMPORTANT: shape and clock must have compatible units, e.g. MHz and us.

        arma::uword firstPt = static_cast<arma::uword>(std::ceil(offset));
        for (arma::uword i = firstPt; i < res.n_rows; i++) {
            double t = (i - offset) / s;
            res(i) = amplitude * std::exp(-3*t) * approxSin(t) * t*t*t / 0.044;
        }
        return res;
    }

    std::map<pad_t, arma::vec> EventGenerator::makeEvent(const Track& tr) const
    {
        arma::mat trMat = tr.getMatrix();
        arma::mat pos = trMat.cols(0, 2);
        arma::vec en = trMat.col(4);
        return makeEvent(pos, en);
    }

    std::map<pad_t, arma::vec> EventGenerator::makeEvent(const arma::mat& pos, const arma::vec& en) const
    {
        arma::mat posTilted = unTiltAndRecenter(pos, beamCtr, tilt);
        arma::mat uncal = uncalibrate(posTilted, vd, clock);
        arma::uvec ne = arma::conv_to<arma::uvec>::from(arma::floor(-arma::diff(en * 1e6 * massNum) / ioniz));
        arma::vec tbs = uncal.col(2);

        std::map<pad_t, arma::vec> result;

        for (arma::uword i = 0; i < uncal.n_rows - 1; i++) {
            pad_t pad = pads.getPadNumberFromCoordinates(uncal(i, 0), uncal(i, 1));
            if (pad != 20000) {
                arma::vec& padSignal = result[pad];
                if (padSignal.is_empty()) {
                    // This means that the signal was just default-constructed by std::map::operator[]
                    padSignal.zeros(512);
                }
                double offset = tbs(i);
                // if (offset > 511) throw TBOverflow(std::to_string(offset));
                if (offset > 511) continue;

                arma::vec pulse = elecPulse(gain * ne(i), shape, clock, offset);
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

    arma::mat EventGenerator::makePeaksTableFromSimulation(const Track& tr) const
    {
        arma::mat pos = tr.getPositionMatrix();
        arma::vec en = tr.getEnergyVector();
        return makePeaksTableFromSimulation(pos, en);
    }

    arma::mat EventGenerator::makePeaksTableFromSimulation(const arma::mat& pos, const arma::vec& en) const
    {
        std::map<pad_t, arma::vec> evt = makeEvent(pos, en);

        const double offset = shape * clock;  // Shaping time shifts the peak toward higher TBs

        std::vector<arma::rowvec> rows;
        for (const auto& pair : evt) {
            const auto& padNum = pair.first;
            const auto& sig = pair.second;

            // Find center of gravity of the peak
            arma::uvec pkPts = arma::find(sig > 0.3*sig.max());
            arma::vec pkVals = sig.elem(pkPts);
            double pkCtrGrav = arma::dot(pkPts, pkVals) / arma::sum(pkVals);

            double maxVal = sig.max();
            auto xy = pads.getPadCenter(padNum);
            rows.push_back(arma::rowvec{xy.at(0), xy.at(1), (pkCtrGrav - offset), maxVal,
                                        static_cast<double>(padNum)});
        }

        arma::mat result (rows.size(), 5);
        for (size_t i = 0; i < rows.size(); i++) {
            result.row(i) = rows.at(i);
        }

        return result;
    }

    arma::vec EventGenerator::makeMeshSignal(const Track& tr) const
    {
        const arma::mat pos = tr.getPositionMatrix();
        const arma::vec en = tr.getEnergyVector();
        return makeMeshSignal(pos, en);
    }

    arma::vec EventGenerator::makeMeshSignal(const arma::mat& pos, const arma::vec& en) const
    {
        std::map<pad_t, arma::vec> evt = makeEvent(pos, en);
        arma::vec res (512, arma::fill::zeros);

        for (const auto& pair : evt) {
            res += pair.second;
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

    arma::mat Trigger::findTriggerSignals(const std::map<pad_t, arma::vec>& event) const
    {
        arma::mat res (numCobos, numTBs, arma::fill::zeros);
        for (const auto& pair : event) {
            const auto& padNum = pair.first;
            const auto& padSig = pair.second;

            arma::vec sig = arma::vec (numTBs, arma::fill::zeros);

            for (arma::uword tb = 0; tb < numTBs; ) {
                if (padSig(tb) > padThresh) {
                    for (arma::uword sqIdx = tb; sqIdx < std::min(tb + trigWidth, numTBs); sqIdx++) {
                        sig(sqIdx) += trigHeight;
                    }
                    tb += trigWidth;
                }
                else {
                    tb += 1;
                }
            }

            if (arma::any(sig)) {
                auto cobo = padmap.reverseFind(padNum).cobo;
                assert(cobo != padmap.missingValue);

                res.row(cobo) += sig.t();
            }
        }
        return res;
    }

    arma::mat Trigger::applyMultiplicityWindow(const arma::mat& trigs) const
    {
        arma::mat res (arma::size(trigs), arma::fill::zeros);
        for (unsigned long j = 0; j < trigs.n_cols; j++) {
            unsigned long min = j < multWindow ? 0 : j - multWindow;
            unsigned long max = j;
            res.col(j) = arma::sum(trigs.cols(min, max), 1) * readCk / writeCk;  // Mult is read on readCk, so rescale this signal
        }
        return res;
    }

    bool Trigger::didTrigger(const std::map<pad_t, arma::vec>& event) const
    {
        arma::mat sigs = applyMultiplicityWindow(findTriggerSignals(event));
        arma::vec maxes = arma::max(sigs, 1);
        return arma::any(maxes > multThresh);
    }
}

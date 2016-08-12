#include "Trigger.h"

namespace mcopt {
    Trigger::Trigger(const unsigned int padThreshMSB_, const unsigned int padThreshLSB_, const double trigWidth_,
            const unsigned long multThresh_, const unsigned long multWindow_, const double writeCk_,
            const double gain_, const double discrFrac_, const PadMap& padmap_)
        : multThresh(multThresh_), writeCk(writeCk_), padmap(padmap_)
    {
        double pt = ((padThreshMSB_ << 4) + padThreshLSB_);
        double discrMax = discrFrac_ * 4096;  // in data ADC bins
        double elecPerBin = gain_ / 4096 / Constants::E_CHG;
        padThresh = (pt / 128) * discrMax * elecPerBin;

        trigWidth = static_cast<decltype(trigWidth)>(std::lround(trigWidth_ * writeCk));
        multWindow = static_cast<decltype(multWindow)>(std::lround(multWindow_ / masterCk * writeCk));
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

                res.row(static_cast<arma::uword>(cobo)) += sig.t();
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

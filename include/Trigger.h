#ifndef MCOPT_TRIGGER_H
#define MCOPT_TRIGGER_H

#include <armadillo>
#include <map>
#include <cassert>
#include "Constants.h"
#include "FitConfig.h"
#include "PadMap.h"

namespace mcopt {
    class Trigger
    {
    public:
        Trigger(const unsigned int padThreshMSB, const unsigned int padThreshLSB, const double trigWidth,
                const unsigned long multThresh, const unsigned long multWindow, const double writeCk,
                const double gain, const double discrFrac, const PadMap& padmap);

        arma::mat findTriggerSignals(const std::map<pad_t, arma::vec>& peaks) const;
        arma::mat applyMultiplicityWindow(const arma::mat& trigs) const;
        bool didTrigger(const std::map<pad_t, arma::vec>& peaks) const;

        double getPadThresh() const { return padThresh; }
        unsigned long getTrigWidth() const { return trigWidth; }
        unsigned long getMultWindow() const { return multWindow; }

    private:
        double padThresh;
        unsigned long trigWidth;
        double trigHeight = 48;  // ADC bins
        unsigned long multThresh;
        unsigned long multWindow;
        double writeCk;
        double readCk = 25e6;
        double masterCk = 100e6;
        const PadMap padmap;
        const arma::uword numCobos = 10;
        const arma::uword numTBs = 512;
    };
}

#endif /* end of include guard: MCOPT_TRIGGER_H */

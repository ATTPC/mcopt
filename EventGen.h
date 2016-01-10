#ifndef MCOPT_EVENTGEN_H
#define MCOPT_EVENTGEN_H

#include <armadillo>
#include <map>
#include <algorithm>
#include <vector>
#include <cassert>
#include <cmath>
#include "Track.h"
#include "PadPlane.h"
#include "PadMap.h"

namespace mcopt
{
    arma::mat calibrate(const Track& tr, const arma::vec& vd, const double clock);
    arma::mat uncalibrate(const Track& tr, const arma::vec& vd, const double clock, const int offset=0);
    arma::vec squareWave(const arma::uword size, const arma::uword leftEdge,
                         const arma::uword width, const double height);

    struct Peak
    {
        unsigned timeBucket;
        unsigned long amplitude;
    };

    std::map<uint16_t, Peak> makePeaksFromSimulation(const PadPlane& pads, const Track& tr, const arma::vec& vd,
                                                     const double clock, const int massNum, const double ioniz);

    class Trigger
    {
    public:
        Trigger(const unsigned int padThreshMSB, const unsigned int padThreshLSB, const double trigWidth,
                const unsigned long multThresh, const unsigned long multWindow, const double writeCk,
                const double gain, const double discrFrac, const PadMap& padmap);

        arma::mat findTriggerSignals(const std::map<pad_t, Peak>& peaks) const;
        arma::mat applyMultiplicityWindow(const arma::mat& trigs) const;
        bool didTrigger(const std::map<pad_t, Peak>& peaks) const;

        double getPadThresh() const { return padThresh; }
        unsigned long getMultWindow() const { return multWindow; }

    private:
        double padThresh;
        unsigned long trigWidth;
        double trigHeight = 48;  // ADC bins
        unsigned long multThresh;
        unsigned long multWindow;
        double writeCk;
        double masterCk = 100e6;
        const PadMap padmap;
    };
}

#endif /* end of include guard: MCOPT_EVENTGEN_H */

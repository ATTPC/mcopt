#ifndef MCOPT_EVENTGEN_H
#define MCOPT_EVENTGEN_H

#include <armadillo>
#include <map>
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
        Trigger(const unsigned int padThreshMSB, const unsigned int padThreshLSB, const unsigned int trigWidth,
                const unsigned long multThresh, const unsigned long multWindow, const double writeCk,
                const double gain, const double discrFrac, const PadMap& padmap);

        std::vector<arma::vec> findTriggerSignals(const std::map<uint16_t, Peak>& peaks);
        arma::vec findMultiplicitySignals(const std::map<uint16_t, Peak>& peaks);

    private:
        double padThresh;
        unsigned long trigWidth;
        double trigHeight = 48;  // ADC bins
        unsigned long multThresh;
        unsigned long multWindow;
        double writeCk;
        double masterCk = 100;
        const PadMap padmap;
    };
}

#endif /* end of include guard: MCOPT_EVENTGEN_H */

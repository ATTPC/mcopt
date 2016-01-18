#ifndef MCOPT_EVENTGEN_H
#define MCOPT_EVENTGEN_H

#include <armadillo>
#include <cmath>
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
    arma::vec elecPulse(const double amplitude, const double shape, const double clock, const arma::uword offset);

    struct Peak
    {
        unsigned timeBucket;
        unsigned long amplitude;
    };

    class EventGenerator
    {
    public:
        EventGenerator(const PadPlane& pads, const arma::vec& vd, const double clock, const double shape,
                       const int massNum, const double ioniz, const unsigned gain=1)
            : pads(pads), vd(vd), clock(clock), shape(shape), massNum(massNum), ioniz(ioniz), gain(gain) {}

        std::map<pad_t, arma::vec> makeEvent(const Track& tr) const;
        std::map<uint16_t, Peak> makePeaksFromSimulation(const Track& tr) const;

    private:
        const PadPlane pads;
        const arma::vec vd;
        const double clock;
        const double shape;
        const int massNum;
        const double ioniz;
        const unsigned gain;
    };

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

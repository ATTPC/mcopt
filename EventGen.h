#ifndef MCOPT_EVENTGEN_H
#define MCOPT_EVENTGEN_H

#include <armadillo>
#include <cmath>
#include <map>
#include <vector>
#include <cassert>
#include <algorithm>
#include "Track.h"
#include "PadPlane.h"
#include "PadMap.h"

namespace mcopt
{
    arma::mat calibrate(const Track& tr, const arma::vec& vd, const double clock);
    arma::mat calibrate(const arma::mat& pos, const arma::vec& vd, const double clock);
    arma::mat uncalibrate(const Track& tr, const arma::vec& vd, const double clock, const int offset=0);
    arma::mat uncalibrate(const arma::mat& pos, const arma::vec& vd, const double clock, const int offset=0);
    arma::mat calibrateWithTilt(const Track& tr, const arma::vec& vd, const double clock, const double tilt);
    arma::mat calibrateWithTilt(const arma::mat& pos, const arma::vec& vd, const double clock, const double tilt);
    arma::mat uncalibrateWithTilt(const Track& tr, const arma::vec& vd, const double clock, const int offset=0);
    arma::mat uncalibrateWithTilt(const arma::mat& pos, const arma::vec& vd, const double clock, const double tilt, const int offset=0);

    arma::mat unTiltAndRecenter(const arma::mat& pos, const arma::vec& beamCtr, const double tilt);

    arma::vec squareWave(const arma::uword size, const arma::uword leftEdge,
                         const arma::uword width, const double height);
    arma::vec elecPulse(const double amplitude, const double shape, const double clock, const arma::uword offset);

    struct Peak
    {
        unsigned timeBucket;
        unsigned long amplitude;
    };

    class TBOverflow : public std::exception
    {
    public:
        TBOverflow(const std::string& m) : msg("TB Overflow: " + m) {}
        TBOverflow() : msg("TB Overflow") {}

        const char* what() const noexcept { return msg.c_str(); }

    private:
        std::string msg;
    };

    class EventGenerator
    {
    public:
        EventGenerator(const PadPlane& pads, const arma::vec& vd, const double clock, const double shape,
                       const unsigned massNum, const double ioniz, const double gain, const double tilt,
                       const arma::vec& beamCtr = arma::zeros<arma::vec>(3))
            : pads(pads), vd(vd), clock(clock), shape(shape), massNum(massNum), ioniz(ioniz), gain(gain),
              tilt(tilt), beamCtr(beamCtr), pulseTemplate(elecPulse(1, shape, clock, 0)) {}

        std::map<pad_t, arma::vec> makeEvent(const Track& tr) const;
        std::map<pad_t, arma::vec> makeEvent(const arma::mat& pos, const arma::vec& en) const;
        std::map<uint16_t, Peak> makePeaksFromSimulation(const Track& tr) const;
        arma::mat makePeaksTableFromSimulation(const Track& tr) const;
        arma::mat makePeaksTableFromSimulation(const arma::mat& pos, const arma::vec& en) const;
        arma::vec makeMeshSignal(const Track& tr) const;
        arma::vec makeMeshSignal(const arma::mat& pos, const arma::vec& en) const;

    private:
        const PadPlane pads;
        const arma::vec vd;
        const double clock;
        const double shape;
        const unsigned massNum;
        const double ioniz;
        const double gain;
        const double tilt;
        const arma::vec3 beamCtr;
        const arma::vec pulseTemplate;  // Precalculated GET electronics pulse. Speeds up inner loop.
    };

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

#endif /* end of include guard: MCOPT_EVENTGEN_H */

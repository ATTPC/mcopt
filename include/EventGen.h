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

    arma::mat unTiltAndRecenter(const arma::mat& pos, const double tilt);

    arma::vec squareWave(const arma::uword size, const arma::uword leftEdge,
                         const arma::uword width, const double height);
    arma::vec elecPulse(const double amplitude, const double shape, const double clock, const double offset);

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
        EventGenerator(const PadPlane* pads_, const arma::vec& vd_, const double clock_, const double shape_,
                       const unsigned massNum_, const double ioniz_, const double gain_, const double tilt_,
                       const double diffSigma_)
            : vd(vd_), massNum(massNum_), ioniz(ioniz_), gain(gain_), tilt(tilt_), diffSigma(diffSigma_), pads(pads_),
              clock(clock_), shape(shape_), pulseTemplate(elecPulse(1, shape, clock, 0)) {}


        arma::vec numElec(const arma::vec& en) const;
        arma::mat diffuseElectrons(const arma::mat& tr) const;
        arma::mat prepareTrack(const arma::mat& pos, const arma::vec& en) const;
        std::map<pad_t, arma::vec> makeEvent(const Track& tr) const;
        std::map<pad_t, arma::vec> makeEvent(const arma::mat& pos, const arma::vec& en) const;
        std::map<uint16_t, Peak> makePeaksFromSimulation(const Track& tr) const;
        arma::mat makePeaksTableFromSimulation(const Track& tr) const;
        arma::mat makePeaksTableFromSimulation(const arma::mat& pos, const arma::vec& en) const;
        arma::vec makeMeshSignal(const Track& tr) const;
        arma::vec makeMeshSignal(const arma::mat& pos, const arma::vec& en) const;
        arma::vec makeHitPattern(const arma::mat& pos, const arma::vec& en) const;

        double getClock() const { return clock; }
        void setClock(const double c) { clock = c; pulseTemplate = elecPulse(1, shape, clock, 0); }
        double getShape() const { return shape; }
        void setShape(const double s) { shape = s; pulseTemplate = elecPulse(1, shape, clock, 0); }

        arma::vec vd;
        unsigned massNum;
        double ioniz;
        double gain;
        double tilt;
        double diffSigma;

    private:
        const PadPlane* const pads;
        double clock;
        double shape;
        arma::vec pulseTemplate;  // Precalculated GET electronics pulse. Speeds up inner loop.
    };
}

#endif /* end of include guard: MCOPT_EVENTGEN_H */

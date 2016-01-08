#ifndef MCOPT_EVENTGEN_H
#define MCOPT_EVENTGEN_H

#include <armadillo>
#include <map>
#include "Track.h"
#include "PadPlane.h"

namespace mcopt
{
    arma::mat calibrate(const Track& tr, const arma::vec& vd, const double clock);
    arma::mat uncalibrate(const Track& tr, const arma::vec& vd, const double clock, const int offset=0);

    std::map<uint16_t, unsigned long> makePeaksFromSimulation(const PadPlane& pads, const Track& tr, const arma::vec& vd,
                                                  const double clock, const int massNum, const double ioniz);
}

#endif /* end of include guard: MCOPT_EVENTGEN_H */

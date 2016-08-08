#ifndef MCOPT_TRACKER_H
#define MCOPT_TRACKER_H

#include "Track.h"
#include <armadillo>
#include <vector>
#include "Exceptions.h"
#include "Constants.h"
#include "Gas.h"

namespace mcopt
{
    class State
    {
    public:
        arma::vec::fixed<3> pos;
        arma::vec::fixed<3> mom;
        double en;
        double de;

        State()
            : pos(arma::zeros<arma::vec>(3)), mom(arma::zeros<arma::vec>(3)), en(0), de(0) {}
        State(const arma::vec3& pos_, const arma::vec3& mom_, const double en_, const double de_)
            : pos(pos_), mom(mom_), en(en_), de(de_) {}
    };

    class Tracker
    {
    public:
        Tracker(const unsigned massNum, const unsigned chargeNum, const Gas& gas,
                const arma::vec3& efield, const arma::vec3& bfield);

        Track trackParticle(const double x0, const double y0, const double z0,
                            const double enu0,  const double azi0, const double pol0) const;

        unsigned int getMassNum() const { return massNum; }
        unsigned int getChargeNum() const { return chargeNum; }
        arma::vec3 getEfield() const { return efield; }
        arma::vec3 getBfield() const { return bfield; }

        const Gas& getGas() const { return gas; }

    private:
        void updateState(State& st, const double tstep) const;

        unsigned int massNum;
        double mass_kg;
        double mass_mc2;
        unsigned int chargeNum;
        double charge;
        Gas gas;
        arma::vec3 efield;
        arma::vec3 bfield;
    };
}

#endif /* end of include guard: MCOPT_TRACKER_H */

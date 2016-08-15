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
        Tracker(const unsigned massNum_, const unsigned chargeNum_, const Gas* gas_,
                const arma::vec3& efield_, const arma::vec3& bfield_)
            : massNum(massNum_), mass_kg(massNum_ * Constants::P_KG), mass_mc2(massNum_ * Constants::P_MC2),
              chargeNum(chargeNum_), charge(chargeNum_ * Constants::E_CHG), gas(gas_),
              efield(efield_), bfield(bfield_) {}

        Track trackParticle(const double x0, const double y0, const double z0,
                            const double enu0,  const double azi0, const double pol0) const;

        unsigned int getMassNum() const { return massNum; }
        unsigned int getChargeNum() const { return chargeNum; }
        arma::vec3 getEfield() const { return efield; }
        arma::vec3 getBfield() const { return bfield; }

    private:
        void updateState(State& st, const double tstep) const;

        unsigned int massNum;
        double mass_kg;
        double mass_mc2;
        unsigned int chargeNum;
        double charge;
        const Gas* const gas;
        arma::vec3 efield;
        arma::vec3 bfield;
    };
}

#endif /* end of include guard: MCOPT_TRACKER_H */

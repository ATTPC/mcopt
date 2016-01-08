#ifndef MCOPT_TRACKER_H
#define MCOPT_TRACKER_H

#include "Track.h"
#include <armadillo>
#include <vector>

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
        State(const arma::vec3& pos, const arma::vec3& mom, const double en, const double de)
            : pos(pos), mom(mom), en(en), de(de) {}
    };

    class Tracker
    {
    public:
        Tracker(const unsigned massNum, const unsigned chargeNum, const std::vector<double>& eloss,
                    const arma::vec3& efield, const arma::vec3& bfield)
            : massNum(massNum), chargeNum(chargeNum), eloss(eloss), efield(efield), bfield(bfield) {}

        Track trackParticle(const double x0, const double y0, const double z0,
                            const double enu0,  const double azi0, const double pol0) const;
        Track trackParticle(const double x0, const double y0, const double z0,
                            const double enu0,  const double azi0, const double pol0,
                            const arma::vec3& bfield) const;

    private:
        void updateState(State& st, const double tstep) const;
        void updateState(State& st, const double tstep, const arma::vec3& thisBfield) const;

        unsigned int massNum;
        unsigned int chargeNum;
        std::vector<double> eloss;
        arma::vec3 efield;
        arma::vec3 bfield;
    };

    class TrackingFailed : public std::exception
    {
    public:
        TrackingFailed(std::string m) : msg(m) {}
        const char* what() const noexcept { return msg.c_str(); }

    private:
        std::string msg;
    };
}

#endif /* end of include guard: MCOPT_TRACKER_H */

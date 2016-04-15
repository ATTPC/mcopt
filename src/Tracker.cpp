#include "Tracker.h"

static const double C_LGT = 299792458.0;
static const double E_CHG = 1.602176565e-19;
static const double P_MC2 = 938.272046;
static const double P_KG = 1.672621777e-27;
static const double POS_STEP = 1e-3;

namespace mcopt
{
    static inline double threshold(const double value, const double threshmin)
    {
        if (value < threshmin) {
            return threshmin;
        }
        else {
            return value;
        }
    }

    static inline double betaFactor(const double en, const double mass_mc2)
    {
        return sqrt(en * (en + 2*mass_mc2)) / (en + mass_mc2);
    }

    Tracker::Tracker(const unsigned massNum, const unsigned chargeNum, const std::vector<double>& eloss,
                     const arma::vec3& efield, const arma::vec3& bfield)
        : massNum(massNum), mass_kg(massNum * P_KG), mass_mc2(massNum * P_MC2), chargeNum(chargeNum),
          charge(chargeNum * E_CHG), eloss(eloss), efield(efield), bfield(bfield) {}

    void Tracker::updateState(State& st, const double tstep) const
    {
        arma::vec3 mom_si = st.mom * 1e6 * E_CHG / C_LGT;

        double beta = betaFactor(st.en, mass_mc2);
        if (beta < 1e-8) {
            st.pos = {0, 0, 0};
            st.mom = {0, 0, 0};
            st.en = 0;
            st.de = 0;
            return;
        }

        double invgamma = sqrt(1 - beta*beta);  // equals 1 / gamma. This is faster to compute.

        arma::vec3 vel = mom_si * invgamma / mass_kg;
        arma::vec3 force = charge * (efield + arma::cross(vel, bfield));

        arma::vec3 new_mom_si = mom_si + force * tstep;
        arma::vec3 new_mom = new_mom_si * 1e-6 / E_CHG * C_LGT;

        // NOTE: I'm assuming the Lorentz force doesn't change the energy appreciably.

        arma::vec3 new_vel = new_mom_si * invgamma / mass_kg;
        arma::vec3 new_pos = st.pos + vel * tstep;

        assert(st.en >= 0);  // This is assumed by the cast to unsigned long in the next line.
        size_t elossIdx = static_cast<size_t>(lround(st.en * 1000));  // Convert to keV, the index units

        double stopping;
        try {
            stopping = eloss.at(elossIdx);
        }
        catch (const std::out_of_range&) {
            throw TrackingFailed("Energy loss index out of range.");
        }
        double dpos = tstep * beta * C_LGT;
        double de = threshold(stopping*dpos, 0);
        assert(de >= 0);

        if (stopping <= 0 || de < 1e-10) {
            // position remains the same
            st.mom = {0, 0, 0};
            st.en = 0;
            st.de = de;
            return;
        }
        else {
            double en = threshold(st.en - de, 0);
            assert(en >= 0);

            double new_mom_mag = sqrt(pow(en + mass_mc2, 2) - pow(mass_mc2, 2));
            double old_mom_mag = sqrt(arma::accu(arma::square(new_mom)));
            double mom_correction_factor = new_mom_mag / old_mom_mag;

            new_mom *= mom_correction_factor;
            new_mom_si *= mom_correction_factor;

            st.pos = new_pos;
            st.mom = new_mom;
            st.en = en;
            st.de = de;
            return;
        }
    }

    Track Tracker::trackParticle(const double x0, const double y0, const double z0,
                                     const double enu0,  const double azi0, const double pol0) const
    {
        const unsigned long maxIters = 10000;

        Track tr;
        State st;

        const double en0 = enu0 * massNum;
        double current_time = 0;

        double mom_mag = sqrt(pow(en0 + mass_mc2, 2) - pow(mass_mc2, 2));
        arma::vec::fixed<3> mom = {mom_mag * cos(azi0) * sin(pol0),
                                   mom_mag * sin(azi0) * sin(pol0),
                                   mom_mag * cos(pol0)};

        st.pos = {x0, y0, z0};
        st.mom = mom;
        st.en = en0;
        st.de = 0;

        tr.append(st.pos(0), st.pos(1), st.pos(2), current_time, st.en / massNum, azi0, pol0);

        for (unsigned long i = 1; i < maxIters; i++) {
            double beta = betaFactor(st.en, mass_mc2);
            if (st.en < 1e-3 || beta < 1e-6) {
                break;
            }
            double tstep = POS_STEP / (beta * C_LGT);

            updateState(st, tstep);

            double azi = atan2(st.mom(1), st.mom(0));
            double pol = atan2(sqrt(st.mom(0)*st.mom(0) + st.mom(1)*st.mom(1)), st.mom(2));

            current_time += tstep;

            tr.append(st.pos(0), st.pos(1), st.pos(2), current_time, st.en / massNum, azi, pol);

            // double rad = arma::norm(st.pos);
            if (st.pos(2) < 0 || st.pos(2) > 1) { // || rad > 0.275) {
                break;
            }
        }

        return tr;
    }
}

#include <cmath>
#include <cassert>
#include "mcopt.h"

static const double C_LGT = 299792458.0;
static const double E_CHG = 1.602176565e-19;
static const double P_MC2 = 938.272046;
static const double P_KG = 1.672621777e-27;
static const double POS_STEP = 1e-3;

static inline double threshold(const double value, const double threshmin)
{
    if (value < threshmin) {
        return threshmin;
    }
    else {
        return value;
    }
}

static inline double betaFactor(const double en, const double massNum)
{
    return (sqrt(en)*sqrt(en + 2*P_MC2*massNum)) / (en + massNum*P_MC2);
}

void updateState(State& st, const Conditions& cond, const double tstep)
{
    arma::vec3 mom_si = st.mom * 1e6 * E_CHG / C_LGT;

    double beta = betaFactor(st.en, cond.massNum);
    if (beta < 1e-8) {
        st.pos = {0, 0, 0};
        st.mom = {0, 0, 0};
        st.en = 0;
        st.de = 0;
        return;
    }

    double gamma = 1 / sqrt(1 - beta*beta);

    arma::vec3 vel = mom_si / (gamma * cond.massNum * P_KG);
    arma::vec3 force = cond.chargeNum * E_CHG * (cond.efield + arma::cross(vel, cond.bfield));

    arma::vec3 new_mom_si = mom_si + force * tstep;
    arma::vec3 new_mom = new_mom_si * 1e-6 / E_CHG * C_LGT;

    // NOTE: I'm assuming the Lorentz force doesn't change the energy appreciably.

    arma::vec3 new_vel = new_mom_si / (gamma * cond.massNum * P_KG);
    arma::vec3 new_pos = st.pos + vel * tstep;

    assert(st.en >= 0);  // This is assumed by the cast to unsigned long in the next line.
    size_t elossIdx = static_cast<size_t>(lround(st.en * 1000));  // Convert to keV, the index units

    double stopping;
    try {
        stopping = cond.eloss.at(elossIdx);
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

        double new_mom_mag = sqrt(pow(en + cond.massNum*P_MC2, 2) - pow(cond.massNum*P_MC2, 2));
        double old_mom_mag = arma::norm(new_mom);  // "old" means before eloss, in this case
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

Track trackParticle(const double x0, const double y0, const double z0,
                    const double enu0,  const double azi0, const double pol0, const Conditions& cond)
{
    const unsigned long maxIters = 10000;

    Track tr;
    State st;

    const double en0 = enu0 * cond.massNum;
    double current_time = 0;

    double mom_mag = sqrt(pow(en0 + cond.massNum*P_MC2, 2) - pow(cond.massNum*P_MC2, 2));
    arma::vec::fixed<3> mom = {mom_mag * cos(azi0) * sin(pol0),
                               mom_mag * sin(azi0) * sin(pol0),
                               mom_mag * cos(pol0)};

    st.pos = {x0, y0, z0};
    st.mom = mom;
    st.en = en0;
    st.de = 0;

    tr.x.push_back(st.pos(0));
    tr.y.push_back(st.pos(1));
    tr.z.push_back(st.pos(2));
    tr.time.push_back(current_time);
    tr.enu.push_back(st.en / cond.massNum);
    tr.azi.push_back(azi0);
    tr.pol.push_back(pol0);

    for (unsigned long i = 1; i < maxIters; i++) {
        double beta = betaFactor(st.en, cond.massNum);
        if (st.en < 1e-3 || beta < 1e-6) {
            break;
        }
        double tstep = POS_STEP / (beta * C_LGT);

        updateState(st, cond, tstep);

        double azi = atan2(st.mom(1), st.mom(0));
        double pol = atan2(sqrt(pow(st.mom(0), 2) + pow(st.mom(1), 2)), st.mom(2));

        current_time += tstep;

        tr.x.push_back(st.pos(0));
        tr.y.push_back(st.pos(1));
        tr.z.push_back(st.pos(2));
        tr.time.push_back(current_time);
        tr.enu.push_back(st.en / cond.massNum);
        tr.azi.push_back(azi);
        tr.pol.push_back(pol);

        // double rad = arma::norm(st.pos);
        // if (st.pos(2) < 0 || st.pos(2) > 1 || rad > 0.275) {
        //     break;
        // }
    }

    #ifndef NDEBUG
        const size_t N = tr.x.size();
        assert(tr.x.size() == N);
        assert(tr.y.size() == N);
        assert(tr.z.size() == N);
        assert(tr.time.size() == N);
        assert(tr.enu.size() == N);
        assert(tr.azi.size() == N);
        assert(tr.pol.size() == N);
    #endif

    return tr;
}

arma::mat findDeviations(const arma::mat& simtrack, const arma::mat& expdata)
{
    // ASSUMPTION: matrices must be sorted in increasing Z order.

    arma::vec xInterp;
    arma::vec yInterp;

    arma::interp1(simtrack.col(2), simtrack.col(0), expdata.col(2), xInterp);
    arma::interp1(simtrack.col(2), simtrack.col(1), expdata.col(2), yInterp);

    return arma::join_horiz(xInterp - expdata.col(0), yInterp - expdata.col(1));
}

double runTrack(const arma::vec& p, const arma::mat& trueValues,
                 const Conditions& condBase)
{
    Conditions cond = condBase;
    cond.bfield = {0, 0, p(6)};

    Track tr = trackParticle(p(0), p(1), p(2), p(3), p(4), p(5), cond);
    arma::vec xv (tr.x);
    arma::vec yv (tr.y);
    arma::vec zv (tr.z);
    arma::mat simtrack = arma::join_horiz(xv, arma::join_horiz(yv, zv));

    double zlenSim = simtrack.col(2).max() - simtrack.col(2).min();
    double zlenTrue = trueValues.col(2).max() - trueValues.col(2).min();

    double chi2 = 0;
    if (simtrack.n_rows > 10 and std::abs(zlenSim - zlenTrue) < 0.5) {
        arma::mat devs = findDeviations(simtrack, trueValues);
        arma::vec temp = arma::sum(arma::square(devs), 1);
        chi2 = arma::median(temp);
    }
    else {
        chi2 = 100;
    }

    return chi2;
}

arma::mat makeParams(const arma::vec ctr, const arma::vec sigma, const unsigned numSets,
                     const arma::vec mins, const arma::vec maxes)
{
    const arma::uword numVars = ctr.n_rows;
    assert(sigma.n_rows == numVars);
    assert(mins.n_rows == numVars);
    assert(maxes.n_rows == numVars);

    arma::mat params = arma::randu(numSets, numVars);

    for (arma::uword i = 0; i < numVars; i++) {
        params.col(i) = arma::clamp(ctr(i) + (params.col(i) - 0.5) * sigma(i),
                                    mins(i), maxes(i));
    }

    return params;
}

MCminimizeResult MCminimize(const arma::vec& ctr0, const arma::vec& sigma0,
                            const arma::mat& trueValues, const Conditions& cond,
                            const unsigned numIters, const unsigned numPts,
                            const double redFactor)
{
    arma::uword numVars = ctr0.n_rows;

    arma::vec mins = ctr0 - sigma0 / 2;
    arma::vec maxes = ctr0 + sigma0 / 2;
    arma::vec ctr = ctr0;
    arma::vec sigma = sigma0;
    arma::mat allParams(numPts * numIters, numVars);
    arma::vec minChis(numIters);
    arma::vec goodParamIdx(numIters);

    for (unsigned i = 0; i < numIters; i++) {
        arma::mat params = makeParams(ctr, sigma, numPts, mins, maxes);
        arma::vec chi2s (numPts, arma::fill::zeros);

        #pragma omp parallel for schedule(static)
        for (unsigned j = 0; j < numPts; j++) {
            arma::vec p = arma::conv_to<arma::colvec>::from(params.row(j));
            double chi2;
            try {
                chi2 = runTrack(p, trueValues, cond);
            }
            catch (const std::exception&) {
                chi2 = arma::datum::nan;
            }
            chi2s(j) = chi2;
        }

        arma::uword minChiLoc = 0;
        double minChi = chi2s.min(minChiLoc);

        ctr = arma::conv_to<arma::colvec>::from(params.row(minChiLoc));
        sigma *= redFactor;

        allParams.rows(i*numPts, (i+1)*numPts-1) = params;
        minChis(i) = minChi;
        goodParamIdx(i) = minChiLoc + i*numPts;
    }
    return std::make_tuple(ctr, allParams, minChis, goodParamIdx);
}

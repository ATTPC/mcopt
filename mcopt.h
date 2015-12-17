#ifndef MCOPT_H
#define MCOPT_H

#include <armadillo>
#include <vector>
#include <string>
#include <tuple>
#include <set>


arma::vec dropNaNs(const arma::vec& data);

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

class Track
{
public:
    void append(const double x, const double y, const double z, const double time,
                const double enu, const double azi, const double pol);

    arma::mat getMatrix() const;
    size_t numPts() const;
    void unTiltAndRecenter(const arma::vec beamCtr, const double tilt);

private:
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> time;
    std::vector<double> enu;
    std::vector<double> azi;
    std::vector<double> pol;
};

typedef std::tuple<arma::vec, arma::mat, arma::vec, arma::vec> MCminimizeResult;

class MCminimizer
{
public:
    MCminimizer(const unsigned massNum, const unsigned chargeNum, const std::vector<double>& eloss,
                const arma::vec3& efield, const arma::vec3& bfield)
        : massNum(massNum), chargeNum(chargeNum), eloss(eloss), efield(efield), bfield(bfield) {}

    Track trackParticle(const double x0, const double y0, const double z0,
                        const double enu0,  const double azi0, const double pol0) const;
    Track trackParticle(const double x0, const double y0, const double z0,
                        const double enu0,  const double azi0, const double pol0,
                        const arma::vec3& bfield) const;

    static arma::mat makeParams(const arma::vec& ctr, const arma::vec& sigma, const unsigned numSets,
                                const arma::vec& mins, const arma::vec& maxes);
    static arma::mat findDeviations(const arma::mat& simtrack, const arma::mat& expdata);
    double runTrack(const arma::vec& p, const arma::mat& trueValues) const;
    MCminimizeResult minimize(const arma::vec& ctr0, const arma::vec& sigma0, const arma::mat& trueValues,
                              const unsigned numIters, const unsigned numPts, const double redFactor) const;


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

class PadPlane
{
public:
    PadPlane(const arma::Mat<uint16_t>& lt, const double xLB, const double xDelta,
             const double yLB, const double yDelta, const double rotAngle=0);
    uint16_t getPadNumberFromCoordinates(const double x, const double y) const;

private:
    double xLowerBound;
    double yLowerBound;
    double xDelta;
    double yDelta;
    double xUpperBound;
    double yUpperBound;
    const arma::Mat<uint16_t> lookupTable;
    double rotAngle;
};

arma::mat calibrate(const Track& tr, const arma::vec vd, const double clock);
arma::mat uncalibrate(const Track& tr, const arma::vec vd, const double clock, const int offset=0);

std::set<uint16_t> findHitPads(const PadPlane& pads, const Track& tr, const arma::vec& vd, const double clock);

#endif /* def MCOPT_H */

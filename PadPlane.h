#ifndef MCOPT_PADPLANE_H
#define MCOPT_PADPLANE_H

#include <armadillo>

namespace mcopt {

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

}

#endif /* def MCOPT_PADPLANE_H */

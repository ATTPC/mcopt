#ifndef MCOPT_PADPLANE_H
#define MCOPT_PADPLANE_H

#include <armadillo>
#include <vector>

#ifdef HAVE_HDF5
    #include <H5Cpp.h>
#endif /* defined HAVE_HDF5 */

namespace mcopt {

    class PadPlane
    {
    public:
        PadPlane(const arma::Mat<uint16_t>& lt, const double xLB, const double xDelta,
                 const double yLB, const double yDelta, const double rotAngle=0);
        uint16_t getPadNumberFromCoordinates(const double x, const double y) const;

        static std::vector<std::vector<std::vector<double>>> generatePadCoordinates(const double rotation_angle);
        std::vector<double> getPadCenter(const size_t padNum) const;

    private:
        double xLowerBound;
        double yLowerBound;
        double xDelta;
        double yDelta;
        double xUpperBound;
        double yUpperBound;
        const arma::Mat<uint16_t> lookupTable;
        double sinRotAngle;
        double cosRotAngle;
        const std::vector<std::vector<std::vector<double>>> padCoords;
    };

#ifdef HAVE_HDF5
    arma::Mat<uint16_t> readLUT(const std::string& path);
#endif /* defined HAVE_HDF5 */

}

#endif /* def MCOPT_PADPLANE_H */

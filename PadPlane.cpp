#include "PadPlane.h"

namespace mcopt {

    PadPlane::PadPlane(const arma::Mat<uint16_t>& lt, const double xLB, const double xDelta,
                       const double yLB, const double yDelta, const double rotAngle)
    : xLowerBound(xLB), yLowerBound(yLB), xDelta(xDelta), yDelta(yDelta), lookupTable(lt), rotAngle(rotAngle)
    {
        xUpperBound = xLowerBound + lookupTable.n_cols * xDelta;
        yUpperBound = yLowerBound + lookupTable.n_rows * yDelta;
    }

    uint16_t PadPlane::getPadNumberFromCoordinates(const double x, const double y) const
    {
        double rotX = cos(-rotAngle) * x - sin(-rotAngle) * y;
        double rotY = sin(-rotAngle) * x + cos(-rotAngle) * y;

        double xPos = std::round((rotX - xLowerBound) / xDelta);
        double yPos = std::round((rotY - yLowerBound) / yDelta);

        if (xPos < 0 || yPos < 0 || xPos >= lookupTable.n_cols || yPos >= lookupTable.n_rows) {
            return 20000;
        }
        return lookupTable(xPos, yPos);
    }

}

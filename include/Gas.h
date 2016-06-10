#ifndef GAS_H
#define GAS_H

#include <armadillo>
#include <vector>
#include <cassert>

#include "Exceptions.h"

namespace mcopt {

    class Gas
    {
    public:
        Gas(const std::vector<double>& eloss, const std::vector<double>& enVsZ);

        double energyLoss(const double energy) const;
        double energyFromVertexZPosition(const double z) const;
        double getProjectileStopPosition() const { return projStopLoc; }

    private:
        std::vector<double> elossData;  // Values in MeV/m indexed in keV
        std::vector<double> enVsZdata;  // Values in MeV indexed in mm
        double projStopLoc;
    };

}

#endif /* end of include guard: GAS_H */

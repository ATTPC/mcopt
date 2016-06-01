#include "Gas.h"

namespace mcopt {

    Gas::Gas(const std::vector<double>& eloss, const std::vector<double>& enVsZ)
    : elossData(eloss), enVsZdata(enVsZ)
    {
        // enVsZdata is indexed in mm between 0 and 1 m, so it must have 1000 elements.
        if (enVsZdata.size() != 1000) {
            throw GasError("enVsZdata must have exactly 1000 elements");
        }
    }

    double Gas::energyLoss(const double energy) const
    {
        if (energy < 0) {
            throw GasError("Energy must be >= 0");
        }
        size_t elossIdx = static_cast<size_t>(lround(energy * 1000));  // Convert to keV, the index units

        double result;
        try {
            result = elossData.at(elossIdx);
        }
        catch (const std::out_of_range&) {
            throw GasError("Energy loss index out of range.");
        }

        return result;
    }

    double Gas::energyFromVertexZPosition(const double z) const
    {
        // Need to assume micromegas at 0, window at 1 m
        if (z < 0 || z > 1) {
            throw GasError("Vertex position must be between 0 and 1");
        }

        size_t zIdx = static_cast<size_t>(lround(z * 1000));  // Convert to mm, the index units

        double result;
        try {
            result = enVsZdata.at(zIdx);
        }
        catch (const std::out_of_range&) {
            throw GasError("Energy loss index out of range.");
        }

        return result;
    }

}

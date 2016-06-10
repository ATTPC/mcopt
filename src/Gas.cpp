#include "Gas.h"

namespace mcopt {

    Gas::Gas(const std::vector<double>& eloss, const std::vector<double>& enVsZ)
    : elossData(eloss), enVsZdata(enVsZ)
    {
        // enVsZdata is indexed in mm between 0 and 1 m, so it must have 1000 elements.
        if (enVsZdata.size() != 1000) {
            throw GasError("enVsZdata must have exactly 1000 elements");
        }

        // Find projectile range. enVsZ should decrease monotonically from 1000 to 0.
        // If it goes back up, we found the stopping point.
        projStopLoc = 1.0;
        for (auto riter = enVsZ.rbegin() + 1; riter != enVsZ.rend(); riter++) {
            if (*riter - *(riter - 1) >= 0) {
                projStopLoc = 1.0 - (riter - enVsZ.rbegin()) / 1000.0;
            }
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
            throw GasError("Energy loss index " + std::to_string(elossIdx) + " out of range in Gas::energyLoss.");
        }

        return result;
    }

    double Gas::energyFromVertexZPosition(const double z) const
    {
        // Need to assume micromegas at 0, window at 1 m
        if (z < 0 || z >= 1) {
            throw GasError("Vertex position z=" + std::to_string(z) + " must be in range [0, 1)");
        }

        size_t zIdx = static_cast<size_t>(lround(z * 1000));  // Convert to mm, the index units

        double result;
        try {
            result = enVsZdata.at(zIdx);
        }
        catch (const std::out_of_range&) {
            throw GasError("Energy loss index (" + std::to_string(zIdx) + ") out of range in Gas::energyFromVertexZPosition.");
        }

        return result;
    }

}

#include "catch.hpp"
#include "PadPlane.h"
#include <vector>

TEST_CASE("Pad coordinates are generated correctly", "[PadPlane]")
{
    std::vector<std::vector<std::vector<double>>> pads = mcopt::PadPlane::generatePadCoordinates(0);

    SECTION("Dimensions are correct")
    {
        REQUIRE(pads.size() == 10240);
        for (const auto& pad : pads) {
            CHECK(pad.size() == 3);
            for (const auto& point : pad) {
                CHECK(point.size() == 2);
            }
        }
    }

    SECTION("Pad heights are ok")
    {
        for (const auto& pad : pads) {
            double height = std::round(std::abs(pad.at(1).at(1) - pad.at(0).at(1)));
            if (!(std::abs(height - 8) < 1e-3) && !(std::abs(height - 4) < 1e-3)) {
                CAPTURE(height);
                FAIL("The height was not one of the valid options.");
            }
        }
    }

    SECTION("Pad widths are ok")
    {
        for (const auto& pad : pads) {
            double width = std::abs(pad.at(2).at(0) - pad.at(0).at(0));
            if (!(std::abs(width - 4.67) < 1e-2) && !(std::abs(width - 9.58) < 1e-2)) {
                CAPTURE(width);
                FAIL("The width was not one of the valid options.");
            }
        }
    }
}

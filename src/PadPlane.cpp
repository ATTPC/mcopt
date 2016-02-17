#include "PadPlane.h"

namespace mcopt {

    PadPlane::PadPlane(const arma::Mat<uint16_t>& lt, const double xLB, const double xDelta,
                       const double yLB, const double yDelta, const double rotAngle)
    : xLowerBound(xLB), yLowerBound(yLB), xDelta(xDelta), yDelta(yDelta), lookupTable(lt), rotAngle(rotAngle),
      sinRotAngle(sin(rotAngle)), cosRotAngle(cos(rotAngle)), padCoords(generatePadCoordinates(rotAngle))
    {
        xUpperBound = xLowerBound + lookupTable.n_cols * xDelta;
        yUpperBound = yLowerBound + lookupTable.n_rows * yDelta;
    }

    uint16_t PadPlane::getPadNumberFromCoordinates(const double x, const double y) const
    {
        const double cosAng =  cosRotAngle;  // cos(-rotAngle)
        const double sinAng = -sinRotAngle;  // sin(-rotAngle)
        double rotX = cosAng * x - sinAng * y;
        double rotY = sinAng * x + cosAng * y;

        double xPos = std::round((rotX - xLowerBound) / xDelta);
        double yPos = std::round((rotY - yLowerBound) / yDelta);

        if (xPos < 0 || yPos < 0 || xPos >= lookupTable.n_cols || yPos >= lookupTable.n_rows) {
            return 20000;
        }
        return lookupTable(xPos, yPos);
    }

    static inline std::vector<std::vector<double>> create_triangle(const double x_offset, const double y_offset,
                                                                   const double side, const double orient)
    {
        return {{x_offset, y_offset},
                {x_offset + side / 2, y_offset + orient * side * std::sqrt(3.) / 2.},
                {x_offset + side, y_offset}};
    }

    std::vector<std::vector<std::vector<double>>>
    PadPlane::generatePadCoordinates(const double rotation_angle)
    {
        double small_z_spacing = 2 * 25.4 / 1000;
        double small_tri_side = 184. * 25.4 / 1000;
        double umega_radius = 10826.772 * 25.4 / 1000;
        double beam_image_radius = 4842.52 * 25.4 / 1000;
        size_t pad_index = 0;

        double small_x_spacing = 2 * small_z_spacing / std::sqrt(3);
        double small_y_spacing = small_x_spacing * std::sqrt(3);
        double dotted_s_tri_side = 4. * small_x_spacing + small_tri_side;
        double dotted_s_tri_hi = dotted_s_tri_side * std::sqrt(3) / 2;
        double dotted_l_tri_side = 2. * dotted_s_tri_side;
        double dotted_l_tri_hi = dotted_l_tri_side * std::sqrt(3) / 2;
        double large_x_spacing = small_x_spacing;
        double large_y_spacing = small_y_spacing;
        double large_tri_side = dotted_l_tri_side - 4 * large_x_spacing;
        double large_tri_hi = dotted_l_tri_side * std::sqrt(3) / 2;
        int row_len_s = static_cast<int>(std::pow(2, std::ceil(std::log(beam_image_radius / dotted_s_tri_side) / std::log(2))));
        int row_len_l = static_cast<int>(std::floor(umega_radius / dotted_l_tri_hi));

        double xoff = 0;
        double yoff = 0;

        // Create half a circle

        std::vector<std::vector<std::vector<double>>> pads (10240,
            std::vector<std::vector<double>>(3, std::vector<double>(2, 0))); // Initializes the results

        for (int j = 0; j < row_len_l; j++) {
            int pads_in_half_hex = 0;
            int pads_in_hex = 0;
            double row_length = std::abs(std::sqrt(std::pow(umega_radius, 2) - std::pow(j * dotted_l_tri_hi + dotted_l_tri_hi / 2, 2)));

            if (j < row_len_s / 2) {
                pads_in_half_hex = (2 * row_len_s - 2 * j) / 4;
                pads_in_hex = 2 * row_len_s - 1 - 2 * j;
            }

            double pads_in_half_row = row_length / dotted_l_tri_side;
            int pads_out_half_hex = static_cast<int>(std::round(2 * (pads_in_half_row - pads_in_half_hex)));
            int pads_in_row = 2 * pads_out_half_hex + 4 * pads_in_half_hex - 1;

            int ort = 1;

            for (int i = 0; i < pads_in_row; i++) {
                if (i == 0) {
                    if (j % 2 == 0) {
                        ort = -1;
                    }
                    if (((pads_in_row - 1) / 2) % 2 == 1) {
                        ort = -ort;
                    }
                }
                else {
                    ort = -ort;
                }

                double pad_x_off = -(pads_in_half_hex + pads_out_half_hex / 2) * dotted_l_tri_side
                                   + i * dotted_l_tri_side / 2 + 2 * large_x_spacing + xoff;

                if ((i < pads_out_half_hex) || (i > pads_in_hex + pads_out_half_hex - 1) || (j > row_len_s / 2 - 1)) {
                    // Outside hex
                    double pad_y_off = j * dotted_l_tri_hi + large_y_spacing + yoff;
                    if (ort == -1) {
                        pad_y_off += large_tri_hi;
                    }
                    pads.at(pad_index) = create_triangle(pad_x_off, pad_y_off, large_tri_side, ort);
                    pad_index += 1;
                }
                else {
                    // inside hex, make 4 small triangles
                    double pad_y_off = j * dotted_l_tri_hi + large_y_spacing + yoff;

                    if (ort == -1) {
                        pad_y_off = j * dotted_l_tri_hi + 2 * dotted_s_tri_hi - small_y_spacing + yoff;
                    }

                    pads.at(pad_index) = create_triangle(pad_x_off, pad_y_off, small_tri_side, ort);
                    pad_index += 1;

                    double tmp_pad_x_off = pad_x_off + dotted_s_tri_side / 2;
                    double tmp_pad_y_off = pad_y_off + ort * dotted_s_tri_hi - 2 * ort * small_y_spacing;
                    pads.at(pad_index) = create_triangle(tmp_pad_x_off, tmp_pad_y_off, small_tri_side, -ort);
                    pad_index += 1;

                    tmp_pad_y_off = pad_y_off + ort * dotted_s_tri_hi;
                    pads.at(pad_index) = create_triangle(tmp_pad_x_off, tmp_pad_y_off, small_tri_side, ort);
                    pad_index += 1;

                    tmp_pad_x_off = pad_x_off + dotted_s_tri_side;
                    pads.at(pad_index) = create_triangle(tmp_pad_x_off, pad_y_off, small_tri_side, ort);
                    pad_index += 1;
                }
            }
        }

        // Create symmetric pads
        for (size_t i = 0; i < pad_index; i++) {
            pads.at(pad_index + i).at(0) = {pads.at(i).at(0).at(0), -pads.at(i).at(0).at(1)};
            pads.at(pad_index + i).at(1) = {pads.at(i).at(1).at(0), -pads.at(i).at(1).at(1)};
            pads.at(pad_index + i).at(2) = {pads.at(i).at(2).at(0), -pads.at(i).at(2).at(1)};
        }

        const double cosang = std::cos(rotation_angle);
        const double sinang = std::sin(rotation_angle);

        for (auto& pad : pads) {
            for (auto& point : pad) {
                auto x = point.at(0);
                auto y = point.at(1);
                point.at(0) = cosang * x - sinang * y;
                point.at(1) = sinang * x + cosang * y;
            }
        }

        return pads;
    }

    std::vector<double> PadPlane::getPadCenter(const size_t padNum) const
    {
        const auto& pad = padCoords.at(padNum);
        const auto& v1 = pad.at(0);
        const auto& v2 = pad.at(1);
        const auto& v3 = pad.at(2);

        return {(v1.at(0) + v2.at(0) + v3.at(0)) / 3, (v1.at(1) + v2.at(1) + v3.at(1)) / 3};
    }
}

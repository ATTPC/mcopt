#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <H5Cpp.h>
#include <string>
#include <cassert>
#include <armadillo>

std::vector<double> readEloss(const std::string& path);
arma::Mat<uint16_t> readLUT(const std::string& path);

#endif /* end of include guard: UTILS_H */

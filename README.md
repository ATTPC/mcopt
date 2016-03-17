# mcopt

This library contains the code needed for a Monte Carlo fit to AT-TPC data.

## Requirements

- Armadillo C++ linear algebra library: http://arma.sourceforge.net/
- OpenMP: for parallel track generation during the minimization. The code *will* work without it, but it will (obviously) be quite slow.

## Compiling

Compilation can be done using CMake:

    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
    make install

If you use CMake, the library will also install a CMake script that will allow you to use `find_package(mcopt)` when writing `CMakeLists.txt` files for other programs.

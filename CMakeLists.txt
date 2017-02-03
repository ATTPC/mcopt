cmake_minimum_required(VERSION 3.1)
project(mcopt VERSION 2.2.0)

# ============================
# Set build flags and settings
# ============================

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_STANDARD 14)

if(APPLE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -gdwarf-2")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -gdwarf-2")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DARMA_NO_DEBUG")

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything -Wno-c++98-compat -Wno-padded -Wno-weak-vtables -Wno-exit-time-destructors")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra -Wfloat-equal -Wpointer-arith -Wcast-qual")
endif()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
    # Hack on support for icc
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# ==============
# Find libraries
# ==============

find_package(OpenMP)
if("${OPENMP_FOUND}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not available. Output will be single-threaded.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-source-uses-openmp")
endif()

find_package(Armadillo REQUIRED)
include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})

find_package(HDF5 COMPONENTS CXX)
if(HDF5_FOUND)
    include_directories(SYSTEM ${HDF5_INCLUDE_DIRS})
    add_definitions(-DHAVE_HDF5)
endif()

# ==============
# Set up targets
# ==============

include_directories(include)

set(MCOPT_SRC src/EventGen.cpp src/Trigger.cpp src/MinimizerBase.cpp src/MCminimizer.cpp src/BeamLocationEstimator.cpp
    src/PadPlane.cpp src/Track.cpp src/Tracker.cpp src/PadMap.cpp src/Gas.cpp)

add_library(mcopt SHARED ${MCOPT_SRC})
target_link_libraries(mcopt ${ARMADILLO_LIBRARIES})
if(HDF5_FOUND)
    target_link_libraries(mcopt ${HDF5_CXX_LIBRARIES})
endif()

set(TEST_MCOPT_SRC test/catch_main.cpp test/test_mcopt.cpp test/test_EventGen.cpp test/test_PadPlane.cpp)

# add_executable(test_mcopt ${TEST_MCOPT_SRC})
# target_link_libraries(test_mcopt ${ARMADILLO_LIBRARIES} mcopt)
# if(HDF5_FOUND)
#     target_link_libraries(test_mcopt ${HDF5_CXX_LIBRARIES})
#     file(COPY "test/LUT.h5" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# endif()
# target_compile_options(test_mcopt PUBLIC "-Wno-shadow")

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# ==============
# Set up install
# ==============

install(TARGETS mcopt EXPORT mcoptTargets
        LIBRARY DESTINATION lib)

install(DIRECTORY include/ DESTINATION include/mcopt FILES_MATCHING PATTERN "*.h")

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/mcoptConfigVersion.cmake"
  VERSION ${mcopt_VERSION}
  COMPATIBILITY AnyNewerVersion
)

export(EXPORT mcoptTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/mcoptTargets.cmake"
  NAMESPACE mcopt::
)
configure_file(cmake/mcoptConfig.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/mcoptConfig.cmake"
  COPYONLY
)

set(ConfigPackageLocation lib/cmake/mcopt)
install(EXPORT mcoptTargets
  FILE
    mcoptTargets.cmake
  NAMESPACE
    mcopt::
  DESTINATION
    ${ConfigPackageLocation}
)
install(
  FILES
    cmake/mcoptConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/mcoptConfigVersion.cmake"
  DESTINATION
    ${ConfigPackageLocation}
)

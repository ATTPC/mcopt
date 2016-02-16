cmake_minimum_required(VERSION 3.1)
project(mcopt VERSION 2.1.0)

# Set up RPATH for build products
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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything -Wno-c++98-compat -Wno-padded -Wno-weak-vtables -Wno-shadow -Wno-exit-time-destructors")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra -Wfloat-equal -Wpointer-arith -Wcast-qual")
endif()

find_package(OpenMP)
if("${OPENMP_FOUND}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not available. Output will be single-threaded.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-source-uses-openmp")
endif()

find_package(Armadillo REQUIRED)
include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})

include_directories(include)

set(MCOPT_SRC src/EventGen.cpp src/MCminimizer.cpp src/PadPlane.cpp src/Track.cpp src/Tracker.cpp src/PadMap.cpp)

add_library(mcopt SHARED ${MCOPT_SRC})
target_link_libraries(mcopt ${ARMADILLO_LIBRARIES})

set(TEST_MCOPT_SRC test/catch_main.cpp test/test_mcopt.cpp test/test_EventGen.cpp test/test_PadPlane.cpp)

add_executable(test_mcopt ${TEST_MCOPT_SRC})
target_link_libraries(test_mcopt ${ARMADILLO_LIBRARIES} mcopt)

install(TARGETS mcopt
        LIBRARY DESTINATION lib)

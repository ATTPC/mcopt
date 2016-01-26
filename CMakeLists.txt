set(MCOPT_SRC EventGen.cpp MCminimizer.cpp PadPlane.cpp Track.cpp Tracker.cpp PadMap.cpp)

add_library(mcopt SHARED ${MCOPT_SRC})
target_link_libraries(mcopt ${ARMADILLO_LIBRARIES})

# add_executable(test_tracking test.cpp)
# target_link_libraries(test_tracking ${ARMADILLO_LIBRARIES} mcopt)

add_executable(test_mcopt catch_main.cpp test_mcopt.cpp test_EventGen.cpp test_PadPlane.cpp)
target_link_libraries(test_mcopt ${ARMADILLO_LIBRARIES} mcopt)

install(TARGETS mcopt
        LIBRARY DESTINATION lib)

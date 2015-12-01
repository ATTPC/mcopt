add_library(mcopt STATIC mcopt.cpp)
target_link_libraries(mcopt ${ARMADILLO_LIBRARIES})

add_executable(test_tracking test.cpp)
target_link_libraries(test_tracking ${ARMADILLO_LIBRARIES} mcopt)

add_executable(test_mcopt test_mcopt.cpp)
target_link_libraries(test_mcopt ${ARMADILLO_LIBRARIES} mcopt)

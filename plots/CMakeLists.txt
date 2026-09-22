# Documentation sample-plot generator.

add_executable(generate_samples generate_samples.cpp)

target_link_libraries(generate_samples PRIVATE approxLCD_static)

add_executable(generate_samples_even generate_samples_even.cpp)

target_link_libraries(generate_samples_even PRIVATE approxLCD_static)
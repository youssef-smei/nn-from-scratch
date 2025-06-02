cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkMNIST)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -O2")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add source files
set(SOURCE_FILES
    src/main.cpp
    src/neural_network.cpp
    src/mnist_loader.cpp
)

# Create executable
add_executable(mnist_classifier ${SOURCE_FILES})

# Install target
install(TARGETS mnist_classifier DESTINATION bin)

# Add a custom target for running the program
add_custom_target(run
    COMMAND $<TARGET_FILE:mnist_classifier>
    DEPENDS mnist_classifier
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Print status message
message(STATUS "Project configured for ${CMAKE_SYSTEM_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
cmake_minimum_required(VERSION 3.14)
project(sqgturb CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Release build flags
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")

# ---------------------------------------------------------------------------
# Find FFTW3 single-precision library
# ---------------------------------------------------------------------------
find_library(FFTW3F_LIB NAMES fftw3f
    HINTS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

find_path(FFTW3_INCLUDE_DIR fftw3.h
    HINTS /usr/include /usr/local/include)

if(NOT FFTW3F_LIB)
    message(FATAL_ERROR
        "FFTW3 single-precision library (libfftw3f) not found.\n"
        "Install with: apt-get install libfftw3-dev")
endif()

message(STATUS "Found FFTW3f: ${FFTW3F_LIB}")
message(STATUS "FFTW3 include: ${FFTW3_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# SQG library
# ---------------------------------------------------------------------------
add_library(sqg STATIC
    sqgturb/sqg.cpp
)
target_include_directories(sqg PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/sqgturb
    ${FFTW3_INCLUDE_DIR}
)
target_link_libraries(sqg PUBLIC ${FFTW3F_LIB} m)

# ---------------------------------------------------------------------------
# Example runner
# ---------------------------------------------------------------------------
add_executable(sqg_run sqg_run.cpp)
target_include_directories(sqg_run PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/sqgturb
    ${FFTW3_INCLUDE_DIR}
)
target_link_libraries(sqg_run PRIVATE sqg)

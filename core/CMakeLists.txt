cmake_minimum_required (VERSION 2.6)
project(drforest)


## ======================================================================================##
## Check for C++11. For GCC this is >=4.7
## ======================================================================================##
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    message("Compiler with C++11 support found.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler, i.e. gcc >= 4.7 or Clang >= 3.0.")
endif()

## ======================================================================================##
## Compiler flags
## ======================================================================================##
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -O2")

## ======================================================================================##
## External libraries
## ======================================================================================##
find_package(Armadillo REQUIRED)
message("Armadillo Include: ${ARMADILLO_INCLUDE_DIRS}")
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(Boost 1.66 REQUIRED COMPONENTS filesystem)
message("Boost Include: ${Boost_INCLUDE_DIR}")
message("Boost Libraries: ${Boost_LIBRARY_DIRS}")
message("Boost Libraries: ${Boost_LIBRARIES}")
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

# NOTE: apple needs to use the clong-omp compilers
find_package(OpenMP REQUIRED)

## ======================================================================================##
## Subdirectories and source files
## ======================================================================================##
include_directories(src tests)
file(GLOB_RECURSE SOURCES src/*.cpp tests/*.cpp)

## ======================================================================================##
## Debug and release targets
## ======================================================================================##
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

ADD_CUSTOM_TARGET(debug
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
    COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
    )
ADD_CUSTOM_TARGET(release
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
    COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)

## ======================================================================================##
## Executable
## ======================================================================================##
add_executable(drforest ${SOURCES})
target_link_libraries(drforest ${ARMADILLO_LIBRARIES} Boost::filesystem OpenMP::OpenMP_CXX)

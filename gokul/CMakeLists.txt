cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(SpeechSeparation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Find LibTorch ──────────────────────────────────────────────────────────────
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ── Find libsndfile ────────────────────────────────────────────────────────────
find_library(SNDFILE_LIBRARY NAMES sndfile
    HINTS
        ${CMAKE_SOURCE_DIR}/third_party
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
)
find_path(SNDFILE_INCLUDE_DIR NAMES sndfile.h
    HINTS
        ${CMAKE_SOURCE_DIR}/third_party/include
        /usr/include
        /usr/local/include
)
if(NOT SNDFILE_LIBRARY OR NOT SNDFILE_INCLUDE_DIR)
    message(FATAL_ERROR "libsndfile not found. Install with: sudo apt install libsndfile1-dev")
endif()
message(STATUS "Found libsndfile: ${SNDFILE_LIBRARY}")
message(STATUS "  include dir:    ${SNDFILE_INCLUDE_DIR}")

# ── Core library ──────────────────────────────────────────────────────────────
add_library(speechsep STATIC
    src/audio_utils.cpp
    src/preprocessing.cpp
    src/stft.cpp
    src/augmentation.cpp
    src/dataset.cpp
    src/conv_tasnet.cpp
    src/unet.cpp
    src/losses.cpp
    src/metrics.cpp
)
target_include_directories(speechsep PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${SNDFILE_INCLUDE_DIR}
)
target_link_libraries(speechsep PUBLIC
    ${TORCH_LIBRARIES}
    ${SNDFILE_LIBRARY}
)

# ── Binaries ──────────────────────────────────────────────────────────────────
add_executable(train src/train.cpp)
target_link_libraries(train PRIVATE speechsep)

add_executable(inference src/inference.cpp)
target_link_libraries(inference PRIVATE speechsep)

# ── GUI (GTK3) ────────────────────────────────────────────────────────────────
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(GTK3 gtk+-3.0)
endif()
if(GTK3_FOUND)
    add_executable(gui src/gui.cpp)
    target_include_directories(gui PRIVATE ${GTK3_INCLUDE_DIRS})
    target_link_libraries(gui PRIVATE ${GTK3_LIBRARIES} pthread stdc++fs)
    target_compile_options(gui PRIVATE ${GTK3_CFLAGS_OTHER})
    message(STATUS "GTK3 found — building GUI")
else()
    message(STATUS "GTK3 not found — GUI will not be built (install libgtk-3-dev)")
endif()

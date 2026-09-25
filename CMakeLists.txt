cmake_minimum_required(VERSION 3.20)

project(evoke LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

find_package(ICU QUIET COMPONENTS i18n uc data)
if(ICU_FOUND)
    set(EVOKE_ICU_TARGETS ICU::i18n ICU::uc)
    if(TARGET ICU::data)
        list(APPEND EVOKE_ICU_TARGETS ICU::data)
    endif()
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(
            ICU_PKG
            QUIET
            IMPORTED_TARGET
            icu-i18n
            icu-uc
        )
    endif()

    if(TARGET PkgConfig::ICU_PKG)
        set(EVOKE_ICU_TARGETS PkgConfig::ICU_PKG)
    else()
        set(
            EVOKE_ICU_PREFIX_HINTS
            /opt/homebrew/opt/icu4c@78
            /opt/homebrew/opt/icu4c@77
            /opt/homebrew/opt/icu4c
            /usr/local/opt/icu4c
        )
        find_path(
            EVOKE_ICU_INCLUDE_DIR
            unicode/ubrk.h
            PATH_SUFFIXES include
            PATHS ${EVOKE_ICU_PREFIX_HINTS}
        )
        find_library(
            EVOKE_ICU_I18N_LIB
            NAMES icui18n
            PATH_SUFFIXES lib
            PATHS ${EVOKE_ICU_PREFIX_HINTS}
        )
        find_library(
            EVOKE_ICU_UC_LIB
            NAMES icuuc
            PATH_SUFFIXES lib
            PATHS ${EVOKE_ICU_PREFIX_HINTS}
        )
        find_library(
            EVOKE_ICU_DATA_LIB
            NAMES icudata
            PATH_SUFFIXES lib
            PATHS ${EVOKE_ICU_PREFIX_HINTS}
        )
        if(
            NOT EVOKE_ICU_INCLUDE_DIR
            OR NOT EVOKE_ICU_I18N_LIB
            OR NOT EVOKE_ICU_UC_LIB
            OR NOT EVOKE_ICU_DATA_LIB
        )
            message(FATAL_ERROR 'Unable to locate ICU headers and libraries')
        endif()

        add_library(evoke_icu INTERFACE)
        target_include_directories(
            evoke_icu
            INTERFACE
            ${EVOKE_ICU_INCLUDE_DIR}
        )
        target_link_libraries(
            evoke_icu
            INTERFACE
            ${EVOKE_ICU_I18N_LIB}
            ${EVOKE_ICU_UC_LIB}
            ${EVOKE_ICU_DATA_LIB}
        )
        set(EVOKE_ICU_TARGETS evoke_icu)
    endif()
endif()

add_library(
    evoke_core
    STATIC
    src/evoke_block_ranges.c
    src/evoke_core.c
    src/evoke_document_tid_lookup.c
    src/evoke_document_cow.c
    src/evoke_initial_fold_stream.c
    src/evoke_lexicon_cow.c
    src/evoke_prefix_cow.c
    src/evoke_posting_heat.c
    src/evoke_query.c
    src/evoke_semantic_accelerator.c
    src/evoke_semantic_accelerator_builder.c
    src/evoke_semantic_accelerator_directory.c
    src/evoke_semantic_forward.c
    src/evoke_semantic_forward_bound.c
    src/evoke_semantic_impact_frontier.c
    src/evoke_scope.c
    src/evoke_weighted_space_saving.c
    src/evoke_semantic_bmp.c
    src/evoke_segments.c
    src/evoke_term_cow.c
    src/evoke_stem.c
    src/evoke_storage.c
    src/evoke_text.c
)
target_include_directories(evoke_core PUBLIC src)
target_compile_options(
    evoke_core
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
if(NOT WIN32)
    target_link_libraries(evoke_core PUBLIC m)
endif()
target_link_libraries(evoke_core PUBLIC ${EVOKE_ICU_TARGETS})

add_executable(evoke_unit_tests tests/test_main.c)
target_link_libraries(evoke_unit_tests PRIVATE evoke_core)
target_include_directories(evoke_unit_tests PRIVATE src)

add_executable(evoke_simd_bench tests/benchmark_simd.c)
target_link_libraries(evoke_simd_bench PRIVATE evoke_core)
target_include_directories(evoke_simd_bench PRIVATE src)

add_executable(evoke_extent_bench tests/benchmark_extents.c)
target_link_libraries(evoke_extent_bench PRIVATE evoke_core)
target_include_directories(evoke_extent_bench PRIVATE src)

add_executable(
    evoke_semantic_bmp_bench
    tests/benchmark_semantic_bmp.c
)
target_link_libraries(evoke_semantic_bmp_bench PRIVATE evoke_core)
target_include_directories(evoke_semantic_bmp_bench PRIVATE src)

enable_testing()
add_test(NAME evoke_unit_tests COMMAND evoke_unit_tests)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt # cmake-lint: disable=C0103
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

file(GLOB unit_test_src CONFIGURE_DEPENDS "unit/*.cpp")

add_executable(unit_test ${unit_test_src})
target_link_libraries(unit_test GTest::gtest_main dynampi)

add_test(NAME unit_test COMMAND unit_test)

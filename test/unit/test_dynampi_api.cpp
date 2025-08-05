#include <gtest/gtest.h>

#include <dynampi/dynampi.hpp>

using namespace dynampi;

TEST(API, Version) {
  EXPECT_EQ(version::to_string(),
            "v" + std::to_string(DYNAMPI_VERSION_MAJOR) + "." +
                std::to_string(DYNAMPI_VERSION_MINOR) + "." +
                std::to_string(DYNAMPI_VERSION_PATCH));
  EXPECT_NE(version::to_string(),
            "v" + std::to_string(DYNAMPI_VERSION_MAJOR) + "." +
                std::to_string(DYNAMPI_VERSION_MINOR) + "." +
                std::to_string(DYNAMPI_VERSION_PATCH + 1));

  EXPECT_TRUE(version::is_at_least(DYNAMPI_VERSION_MAJOR, DYNAMPI_VERSION_MINOR,
                                   DYNAMPI_VERSION_PATCH));
  EXPECT_FALSE(version::is_at_least(
      DYNAMPI_VERSION_MAJOR + 1, DYNAMPI_VERSION_MINOR, DYNAMPI_VERSION_PATCH));
  EXPECT_FALSE(version::is_at_least(
      DYNAMPI_VERSION_MAJOR, DYNAMPI_VERSION_MINOR + 1, DYNAMPI_VERSION_PATCH));
  EXPECT_FALSE(version::is_at_least(
      DYNAMPI_VERSION_MAJOR, DYNAMPI_VERSION_MINOR, DYNAMPI_VERSION_PATCH + 1));
  EXPECT_TRUE(version::is_at_least(DYNAMPI_VERSION_MAJOR, DYNAMPI_VERSION_MINOR,
                                   DYNAMPI_VERSION_PATCH - 1));
  EXPECT_TRUE(version::is_at_least(
      DYNAMPI_VERSION_MAJOR - 1, DYNAMPI_VERSION_MINOR, DYNAMPI_VERSION_PATCH));
}

#pragma once

#include <cstdint>
#include <string>

namespace dynampi {

namespace version {
static const int major = DYNAMPI_VERSION_MAJOR;
static const int minor = DYNAMPI_VERSION_MINOR;
static const int patch = DYNAMPI_VERSION_PATCH;

static const std::string to_string() {
  return "v" + std::to_string(major) + "." + std::to_string(minor) + "." +
         std::to_string(patch);
}

static constexpr bool is_at_least(int major, int minor, int patch) {
  return (version::major > major) ||
         (version::major == major && version::minor > minor) ||
         (version::major == major && version::minor == minor &&
          version::patch >= patch);
}
}; // namespace version

} // namespace dynampi

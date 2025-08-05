#pragma once

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

static constexpr bool is_at_least(int v_major, int v_minor, int v_patch) {
  return (version::major > v_major) ||
         (version::major == v_major && version::minor > v_minor) ||
         (version::major == v_major && version::minor == v_minor &&
          version::patch >= v_patch);
}
}; // namespace version

} // namespace dynampi

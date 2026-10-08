#pragma once

#include "version.h"

#include <cstdlib>
#include <cstring>


namespace celerity::detail {

/// Whether strided device copies are performed through gpu3dcopylib: Celerity must be built with CELERITY_ENABLE_COPYLIB and the environment variable
/// CELERITY_COPYLIB must not be set to `off`.
inline bool copylib_enabled() {
	static const bool enabled = CELERITY_ENABLE_COPYLIB && !(std::getenv("CELERITY_COPYLIB") && std::strcmp(std::getenv("CELERITY_COPYLIB"), "off") == 0);
	return enabled;
}

} // namespace celerity::detail

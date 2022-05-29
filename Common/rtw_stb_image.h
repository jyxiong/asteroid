#pragma once

// Disable pedantic warnings for this external library.
#ifdef _MSC_VER
// Microsoft Visual C++ Compiler
#pragma warning (push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "External/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "External/stb_image_write.h"

// Restore warning levels.
#ifdef _MSC_VER
// Microsoft Visual C++ Compiler
#pragma warning (pop)
#endif

/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef FILEIO_PNGIMAGE_H
#define FILEIO_PNGIMAGE_H

#if GPU_ENABLED
#include <vector>
#include <stdint.h>

namespace osr {

void png_version_info(void);

std::vector<uint8_t> readPNG(const char *fname, int& width, int& height, int *pchannels = nullptr);
void writePNG(const char *iname, int width, int height, const void* data); 

}
#endif

#endif

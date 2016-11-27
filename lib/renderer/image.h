#ifndef IMAGE_H
#define IMAGE_H

#include <vector>

struct Image {
	/*
 	 * Image data in GL_RGB sequence.
	 * Notes: because of some funny alignment problem it's recommended to
	 * transform the data into GL_RGBA format before calling
	 * glTexSubImage2D if you want to use the data for texture mapping
	 */
	std::vector<unsigned char> bytes;
	int width;
	int height;
	int stride; // Stores the actual number of bytes for a scan line, you can ignore this for our current case.
};

#endif

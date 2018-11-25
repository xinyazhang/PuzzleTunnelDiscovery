#if GPU_ENABLED

#include "pngimage.h"
#include <zlib.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

using std::string;

/* current versions of libpng should provide this macro: */
#ifndef png_jmpbuf
#  define png_jmpbuf(png_ptr)   ((png_ptr)->jmpbuf)
#endif

#ifdef DEBUG
#  define Trace(x)  {fprintf x ; fflush(stderr); fflush(stdout);}
#else
#  define Trace(x)  ;
#endif

namespace {

struct PNGReader {
	typedef unsigned char   uch;
	typedef unsigned short  ush;
	typedef unsigned long   ulg;

	png_structp png_ptr = NULL;
	png_infop info_ptr = NULL;

	png_uint_32  width, height;
	int  bit_depth, color_type;

	enum {
		SUCCESS = 0,
		BAD_SIGNATURE = 1,
		BAD_IHDR = 2,
		NO_MEMORY = 4,
		FOPEN_FAILURE = 8
	};

	~PNGReader()
	{
		if (png_ptr && info_ptr) {
			png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
			png_ptr = NULL;
			info_ptr = NULL;
		}
	}

	/*
 	 * return value:
	 * 0 for success,
	 * 1 for bad sig,
	 * 2 for bad IHDR,
	 * 4 for no mem,
	 * 8 for file open failure
	 */
	int init(const char* filename)
	{
		uch sig[8];
		FILE *infile;

		if ((infile = fopen(filename, "rb")) == NULL)
			return FOPEN_FAILURE;

		/* check that the file really is a PNG image; could
		 * have used slightly more general png_sig_cmp() function instead */

		fread(sig, 1, 8, infile);
		if (png_sig_cmp(sig, 0, 8) != 0)
			return BAD_SIGNATURE;

		png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
		if (!png_ptr)
			return NO_MEMORY;

		info_ptr = png_create_info_struct(png_ptr);
		if (!info_ptr) {
			png_destroy_read_struct(&png_ptr, NULL, NULL);
			return NO_MEMORY;
		}

		/* we could create a second info struct here (end_info), but it's only
		 * useful if we want to keep pre- and post-IDAT chunk info separated
		 * (mainly for PNG-aware image editors and converters) */

		/* setjmp() must be called in every function that calls a PNG-reading
		 * libpng function */

		if (setjmp(png_jmpbuf(png_ptr))) {
			png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
			return BAD_IHDR;
		}

		png_init_io(png_ptr, infile);
		png_set_sig_bytes(png_ptr, 8);  /* we already read the 8 signature bytes */
		png_read_info(png_ptr, info_ptr);  /* read all PNG info up to image data */

		/* alternatively, could make separate calls to png_get_image_width(),
		 * etc., but want bit_depth and color_type for later [don't care about
		 * compression_type and filter_type => NULLs] */

		png_get_IHDR(png_ptr, info_ptr,
				&width, &height, &bit_depth, &color_type,
				NULL, NULL, NULL);
		/* OK, that's all we need for now; return happy */

		return 0;
	}

	uch *get_image(double display_exponent, int &pChannels, int &pRowbytes)
	{
		double  gamma;
		png_uint_32  i, rowbytes;

		/* setjmp() must be called in every function that calls a PNG-reading
		 * libpng function */

		if (setjmp(png_jmpbuf(png_ptr))) {
			png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
			return NULL;
		}

		/* expand palette images to RGB, low-bit-depth grayscale images to 8 bits,
		 * transparency chunks to full alpha channel; strip 16-bit-per-sample
		 * images to 8 bits per sample; and convert grayscale to RGB[A] */

		if (color_type == PNG_COLOR_TYPE_PALETTE)
			png_set_expand(png_ptr);
		if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
			png_set_expand(png_ptr);
		if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
			png_set_expand(png_ptr);
		if (bit_depth == 16)
			png_set_strip_16(png_ptr);
		if (color_type == PNG_COLOR_TYPE_GRAY ||
				color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
			png_set_gray_to_rgb(png_ptr);

		/* unlike the example in the libpng documentation, we have *no* idea where
		 * this file may have come from--so if it doesn't have a file gamma, don't
		 * do any correction ("do no harm") */

		if (png_get_gAMA(png_ptr, info_ptr, &gamma))
			png_set_gamma(png_ptr, display_exponent, gamma);

		/* all transformations have been registered; now update info_ptr data,
		 * get rowbytes and channels, and allocate image memory */

		png_read_update_info(png_ptr, info_ptr);

		pRowbytes = rowbytes = png_get_rowbytes(png_ptr, info_ptr);
		pChannels = (int)png_get_channels(png_ptr, info_ptr);

		uch *image_data = NULL;

		if ((image_data = (uch *)malloc(rowbytes*height)) == NULL) {
			png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
			return NULL;
		}

		std::vector<png_bytep> row_pointers(height);

		Trace((stderr, "readpng_get_image:  channels = %d, rowbytes = %ld, height = %ld\n", *pChannels, rowbytes, height));

		/* set the individual row_pointers to point at the correct offsets */

		for (i = 0;  i < height;  ++i)
			row_pointers[i] = image_data + i*rowbytes;

		/* now we can go ahead and just read the whole image */

		png_read_image(png_ptr, row_pointers.data());

		/* and we're done!  (png_read_end() can be omitted if no processing of
		 * post-IDAT text/time/etc. is desired) */

		png_read_end(png_ptr, NULL);
		return image_data;
	}
};

}; // Anonymous namespace

namespace osr {

void png_version_info(void)
{
	fprintf(stderr, "   Compiled with libpng %s; using libpng %s.\n",
		PNG_LIBPNG_VER_STRING, png_libpng_ver);
	fprintf(stderr, "   Compiled with zlib %s; using zlib %s.\n",
		ZLIB_VERSION, zlib_version);
}


std::vector<uint8_t> readPNG(const char *fname, int& width, int& height, int *pchannels)
{
	PNGReader reader;
	if (reader.init(fname) != 0)
		return std::vector<uint8_t>(); 
	width = reader.width;
	height = reader.height;

	static constexpr double gamma = 2.2;
	int channels, rowBytes;
	unsigned char* indata = reader.get_image(gamma, channels, rowBytes);
	if (!indata)
		return std::vector<uint8_t>(); 
	if (pchannels)
		*pchannels = channels;
	int bufsize = rowBytes * height;
	std::vector<uint8_t> data(bufsize);
	for (int j = 0; j < height; j++)
		for (int i = 0; i < rowBytes; i += channels)
			for (int k = 0; k < channels; k++)
				data[k + i + j * rowBytes] = indata[k + i + (height - j - 1) * rowBytes];
	free(indata);
	return data;
}

/*
 * Copyright 2002-2010 Guillaume Cottenceau.
 *
 * This software may be freely redistributed under the terms
 * of the X11 license.
 *
 */

void writePNG(const char *fname, int width, int height, const void *data)
{
	constexpr png_byte color_type = PNG_COLOR_TYPE_RGB;
	constexpr png_byte bit_depth = 8;

	png_structp png_ptr;
	png_infop info_ptr;
	int number_of_passes;
	/* create file */
	FILE *fp = fopen(fname, "wb");
	if (!fp)
		throw string("[write_png_file] File could not be opened for writing: ") + fname;

	/* initialize stuff */
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!png_ptr)
		throw string("[write_png_file] png_create_write_struct failed");

	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
		throw string("[write_png_file] png_create_info_struct failed");

	if (setjmp(png_jmpbuf(png_ptr)))
		throw string("[write_png_file] Error during init_io");

	png_init_io(png_ptr, fp);

	/* write header */
	if (setjmp(png_jmpbuf(png_ptr)))
		throw string("[write_png_file] Error during writing header");

	png_set_IHDR(png_ptr, info_ptr, width, height,
			bit_depth, color_type, PNG_INTERLACE_NONE,
			PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	std::vector<png_bytep> row_pointers(height);
	for (int i = 0; i < height; i++)
		row_pointers[height - i - 1] = (unsigned char*)data + i * width * 3;

#if 1
	png_set_rows(png_ptr, info_ptr, row_pointers.data());
	png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
#else
	png_write_info(png_ptr, info_ptr);
	/* write bytes */
	if (setjmp(png_jmpbuf(png_ptr)))
		abort_("[write_png_file] Error during writing bytes");
	png_write_image(png_ptr, row_pointers);
#endif

	/* end write */
	if (setjmp(png_jmpbuf(png_ptr)))
		throw string("[write_png_file] Error during end of write");

	png_write_end(png_ptr, NULL);

	fclose(fp);
	png_destroy_write_struct(&png_ptr, &info_ptr);
}

}

#endif

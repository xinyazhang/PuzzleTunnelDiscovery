/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "../MaxRectsBinPackReal.h"
#include <cstdio>
#include <vector>

int main(int argc, char **argv)
{
	
	if (argc < 5 || argc % 2 != 1)
	{
		printf("Usage: MaxRectsBinPackTest binWidth binHeight w_0 h_0 w_1 h_1 w_2 h_2 ... w_n h_n\n");
		printf("       where binWidth and binHeight define the size of the bin.\n");
		printf("       w_i is the width of the i'th rectangle to pack, and h_i the height.\n");
		printf("Example: MaxRectsBinPackTest 256 256 30 20 50 20 10 80 90 20\n");
		return 0;
	}
	
	using namespace rbp;
	
	// Create a bin to pack to, use the bin size from command line.
	MaxRectsBinPack bin;
	double binWidth = atof(argv[1]);
	double binHeight = atof(argv[2]);
	printf("Initializing bin to size %fx%f.\n", binWidth, binHeight);
	bin.Init(binWidth, binHeight);
	
	// Pack each rectangle (w_i, h_i) the user inputted on the command line.
	for(int i = 3; i < argc; i += 2)
	{
		// Read next rectangle to pack.
		double rectWidth = atof(argv[i]);
		double rectHeight = atof(argv[i+1]);
		printf("Packing rectangle of size %fx%f: ", rectWidth, rectHeight);

		// Perform the packing.
		MaxRectsBinPack::FreeRectChoiceHeuristic heuristic = MaxRectsBinPack::RectBestShortSideFit; // This can be changed individually even for each rectangle packed.
		Rect packedRect = bin.Insert(rectWidth, rectHeight, heuristic);

		// Test success or failure.
		if (packedRect.height > 0)
			printf("Packed to (x,y)=(%f,%f), (w,h)=(%f,%f). Free space left: %.2f%%\n", packedRect.x, packedRect.y, packedRect.width, packedRect.height, 100.f - bin.Occupancy()*100.f);
		else
			printf("Failed! Could not find a proper position to pack this rectangle into. Skipping this one.\n");
	}
	printf("Done. All rectangles packed.\n");

	bin.Init(binWidth, binHeight);

	std::vector<RectSize> rects_in;
	for(int i = 3; i < argc; i += 2) {
		double rectWidth = atof(argv[i]);
		double rectHeight = atof(argv[i+1]);
		rects_in.emplace_back(rectWidth, rectHeight);
	}
	std::vector<Rect> rects_out;
	bin.Insert(rects_in, rects_out, MaxRectsBinPack::RectBestShortSideFit);
	if (rects_out.size() == 0) {
		printf("Batch method failed!\n");
		return -1;
	}
	for (const auto& packedRect : rects_out) {
		if (packedRect.height > 0)
			printf("Packed to (x,y)=(%f,%f), (w,h)=(%f,%f). Free space left: %.2f%%\n", packedRect.x, packedRect.y, packedRect.width, packedRect.height, 100.f - bin.Occupancy()*100.f);
		else
			printf("Failed! Could not find a proper position to pack this rectangle into. Skipping this one.\n");
	}
}


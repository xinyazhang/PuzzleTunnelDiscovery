/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "readheat.h"

using std::string;

namespace {
	void
	skip_to_needle(std::istream& fin, const string& needle)
	{
		string s;
		do {
			fin >> s;
		} while(!fin.eof() && s != needle);
	}
}

HeatReader::HeatReader(std::ifstream& _fin)
	:fin(_fin)
{
	common_init();
}

void
HeatReader::common_init()
{
	if (!fin.is_open())
		throw std::runtime_error("[HeatReader] Cannot open file for read");
	if (fin.peek() == 0) {
		binary_ = true;
		fin.get(); fin.get(); // Skip "\0\n" header
	}
}

bool
HeatReader::ascii_read(HeatFrame& frame)
{
	skip_to_needle(fin, "t:");
	if (fin.eof())
		return false;
	fin >> frame.t >> frame.nvert;
	Eigen::VectorXd& field = frame.hvec;
	field.resize(frame.nvert);
	for(size_t i = 0; i < frame.nvert; i++)
		fin >> field(i);
	return !fin.eof();
}

bool
HeatReader::bin_read(HeatFrame& frame)
{
	fin.read((char*)&frame.t, sizeof(frame.t));
	if (fin.eof())
		return false;
	uint32_t nvert;
	fin.read((char*)&nvert, sizeof(nvert));
	frame.nvert = size_t(nvert);
	Eigen::VectorXd& field = frame.hvec;
	field.resize(nvert);
	fin.read((char*)field.data(), sizeof(double) * nvert);
	fin.read((char*)&frame.sum, sizeof(frame.sum));
	return !fin.eof();
}

bool
HeatReader::read_frame(HeatFrame& frame)
{
	if (!binary_)
		return ascii_read(frame);
	else
		return bin_read(frame);
}

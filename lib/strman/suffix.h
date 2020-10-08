/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef STRMAN_SUFFIX_H
#define STRMAN_SUFFIX_H

#include <string>

namespace strman {

using std::string;

inline bool has_suffix(const string &str, const string &suffix)
{
	return str.size() >= suffix.size() &&
		str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

string replace_suffix(const string &str, const string &suffix, const string &newsuffix)
{
	if (!has_suffix(str, suffix))
		return string();
	string ret(str);
	ret.replace(str.size() - suffix.size(), suffix.size(), newsuffix);
	return ret;
}

}

#endif

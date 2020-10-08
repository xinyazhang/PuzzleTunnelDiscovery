/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "map.h"
#include <fstream>
#include <queue>
#include <map>
#include <iostream>

namespace omplaux {

void Map::readMap(const std::string& fn)
{
	std::ifstream fin(fn);
	std::map<int, std::vector<int>> edges;
	while (true) {
		char c;
		fin >> c;
		if (fin.eof())
			break;
		else if (fin.fail())
			throw std::runtime_error("Failure occurs during reading " + fn);
		if (c == 'v') {
			double x, y, z;
			fin >> x >> y >> z;
			T.emplace_back(x, y, z);
			double qx, qy, qz, qw;
			fin >> qx >> qy >> qz >> qw;
			Q.emplace_back(qw, qx, qy, qz);
		} else if (c == 'e' || c == 'b') {
			int v1, v2;
			fin >> v1 >> v2;
			if (c == 'e') {
				E.emplace_back(v1, v2);
				edges[v1].emplace_back(v2);
			}
		}
	}
	std::vector<int> old2new(T.size(), 0);
	std::queue<int> queue;
	queue.push(1);
	old2new[1] = 1;
	// locate vertices to keep
	// 1: keep
	// 0: drop (this is default)
	while (!queue.empty()) {
		int v1 = queue.front();
		queue.pop();
		for (int v2 : edges[v1]) {
			if (old2new[v2])
				continue;
			queue.push(v2);
			old2new[v2] = 1;
		}
	}
	int newidx = 0;
	for (int i = 0; i < int(T.size()); i++) {
		if (old2new[i])
			old2new[i] = newidx++;
		else
			old2new[i] = -1;
	}
	std::cerr << "total vertices left: " << newidx << std::endl;
	for (auto& e : E) {
		int nv1 = old2new[e.first];
		int nv2 = old2new[e.second];
		e = std::pair<int, int>(nv1, nv2);
	}
	int i = 0;
	for (auto it = T.begin(); it != T.end(); i++) {
		if (old2new[i] < 0)
			it = T.erase(it);
		else
			it++;
	}
	i = 0;
	for (auto it = Q.begin(); it != Q.end(); i++) {
		if (old2new[i] < 0)
			it = Q.erase(it);
		else
			it++;
	}
	for (auto it = E.begin(); it != E.end(); ) {
		if (it->first < 0 || it->second < 0)
			it = E.erase(it);
		else
			it++;
	}
}

}

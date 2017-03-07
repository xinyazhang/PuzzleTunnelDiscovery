#include "map.h"
#include <fstream>

namespace omplaux {

void Map::readMap(const std::string& fn)
{
	std::ifstream fin(fn);
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
		} else if (c == 'e') {
			int v1, v2;
			fin >> v1 >> v2;
			E.emplace_back(v1, v2);
		}
	}
}

}

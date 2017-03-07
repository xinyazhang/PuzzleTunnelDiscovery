#include <omplaux/map.h>
#include <fstream>
#include <strman/suffix.h>

using std::endl;

void translateToPly(const omplaux::Map& planmap, std::ostream& fout)
{
	fout << "ply" << endl << "format ascii 1.0" << endl;
	fout << "element vertex " << planmap.T.size() << endl;
	fout << "property double x" << endl;
	fout << "property double y" << endl;
	fout << "property double z" << endl;
	fout << "property double nx" << endl;
	fout << "property double ny" << endl;
	fout << "property double nz" << endl;
	fout << "element edge " << planmap.E.size() << endl;
	fout << "property int vertex1" << endl;
	fout << "property int vertex2" << endl;
	fout << "end_header" << endl;
	Eigen::Vector3d init_direction(0, 1, 0);
	for (size_t i = 0; i < planmap.T.size(); i++) {
		const auto& v = planmap.T[i];
		const auto& q = planmap.Q[i];
		Eigen::Vector3d rot = q.toRotationMatrix() * init_direction;
		fout << v.transpose() << ' ' << rot.transpose() << endl;
	}
	for (size_t i = 0; i < planmap.E.size(); i++) {
		auto pair = planmap.E[i];
		fout << pair.first << " " << pair.second << endl;
	}
}

int main(int argc, char* argv[])
{
	for (int i = 1; i < argc; i++) {
		std::string inf(argv[i]);
		std::string ouf = strman::replace_suffix(inf, ".map", ".ply");
		if (ouf.empty())
			continue;
		omplaux::Map planmap;
		planmap.readMap(inf);
		std::ofstream fout(ouf);
		fout.precision(17);
		translateToPly(planmap, fout);
	}
	return 0;
}

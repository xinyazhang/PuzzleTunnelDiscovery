#include "readtet.h"
#include <unistd.h>
#include <string>
#include <Eigen/Core>
#include <iostream>
#include <igl/barycenter.h>
#include <igl/viewer/Viewer.h>
#include <igl/jet.h>

using std::string;
using std::endl;
using std::cerr;
using std::fixed;
using std::vector;

void usage()
{
	std::cerr << "Options: -i <tetgen file prefix> -f <heat field data file>" << endl;
}

class KeyDown {
private:
	Eigen::MatrixXd& V_;
	Eigen::MatrixXi& E_;
	Eigen::MatrixXi& P_;
	Eigen::MatrixXd B;
	vector<Eigen::VectorXd>& fields_;
	int frameid_ = 0;
public:
	KeyDown(
		Eigen::MatrixXd& V,
		Eigen::MatrixXi& E,
		Eigen::MatrixXi& P,
		vector<Eigen::VectorXd>& fields
		)
		: V_(V), E_(E), P_(P), fields_(fields)
	{
		igl::barycenter(V,P,B);
		frameid_ = fields_.size()/2;
		std::cerr << "KeyDown constructor was called " << endl;
	}

	bool operator()(igl::viewer::Viewer& viewer, unsigned char key, int modifier)
	{
		using namespace std;
		using namespace Eigen;

		if (key == 'K') {
			frameid_ -= fields_.size()/10;
		} else if (key == 'J') {
			frameid_ += fields_.size()/10;
		}
		frameid_ = std::max(frameid_, 0);
		frameid_ = std::min(int(fields_.size() - 1), frameid_);
		std::cerr << "Frame ID: " << frameid_
			<< "\tStepping: " << fields_.size() / 10
			<< "\tKey: " << key << " was pressed "
			<< endl;

		if (key >= '1' && key <= '9')
		{
			double t = double((key - '1')+1) / 9.0;

			VectorXd v = B.col(2).array() - B.col(2).minCoeff();
			v /= v.col(0).maxCoeff();

			vector<int> s;

			for (unsigned i=0; i<v.size();++i)
				if (v(i) < t)
					s.push_back(i);

			MatrixXd V_temp(s.size()*4,3);
			MatrixXi F_temp(s.size()*4,3);
			VectorXd Z_temp(s.size()*4);
			VectorXd& FV(fields_[frameid_]);
			MatrixXd C(s.size()*4, 3);

			for (unsigned i=0; i<s.size();++i)
			{
				V_temp.row(i*4+0) = V_.row(P_(s[i],0));
				V_temp.row(i*4+1) = V_.row(P_(s[i],1));
				V_temp.row(i*4+2) = V_.row(P_(s[i],2));
				V_temp.row(i*4+3) = V_.row(P_(s[i],3));
				F_temp.row(i*4+0) << (i*4)+0, (i*4)+1, (i*4)+3;
				F_temp.row(i*4+1) << (i*4)+0, (i*4)+2, (i*4)+1;
				F_temp.row(i*4+2) << (i*4)+3, (i*4)+2, (i*4)+0;
				F_temp.row(i*4+3) << (i*4)+1, (i*4)+2, (i*4)+3;
				Z_temp(i*4+0) = FV(P_(s[i],0));
				Z_temp(i*4+1) = FV(P_(s[i],1));
				Z_temp(i*4+2) = FV(P_(s[i],2));
				Z_temp(i*4+3) = FV(P_(s[i],3));
			}
			igl::jet(Z_temp,true,C);

			viewer.data.clear();
			viewer.data.set_mesh(V_temp,F_temp);
			viewer.data.set_colors(C);
			viewer.data.set_face_based(false);
		}

		return false;
	}
};

void skip_to_needle(std::istream& fin, const string& needle)
{
	string s;
	do {
		fin >> s;
	} while(!fin.eof() && s != needle);
}

int main(int argc, char* argv[])
{
	int opt;
	string iprefix, ffn;
	while ((opt = getopt(argc, argv, "i:f:")) != -1) {
		switch (opt) {
			case 'i': 
				iprefix = optarg;
				break;
			case 'f':
				ffn = optarg;
				break;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	if (iprefix.empty() || ffn.empty()) {
		std::cerr << "Missing input file" << endl;
		usage();
		return -1;
	}

	Eigen::MatrixXd V;
	Eigen::MatrixXi E;
	Eigen::MatrixXi P;
	vector<Eigen::VectorXd> fields;
	vector<double> times;
	try {
		readtet(V, E, P, iprefix);

		std::ifstream fin(ffn);
		if (!fin.is_open())
			throw std::runtime_error("Cannot open " + ffn + " for read");
		while (true) {
			skip_to_needle(fin, "t:");
			if (fin.eof())
				break;
			double t;
			size_t nvert;
			fin >> t >> nvert;
			times.emplace_back(t);
			Eigen::VectorXd field;
			field.resize(nvert);
			for(size_t i = 0; i < nvert; i++) {
				fin >> field(i);
			}
			fields.emplace_back(field);
		}
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}

	igl::viewer::Viewer viewer;
	viewer.callback_key_down = KeyDown(V,E,P, fields);
	viewer.launch();

	return 0;
}

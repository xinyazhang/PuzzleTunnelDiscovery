#include <readtet.h>
#include <write_ply_vc.h>
#include <unistd.h>
#include <string>
#include <unordered_map>
#include <Eigen/Core>
#include <iostream>
#include <igl/barycenter.h>
#include <igl/viewer/Viewer.h>
#include <igl/jet.h>
#include <time.h>

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

	void calibre_frameid()
	{
		frameid_ = std::max(frameid_, 0);
		frameid_ = std::min(int(fields_.size() - 1), frameid_);
	}

	vector<int> tetleft_;
	Eigen::MatrixXd V_temp_;
	Eigen::MatrixXi F_temp_;
	Eigen::VectorXd Z_temp_;
	std::unordered_map<int, int> vertidmap_; // Old vert id -> New vert id
	vector<int> vertback_;
	bool flush_viewer_ = false;
	vector<int> pathvert_;
	vector<int> pathvert_temp_;
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
		frameid_ = 0;
		std::cerr << "KeyDown constructor was called " << endl;
		adjust_slice_plane(0.5);
	}

	void save_frame()
	{
		time_t tnow = time(NULL);
		string now = ctime(&tnow);
		now = now.substr(0, now.size() - 1);
		Eigen::MatrixXd C;
		igl::jet(Z_temp_, 0.0, 1.0, C);
		for(auto vert : pathvert_temp_) {
			C.row(vert) = Eigen::Vector3d(1.0, 1.0, 1.0);
		}
		string fn = "visheat-snapshot-at-"+now+"-frame-"+std::to_string(frameid_)+".ply";
		std::cerr << "Saving model to file " << fn;
		write_ply_vc(fn,
				V_temp_,
				F_temp_,
				C);
		std::cerr << " done" << endl;
	}

	void adjust_slice_plane(double t)
	{
		Eigen::VectorXd v = B.col(2).array() - B.col(2).minCoeff();
		v /= v.col(0).maxCoeff();

		Eigen::VectorXd vmark;
		vmark.setZero(v.size());
		tetleft_.clear();
		for (unsigned i = 0; i < v.size(); ++i) {
			if (v(i) < t) {
				tetleft_.emplace_back(i);
				for(int j = 0; j < 4; j++) {
					vmark(P_(i, j)) = 1;
				}
			}
		}
		vertidmap_.clear();
		vertback_.clear();
		int vertid = 0;
		for(unsigned i = 0; i < v.size(); i++) {
			if (vmark(i) > 0) {
				vertidmap_[i] = vertid; // forward mapping, old -> new
				vertback_.emplace_back(i); // back mapping, new -> old
				vertid++;
			}
		}
		V_temp_.resize(vertback_.size(), 3);
		for(unsigned i = 0; i < vertback_.size(); i++) {
			V_temp_.row(i) = V_.row(vertback_[i]);
		}

		F_temp_.resize(tetleft_.size()*4,3);
		// Put old vert id to F_temp_
		for (unsigned i = 0; i < tetleft_.size(); ++i) {
			Eigen::VectorXi tet = P_.row(tetleft_[i]);
			F_temp_.row(i*4+0) << tet(0), tet(1), tet(3);
			F_temp_.row(i*4+1) << tet(0), tet(2), tet(1);
			F_temp_.row(i*4+2) << tet(3), tet(2), tet(0);
			F_temp_.row(i*4+3) << tet(1), tet(2), tet(3);
		}
		// Translate to new vert id
		for(unsigned j = 0; j < tetleft_.size()*4; j++)
			for(unsigned k = 0; k < 3; k++)
				F_temp_(j,k) = vertidmap_[F_temp_(j,k)];
		Z_temp_.resize(vertback_.size());

		pathvert_temp_.clear();
		for (int vert : pathvert_) {
			auto iter = vertidmap_.find(vert);
			if (iter == vertidmap_.end())
				continue ;
			pathvert_temp_.emplace_back(iter->second);
		}

		flush_viewer_ = true;
	}

	void update_frame(igl::viewer::Viewer& viewer)
	{
		Eigen::VectorXd& FV(fields_[frameid_]);
#if 0
		for (unsigned i = 0; i < tetleft_.size(); ++i) {
#if 0
			Z_temp_(i*4+0) = FV(P_(tetleft_[i],0));
			Z_temp_(i*4+1) = FV(P_(tetleft_[i],1));
			Z_temp_(i*4+2) = FV(P_(tetleft_[i],2));
			Z_temp_(i*4+3) = FV(P_(tetleft_[i],3));
#else
			Z_temp_(i*4+0) = V_(P_(tetleft_[i],0), 2);
			Z_temp_(i*4+1) = V_(P_(tetleft_[i],1), 2);
			Z_temp_(i*4+2) = V_(P_(tetleft_[i],2), 2);
			Z_temp_(i*4+3) = V_(P_(tetleft_[i],3), 2);
#endif
		}
#else
		for (unsigned i = 0; i < vertback_.size(); ++i) {
			Z_temp_(i) = FV(vertback_[i]);
		}
#endif
		if (flush_viewer_) {
			viewer.data.clear();
			viewer.data.set_mesh(V_temp_, F_temp_);
			flush_viewer_ = false;
		}
		viewer.data.set_face_based(false);
		viewer.data.V_material_diffuse.resize(vertback_.size(), 3);
#if 1
		igl::jet(Z_temp_, 0.0, 1.0, viewer.data.V_material_diffuse);
#else
		igl::jet(Z_temp_, true, viewer.data.V_material_diffuse);
#endif
		for(auto vert : pathvert_temp_) {
			viewer.data.V_material_diffuse.row(vert) = Eigen::Vector3d(1.0, 1.0, 1.0);
		}
		viewer.data.V_material_ambient = 0.1 * viewer.data.V_material_diffuse;
		constexpr double grey = 0.3;
		viewer.data.V_material_specular = grey+0.1*(viewer.data.V_material_diffuse.array()-grey);
		viewer.data.dirty |= igl::viewer::ViewerData::DIRTY_DIFFUSE;
		// The code above replaces viewer.data.set_colors(C);
		// to calculate the result in-place
	}

	bool operator()(igl::viewer::Viewer& viewer, unsigned char key, int modifier)
	{
		using namespace std;
		using namespace Eigen;
		bool redraw = false;

		if (toupper(key) == 'K') {
			frameid_ -= fields_.size()/10;
			redraw = true;
		} else if (toupper(key) == 'J') {
			frameid_ += fields_.size()/10;
			redraw = true;
		} 
		calibre_frameid();

		std::cerr << "Frame ID: " << frameid_
			<< "\tStepping: " << fields_.size() / 10
			<< "\tKey: " << key << " was pressed "
			<< endl;

		if (key >= '1' && key <= '9') {
			double t = double((key - '1')+1) / 8.0;
			redraw = true;
			adjust_slice_plane(t);
			std::cerr << "Tet left: " << tetleft_.size() << endl;
		}
		if (redraw)
			update_frame(viewer);

		if (toupper(key) == 'S') {
			save_frame();
			viewer.core.is_animating = false;
		}

		return false;
	}

	void next_frame() 
	{
		frameid_++;
		std::cerr << frameid_ << ' ';
		calibre_frameid();
	}

	void load_path(const string& path_file)
	{
		if (path_file.empty())
			return;
		std::ifstream fin(path_file);
		if (!fin.is_open())
			return;
		double x,y,z,tmp;
		int vert;
		while (true) {
			fin >> x >> y >> z >> vert >> tmp;
			if (!fin.eof()) {
				pathvert_.emplace_back(vert);
			} else {
				break;
			}
		}
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
	string iprefix, ffn, pfn;
	while ((opt = getopt(argc, argv, "i:f:p:")) != -1) {
		switch (opt) {
			case 'i': 
				iprefix = optarg;
				break;
			case 'f':
				ffn = optarg;
				break;
			case 'p':
				pfn = optarg;
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
	Eigen::VectorXi EBM;
	vector<Eigen::VectorXd> fields;
	vector<double> times;
	try {
		readtet(iprefix, V, E, P, &EBM);

		std::ifstream fin(ffn);
		if (!fin.is_open())
			throw std::runtime_error("Cannot open " + ffn + " for read");
		bool binary = false;
		if (fin.peek() == 0) {
			binary = true;
		}
		if (!binary) {
			while (true) {
				skip_to_needle(fin, "t:");
				if (fin.eof())
					break;
				double t;
				size_t nvert;
				fin >> t >> nvert;
				times.emplace_back(t);
				fields.emplace_back();
				Eigen::VectorXd& field = fields.back();
				field.resize(nvert);
				for(size_t i = 0; i < nvert; i++) {
					fin >> field(i);
				}
			}
		} else {
			fin.get(); fin.get(); // Skip "\0\n" header
			std::cerr.precision(17);
			while (!fin.eof()) {
				double t;
				uint32_t nvert;
				fin.read((char*)&t, sizeof(t));
				if (fin.eof())
					break;
				times.emplace_back(t);
				fin.read((char*)&nvert, sizeof(nvert));
				fields.emplace_back();
				Eigen::VectorXd& field = fields.back();
				field.resize(nvert);
				fin.read((char*)field.data(), sizeof(double) * nvert);
				double sum;
				fin.read((char*)&sum, sizeof(sum));
#if 0
				std::cerr << "t: " << t << "\t" << field.rows() << endl;
				std::cerr << field << endl;
				std::cerr << "sum: " << field.sum() << endl;
#endif
			}
		}
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}

	igl::viewer::Viewer viewer;
	KeyDown kd(V,E,P, fields);
	kd.load_path(pfn);
	viewer.callback_key_pressed = [&kd](igl::viewer::Viewer& viewer, unsigned char key, int modifier) -> bool { return kd.operator()(viewer, key, modifier); } ;
	viewer.callback_pre_draw = [&kd](igl::viewer::Viewer& viewer) -> bool
	{
		if (viewer.core.is_animating) {
			kd.next_frame();
			kd.update_frame(viewer);
		}
		return false;
	};
	viewer.core.is_animating = true;
	viewer.core.animation_max_fps = 30.;
	viewer.launch();

	return 0;
}

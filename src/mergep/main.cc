#include <iostream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <stdexcept>
#include <unsupported/Eigen/SparseExtra>
#include <fstream>

using std::string;
using std::endl;

void usage()
{
	std::cerr <<
R"zzz(
Options: -d file -p file -m file -o prefix
	-d file: the master discrete Laplacian matrix
	-p file: the periodical Laplacian matrix
	-m file: master node -> periodical node mapping file
	-o file: file name for output Laplacian matrix
)zzz";
}

class LapBlender {
public:
	LapBlender(const string& dfn,
	           const string& pfn,
	           const string& mfn)
	{
		Eigen::loadMarket(dlap_, dfn);
		Eigen::loadMarket(plap_, pfn);
		existingNodes_.setZero(plap_.rows());

		std::ifstream fin(mfn);
		while (!fin.eof()) {
			int oldvi, newvi;
			fin >> oldvi >> newvi;
			old2new_[oldvi] = newvi;
			existingNodes_(newvi) = 1;
		}
	}

	void init_blend()
	{
		int nhidden_nodes = plap_.rows() - existingNodes_.sum();
		mlap_ = dlap_;
		mlap_.conservativeResize(dlap_.rows() + nhidden_nodes, dlap_.cols() + nhidden_nodes);
		new2old_.clear();
		new2old_.reserve(old2new_.size());
		for (const auto& pair : old2new_) {
			int oldvi = pair.first;
			int newvi = pair.second;
			new2old_[newvi] = oldvi;
		}
		hidden_node_idx_ = dlap_.rows();
	}

	void blend()
	{
		init_blend();
		for (int k = 0; k < plap_.outerSize(); ++k) {
			for (decltype(plap_)::InnerIterator it(plap_, k); it; ++it) {
				it.value();
				it.row();   // row index
				it.col();   // col index (here it is equal to k)
				int oldvid0 = get_oldvid(it.row());
				int oldvid1 = get_oldvid(it.col());
				// Note: we must override the existing value,
				// because node pairs in the same layer
				// already have the value computed and stored.
				mlap_.coeffRef(oldvid0, oldvid1) = it.value();
			}
		}
	}

	int get_oldvid(int vid)
	{
		auto iter = new2old_.find(vid);
		if (iter == new2old_.end()) {
			// Insert to new
			int ret = hidden_node_idx_++;
			new2old_[vid] = ret;
			return ret;
		}
		return iter->second;
	}

	void dump(const string& fn)
	{
		Eigen::saveMarket(mlap_, fn);
	}
private:
	Eigen::SparseMatrix<double, Eigen::RowMajor> dlap_, plap_, mlap_;
	std::unordered_map<int, int> old2new_, new2old_;
	size_t hidden_node_idx_;
	Eigen::VectorXi existingNodes_;
};

int main(int argc, char* argv[])
{
	string dfn, pfn, mfn, ofn;
	int opt;
	while ((opt = getopt(argc, argv, "d:o:p:m:")) != -1) {
		switch (opt) {
			case 'd':
				dfn = optarg;
				break;
			case 'o':
				ofn = optarg;
				break;
			case 'p':
				pfn = optarg;
				break;
			case 'm':
				mfn = optarg;
				break;
			default:
				usage();
				return -1;
		}
	}
	if (dfn.empty() || ofn.empty() || pfn.empty() || mfn.empty()) {
		std::cerr << "Missing options" << endl;
		usage();
		return -1;
	}

	try {
		LapBlender blender(dfn, pfn, mfn);
		blender.blend();
		blender.dump(ofn);
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << endl;
		return -1;
	}
}

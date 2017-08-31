#include <string>
#include <iostream>
#include <math.h>
#include <readtet.h>
#include <geopick/pick2d.h>
#include <numerical/stability.h>
#include <atomic>
#include <igl/is_boundary_edge.h>
#include <igl/boundary_loop.h>
#include <igl/all_edges.h>
#include <igl/writeOBJ.h>
#include <unordered_map>
#include <unistd.h>

using std::string;
using std::endl;
using std::cerr;

void usage()
{
	std::cerr << 
R"zzz(This program generate the periodical part geometry from the obstacle space geometry
Options: -i <prefix> [-o file -m file]
	-i prefix: input tetgen prefix" << endl
	-o file: output the result to file instead of stdout
	-m file: output the mapping from the vertex id in tet to the vertex id in the output geometry
)zzz";
}

#define FACE_PER_TET 4

const int
proto_face_number[FACE_PER_TET][3] = {
	{1, 3, 2},
	{0, 2, 3}, 
	{3, 1, 0}, 
	{0, 1, 2}
};

constexpr double kBtmTheta = 0.0;
constexpr double kTopTheta = M_PI * 2; 

void slice(const Eigen::MatrixXd& V,
           const Eigen::MatrixXi& E,
           const Eigen::MatrixXi& P,
           double z,
           Eigen::VectorXi& sliceV,
           Eigen::MatrixXi& sliceF)
{
	Eigen::VectorXi marker;
	marker.setZero(V.rows());
#pragma omp parallel for
	for (int i = 0; i < V.rows(); i++) {
		if (fpclose(V(i, 2), z))
			marker(i) = 1;
	}
	sliceF.resize(P.rows() * FACE_PER_TET, 3);
	std::atomic<int> nface(0);
#pragma omp parallel for
	for (int i = 0; i < P.rows(); i++) {
		for (int j = 0; j < FACE_PER_TET; j++) {
			int iv0 = P(i, proto_face_number[j][0]);
			int iv1 = P(i, proto_face_number[j][1]);
			int iv2 = P(i, proto_face_number[j][2]);
			if (marker(iv0) > 0 && 
			    marker(iv1) > 0 &&
			    marker(iv2) > 0) {
				int f = nface++;
				sliceF(f, 0) = iv0;
				sliceF(f, 1) = iv1;
				sliceF(f, 2) = iv2;
			}
		}
	}
	sliceF.conservativeResize(nface.load(), Eigen::NoChange);
}

double
distance(const Eigen::MatrixXd& V,
         const std::vector<int>& objBL,
         const std::vector<int>& srcBL,
         double zdistance
         )
{
	double sqzd = zdistance * zdistance;
	Eigen::MatrixXd distmat;
	distmat.setZero(objBL.size(), srcBL.size());
//#pragma omp parallel for
	for (size_t i = 0; i < objBL.size(); i++) {
		Eigen::VectorXd objV = V.row(objBL[i]);
		for (size_t j = 0; j < srcBL.size(); j++) {
			Eigen::VectorXd srcV = V.row(srcBL[j]);
			distmat(i,j) = (objV - srcV).squaredNorm() - sqzd;
		}
	}
	return distmat.minCoeff();
}

const std::vector<int>&
pick_closest_BL(const Eigen::MatrixXd& V,
		const std::vector<int>& objBL,
		const std::vector<std::vector<int>>& BLcans,
		double zdistance,
		Eigen::VectorXi& usedMarker
		)
{
	int i = 0;
	while (usedMarker(i) != 0 && i < usedMarker.rows())
		i++;
	if (i >= usedMarker.rows())
		throw std::runtime_error("Unexpected calling to pick_closest_BL: all boundary lists have been matched");
	//std::cerr << __func__ << " first i: " << i << endl;
	double min_d = distance(V, objBL, BLcans[i], zdistance);
	size_t retidx = i;
	for (; i < int(BLcans.size()); i++) {
		if (usedMarker(i) > 0)
			continue;
		const std::vector<int>& BL = BLcans[i];
		double d = distance(V, objBL, BL, zdistance);
		if (d < min_d) {
			min_d = d;
			retidx = i;
		}
	}
	usedMarker(retidx) = 1;
	//std::cerr << __func__ << " returns: " << retidx << endl;
	return BLcans[retidx];
}

Eigen::MatrixXi
seal(const Eigen::MatrixXd& V,
     const std::vector<int>& btmBL,
     const std::vector<int>& topBL,
     double zdistance)
{
	Eigen::MatrixXi ret;
	std::vector<Eigen::VectorXi> faces;
#if 0
	auto lambda = [&faces](int bc, int bn, int tc, int tn) {
		std::cerr << "faces: " << bc << ' ' << bn << ' ' << tc << ' ' << tn << endl;
		faces.emplace_back(Eigen::Vector3i(bc, bn, tc));
		faces.emplace_back(Eigen::Vector3i(bn, tn, tc));
	};
#endif
#if 0
	std::cerr << "Sealing\nbtmBL:";
	for (auto vi : btmBL)
		std::cerr << ' ' << vi;
	std::cerr << "\ntopBL:";
	for (auto vi : topBL)
		std::cerr << ' ' << vi;
	std::cerr << endl;
#endif
	double sqzd = zdistance * zdistance;
	// Find the btmBL.front() -> topBL.???
	int topidx = 0;
	Eigen::VectorXd btmV = V.row(btmBL[0]);
	{
		Eigen::VectorXd topV = V.row(topBL[topidx]);
		double min_d = (topV - btmV).squaredNorm() - sqzd;
		for (size_t i = 1; i < topBL.size(); i++) {
			topV = V.row(topBL[i]);
			double d = (topV - btmV).squaredNorm() - sqzd;
			if (d < min_d) {
				min_d = d;
				topidx = int(i);
			}
		}
	}
	int direction = 0; // Uninitialized
	{
		Eigen::VectorXd topV = V.row(topBL[topidx]);
		Eigen::VectorXd topVfwd = V.row(topBL[(topidx + 1)%topBL.size()]);
		Eigen::VectorXd topVbwd = V.row(topBL[(topidx - 1 + topBL.size())%topBL.size()]);
		Eigen::VectorXd btmVfwd = V.row(btmBL[1]);
		//Eigen::VectorXd btmVbwd = V.row(topBL.back());
		Eigen::VectorXd topfwdV = (topVfwd - topV).normalized();
		Eigen::VectorXd topbwdV = (topVbwd - topV).normalized();
		Eigen::VectorXd btmfwdV = (btmVfwd - btmV).normalized();
		if (topfwdV.dot(btmfwdV) > topbwdV.dot(btmfwdV)) {
			// Same direction
			direction = 1;
		} else {
			// Opposite direction
			direction = -1;
		}
		std::cerr << "top V: " << topV.transpose() << endl;
		std::cerr << "top V (fwd): " << topVfwd.transpose() << endl;
		std::cerr << "btm V: " << btmV.transpose() << endl;
		std::cerr << "top fwd V: " << topfwdV.transpose() << endl;
		std::cerr << "top bwd V: " << topbwdV.transpose() << endl;
		std::cerr << "btm fwd V: " << btmfwdV.transpose() << endl;
		std::cerr << "direction: " << direction << endl;
#if 0
		double dfwd = (topVfwd - btmV).squaredNorm() - sqzd;
		double dbwd = (topVbwd - btmV).squaredNorm() - sqzd;
		if (dfwd <= dbwd)
			direction = 1;
		else
			direction = -1;
#endif
	}
	int init_topc = topidx;
	int topc = init_topc;
	int btmc = 0;
#if 0
	for (size_t i = 0; i < btmBL.size(); i++) {
		int topc = topidx + i * direction;
		int topn = topidx + (i + 1) * direction;
		topc = (topc + topBL.size()) % int(topBL.size());
		topn = (topn + topBL.size()) % int(topBL.size());
		int btmc = i;
		int btmn = (i + 1 + btmBL.size()) % int(btmBL.size());
		lambda(btmBL[btmc], btmBL[btmn], topBL[topc], topBL[topn]);
	}
#endif
	double topd = 0;
	double btmd = 0;
	constexpr int epick = 590;
	int counter = 0;
	bool top_stop = false;
	bool btm_stop = false;
	while (faces.size() == 0 || !top_stop || !btm_stop) {
		int topn = (topc + direction + topBL.size()) % topBL.size();
		int btmn = (btmc + 1 + btmBL.size()) % btmBL.size();
		int topcvi = topBL[topc];
		int topnvi = topBL[topn];
		int btmcvi = btmBL[btmc];
		int btmnvi = btmBL[btmn];
		Eigen::VectorXd topcV = V.row(topcvi);
		Eigen::VectorXd topnV = V.row(topnvi);
		double dtopd = (topcV - topnV).norm();
		Eigen::VectorXd btmcV = V.row(btmcvi);
		Eigen::VectorXd btmnV = V.row(btmnvi);
		double dbtmd = (btmcV - btmnV).norm();
		if (!top_stop && (btm_stop || topd + dtopd < btmd + dbtmd)) {
			// pickup topn
			faces.emplace_back(Eigen::Vector3i(topnvi, btmcvi, topcvi)); // Note: we are going to 'flip' bottom above top
			topc = topn;
			topd += dtopd;
			if (topc == init_topc)
				top_stop = true;
		} else {
			// pickup btmn 
			faces.emplace_back(Eigen::Vector3i(btmnvi, btmcvi, topcvi)); // Ditto
			btmc = btmn;
			btmd += dbtmd;
			if (btmc == 0)
				btm_stop = true;
		}
		//std::cerr << "faces " << faces.back().transpose() << std::endl;
		//if (counter++ > epick)
		//	break;
	}
	ret.resize(faces.size(), 3);
	for (size_t i = 0; i < faces.size(); i++)
		ret.row(i) = faces[i];

	return ret;
}

void glue_boundary(const Eigen::MatrixXd& V,
                   const Eigen::VectorXi& btmV,
                   const Eigen::MatrixXi& btmF,
                   const Eigen::VectorXi& topV,
                   const Eigen::MatrixXi& topF,
                   Eigen::MatrixXi& glueF)
{
#if 0
	Eigen::MatrixXi btmE, topE;
	Eigen::VectorXi btmEM, topEM;
	igl::all_edges(btmF, btmE);
	igl::all_edges(topF, topE);
	igl::is_boundary_edge(btmE, btmpF, btmEM);
	igl::is_boundary_edge(topE, topF, topEM);
#endif
	std::vector<std::vector<int>> btmBLs, topBLs;
	std::vector<Eigen::MatrixXi> Fchain;
	igl::boundary_loop(btmF, btmBLs);
	igl::boundary_loop(topF, topBLs);
	
	Eigen::VectorXi usedMarker;
	usedMarker.setZero(topBLs.size(), 1);
	int iter = 0;
	constexpr int ipick = 3;
	for (const auto& btmBL : btmBLs) {
#if 0
		int i = iter++;
		//if (i == 1)
		//	continue;
		if (i != ipick)
			continue ;
		if (i > ipick)
			break;
#endif
		const std::vector<int>& topBL = pick_closest_BL(V, btmBL, topBLs, kTopTheta - kBtmTheta, usedMarker);
		//std::cerr << "usedMarker: " << usedMarker.transpose() << endl;
		Fchain.emplace_back(seal(V, btmBL, topBL, kTopTheta - kBtmTheta));
	}
	size_t nrows = 0;
	for (const auto& F : Fchain)
		nrows += F.rows();
	glueF.resize(nrows, 3);
	nrows = 0;
	for (const auto& F : Fchain) {
		glueF.block(nrows, 0, F.rows(), 3) = F;
		nrows += F.rows();
	}
}

int main(int argc, char* argv[])
{
	int opt;
	string iprefix, ofn("/dev/stdout"), mfn;
	while ((opt = getopt(argc, argv, "i:o:m:")) != -1) {
		switch (opt) {
			case 'i': 
				iprefix = optarg;
				break;
			case 'o':
				ofn = optarg;
				break;
			case 'm':
				mfn = optarg;
				break;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	Eigen::MatrixXd V;
	Eigen::MatrixXi E;
	Eigen::MatrixXi P;
	Eigen::VectorXi EBM;
	try {
		readtet(iprefix, V, E, P, &EBM);
		Eigen::VectorXi btmVI, topVI;
		Eigen::MatrixXi btmF, topF, glueF;
		slice(V, E, P, kBtmTheta, btmVI, btmF);
		slice(V, E, P, kTopTheta, topVI, topF);
		glue_boundary(V, btmVI, btmF, topVI, topF, glueF);
		Eigen::MatrixXd prdcV; // PeRioDiCal Vertices
		Eigen::MatrixXi prdcF; // PeRioDiCal Faces
		std::unordered_map<int, int> old2new;
		geopick(V, {btmF, topF, glueF}, prdcV, prdcF, &old2new);
		//geopick(V, {glueF}, prdcV, prdcF);
#if 1
#pragma omp parallel for
		for (int i = 0; i < prdcV.rows(); i++) {
			if (fpclose(prdcV(i,2), kBtmTheta))
				prdcV(i,2) = kTopTheta + M_PI/2;
		}
#endif
		igl::writeOBJ(ofn, prdcV, prdcF);
		if (!mfn.empty()) {
			std::ofstream fout(mfn);
			fout.exceptions(std::ios::failbit);
			for (const auto& pair : old2new) {
				fout << pair.first << ' ' << pair.second << endl;
			}
		}
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}
	return 0;
}

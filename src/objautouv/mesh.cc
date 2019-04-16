#include "mesh.h"
#include <Eigen/Geometry>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/mat_max.h>
#include <rectpack/MaxRectsBinPackReal.h>
#include <cmath>

struct UVBox {
	int vertex_id [4];
	int vertex_index [2][4];
	int face_id [2];
	Eigen::MatrixXd rel_uv; // also igl standard: vertex per row
	double size_u, size_v;

	double area() const {
		return size_u * size_v;
	}

	void calibrate(Eigen::MatrixXd rel_uv) {
		double mins_0 = rel_uv.col(0).minCoeff();
		double mins_1 = rel_uv.col(1).minCoeff();
		rel_uv.col(0) = rel_uv.col(0).array() - mins_0;
		rel_uv.col(1) = rel_uv.col(1).array() - mins_1;
		Eigen::ArrayXd sizes = rel_uv.colwise().maxCoeff();

		this->rel_uv = rel_uv.block(0,0, rel_uv.rows(), 2);
		size_u = sizes(0);
		size_v = sizes(1);
	}
};

Mesh::Mesh()
{
}

Mesh::~Mesh()
{
}

void Mesh::PairWithLongEdge(bool do_pairing)
{
	boxes.clear();

	Eigen::MatrixXi igl_tta_face;
	igl::triangle_triangle_adjacency(F, igl_tta_face);
	// Translate convention from [0,1] [1,2] [2,3] to
	// [1,2], [2,3], [0,1]
	tta_face.resize(igl_tta_face.rows(), igl_tta_face.cols());
	tta_face.col(0) = igl_tta_face.col(1);
	tta_face.col(1) = igl_tta_face.col(2);
	tta_face.col(2) = igl_tta_face.col(0);
	// std::cerr << "tta_face\n" << tta_face << std::endl;;

	igl::mat_max(el, 2, face_longedge, face_longedge_id);
	// std::cerr << "face long edge\n" << face_longedge << std::endl;;
	// std::cerr << "face long edge id\n" << face_longedge_id << std::endl;;
	adjface_longedge.resize(tta_face.rows());
	for (int i = 0; i < tta_face.rows(); i++) {
		adjface_longedge(i) = tta_face(i, face_longedge_id(i));
	}

	face_pair = Eigen::MatrixXi::Constant(F.rows(), 2, -1);

	if (!do_pairing) {
		for (int f = 0; f < F.rows(); f++) {
			face_pair(boxes.size(), 0) = f;
			boxes.emplace_back(CreateBB(f));
		}
		return ;
	}

	Eigen::VectorXi paired = Eigen::VectorXi::Constant(F.rows(), 1, 0);
	for (int f = 0; f < tta_face.rows(); f++) {
		int other_f = adjface_longedge(f);
		// Criterion 1: ordered (f, other_f) to avoid duplication
		if (other_f < f)
			continue;
		// Criterion 2: circular consisitancy
		if (f != adjface_longedge(other_f))
			continue;
		// Criterion 3: actually saves space
		auto unioned_bb = CreateBB(f, other_f);
		auto bb1 = CreateOptimalBB(f);
		auto bb2 = CreateOptimalBB(other_f);
		if (unioned_bb.area() > bb1.area() + bb2.area())
			continue;
		paired(f) = 1;
		paired(other_f) = 1;
		face_pair(boxes.size(), 0) = f;
		face_pair(boxes.size(), 1) = other_f;
		boxes.emplace_back(unioned_bb);
	}

	// Add unpaired triangles.
	for (int f = 0; f < tta_face.rows(); f++) {
		if (paired(f) == 0) {
			face_pair(boxes.size(), 0) = f;
			boxes.emplace_back(CreateOptimalBB(f));
		}
	}
	std::cerr << "pairing results\n" << face_pair << std::endl;
}

namespace {
Eigen::Matrix<double, 4, 4> makeAffine(const Eigen::Matrix<double, 3, 4>& in)
{
	Eigen::Matrix<double, 4, 4> out;
	out.block<3,4>(0,0) = in;
	out.row(3) << 0, 0, 0, 1;
	return out;
}

auto affinePoint(const Eigen::Matrix<double, 1, 3>& vec3)
{
	Eigen::Vector4d out;
	out << vec3(0), vec3(1), vec3(2), 1.0;
	return out;
}

}

UVBox Mesh::CreateBB(int f, int other_f)
{
	using Eigen::Matrix;
	using Eigen::MatrixXd;
	using Eigen::ArrayXd;
	using Eigen::Vector4d;

	int vi0 = face_longedge_id(f);
	int vi1 = (vi0 + 1) % 3;
	int vi2 = (vi0 + 2) % 3;
	int v0 = F(f, vi0);
	int v1 = F(f, vi1);
	int v2 = F(f, vi2);
	Matrix<double, 3, 4> base = Matrix<double, 3, 4>::Zero();
	base.col(0) = (V.row(v1) - V.row(v0)).normalized();
	base.col(2) = face_normals.row(f);
	base.col(1) = base.col(2).cross(base.col(0));
	base.col(3) = V.row(v1); // Shared vertex of the other_f as origin
	auto tx = makeAffine(base);
	auto itx = tx.inverse();

	UVBox box;
	MatrixXd rel_uv; // temp relative uv
	box.vertex_index[0][0] = vi0;
	box.vertex_index[0][1] = vi1;
	box.vertex_index[0][2] = vi2;
	if (other_f >= 0) {
		int other_vi0 = face_longedge_id(other_f);
		int other_vi1 = (other_vi0 + 1) % 3;
		int other_vi2 = (other_vi0 + 2) % 3;
		int other_v0 = F(other_f, other_vi0);
		int other_v1 = F(other_f, other_vi1);
		int other_v2 = F(other_f, other_vi2);
		Matrix<double, 3, 4> other_b = Matrix<double, 3, 4>::Zero();
		// Other normal
		other_b.col(2) = face_normals.row(other_f);
		// Other X edge
		other_b.col(0) = (V.row(other_v2) - V.row(other_v1)).normalized();
		other_b.col(1) = other_b.col(2).cross(other_b.col(0));
		// But in our vertex
		other_b.col(3) = V.row(v1);

		assert(other_v1 == v2);
		assert(other_v2 == v1);

		Matrix<double, 3, 4> hinge = Matrix<double, 3, 4>::Zero();
		hinge.col(0) = other_b.col(0);
		hinge.col(2) = face_normals.row(f);
		hinge.col(1) = hinge.col(2).cross(hinge.col(0));
		hinge.col(3) = V.row(v1);
		auto hinge_tx = makeAffine(hinge);

		auto other_itx = makeAffine(other_b).inverse();
		Vector4d projected_x = hinge_tx * (other_itx * affinePoint(V.row(other_v0)));

		rel_uv.resize(4, 4);

		rel_uv.row(0) = itx * affinePoint(V.row(v0));
		rel_uv.row(1) = itx * affinePoint(V.row(v1));
		rel_uv.row(2) = itx * projected_x;
		rel_uv.row(3) = itx * affinePoint(V.row(v2));
		box.vertex_id[0] = v0;
		box.vertex_id[1] = v1;
		box.vertex_id[2] = v2;
		box.vertex_id[3] = other_v0;
		box.vertex_index[1][0] = other_vi0;
		box.vertex_index[1][1] = other_vi1;
		box.vertex_index[1][2] = other_vi2;
	} else {
		rel_uv.resize(3, 4);
		rel_uv.row(0) = itx * affinePoint(V.row(v0));
		rel_uv.row(1) = itx * affinePoint(V.row(v1));
		rel_uv.row(2) = itx * affinePoint(V.row(v2));
		box.vertex_id[0] = v0;
		box.vertex_id[1] = v1;
		box.vertex_id[2] = v2;
	}
	box.face_id[0] = f;
	box.face_id[1] = other_f;
#if 0
	// TODO: sancheck: col(2,3) are close to 0.0
	double mins_0 = rel_uv.col(0).minCoeff();
	double mins_1 = rel_uv.col(1).minCoeff();
#if 0
	rel_uv.col(0) += origin(0);
	rel_uv.col(1) += origin(1);
#else
	// rel_uv.block(0, 0, rel_uv.rows(), 2).rowwise() -= origin;
	rel_uv.col(0) = rel_uv.col(0).array() - mins_0;
	rel_uv.col(1) = rel_uv.col(1).array() - mins_1;
#endif
	ArrayXd sizes = rel_uv.colwise().maxCoeff();

	box.rel_uv = rel_uv.block(0,0, rel_uv.rows(), 2);
	box.size_u = sizes(0);
	box.size_v = sizes(1);
#endif
	box.calibrate(rel_uv);

	return box;
}

UVBox Mesh::CreateOptimalBB(int f)
{
	using Eigen::Matrix;
	using Eigen::MatrixXd;
	using Eigen::ArrayXd;
	using Eigen::Vector4d;

	std::vector<UVBox> boxes(3);
	for (int i = 0; i < 3; i++) {
		auto& box = boxes[i];
		int vi0 = i;
		int vi1 = (vi0 + 1) % 3;
		int vi2 = (vi0 + 2) % 3;
		int v0 = F(f, vi0);
		int v1 = F(f, vi1);
		int v2 = F(f, vi2);
		Matrix<double, 3, 4> base = Matrix<double, 3, 4>::Zero();
		base.col(0) = (V.row(v1) - V.row(v0)).normalized();
		base.col(2) = face_normals.row(f);
		base.col(1) = base.col(2).cross(base.col(0));
		base.col(3) = V.row(v1); // Shared vertex of the other_f as origin
		auto tx = makeAffine(base);
		auto itx = tx.inverse();

		box.vertex_index[0][0] = vi0;
		box.vertex_index[0][1] = vi1;
		box.vertex_index[0][2] = vi2;

		MatrixXd rel_uv; // temp relative uv
		rel_uv.resize(3, 4);
		rel_uv.row(0) = itx * affinePoint(V.row(v0));
		rel_uv.row(1) = itx * affinePoint(V.row(v1));
		rel_uv.row(2) = itx * affinePoint(V.row(v2));
		box.vertex_id[0] = v0;
		box.vertex_id[1] = v1;
		box.vertex_id[2] = v2;
		box.face_id[0] = f;
		box.face_id[1] = -1;

		box.calibrate(rel_uv);
	}
	for (int i = 15; i < 360; i += 15) {
		double theta = i / 180.0 * M_PI;
		UVBox box = boxes[0];
		MatrixXd rel_uv = box.rel_uv;
		Eigen::Matrix2d rot;
		rot << std::cos(theta), -std::sin(theta),
		       std::sin(theta),  std::cos(theta);
		rel_uv = (rot * rel_uv.transpose()).transpose();
		box.calibrate(rel_uv);

		boxes.emplace_back(box);
	}
	double min_area = boxes[0].area();
	size_t min_box = 0;
	for (size_t i = 1; i < boxes.size(); i++) {
		if (min_area > boxes[i].area()) {
			min_area = boxes[i].area();
			min_box = i;
		}
	}
	return boxes[min_box];
}


bool in(const Eigen::Vector2d& uv, const rbp::Rect& rect)
{
	return uv(0) >= rect.x && uv(0) <= rect.x + rect.width &&
	       uv(1) >= rect.y && uv(1) <= rect.y + rect.height;
}

void Mesh::Program(int res, double boxw, double boxh, int margin_pix)
{
	assert(boxw * boxh > 0); // boxw and boxh must have the same sign
	double total_area = 0;
	double max_edge = 0.0;
	std::vector<rbp::RectSize> rects_in;
	std::vector<rbp::Rect> rects_out;
	double margin_u = 0.0, margin_v = 0.0;
	if (margin_pix > 0) {
		if (res > 0 && boxw > 0) {
			margin_u = (boxw * margin_pix) / res;
			margin_v = (boxh * margin_pix) / res;
		} else {
			throw std::runtime_error("margin_pix requires pre-defined res, boxw and boxh");
		}
	}
	for (const auto& box :boxes) {
		total_area += box.area();
#if 1
		std::cerr << "Box " << rects_in.size() << " size: " << box.size_u << '\t' << box.size_v << std::endl;
		// std::cerr << "rel_uv: " << box.rel_uv << std::endl;
#endif
		max_edge = std::max(max_edge, box.size_u);
		max_edge = std::max(max_edge, box.size_v);
		rects_in.emplace_back(box.size_u + 2 * margin_u, box.size_v + 2 * margin_v);
		rects_in.back().cookie = reinterpret_cast<void*>(rects_in.size() - 1);
#if 0
		std::cerr << "Box size input: " << rects_in.back().width
		          << '\t' << rects_in.back().height << std::endl;
#endif
	}
	double edges[2];
	bool probe_box_size = true;
	if (boxw > 0) {
		edges[0] = boxw;
		edges[1] = boxh;
		probe_box_size = false;
	} else {
		edges[0] = edges[1] = max_edge;
		probe_box_size = true;
	}
	// edges[0] = edges[1] = std::max(max_edge, std::sqrt(total_area) * 1.501);
	std::cerr << "Init guessed sizes " << edges[0] << '\t' << edges[1] << std::endl;
	rbp::MaxRectsBinPack bin;
	while (rects_out.size() != rects_in.size()) {
#if 0
		for (int axis = 0; axis < 2; axis++) {
			bin.Init(edges[0], edges[1]);
			bin.Insert(rects_in, rects_out, rbp::MaxRectsBinPack::RectBestShortSideFit);
#if 0
			for (const auto& p : rects_out) {
				std::cerr << "pack to (" << p.x << ", " << p.y << ")"
					  << " with size (" << p.width << ", " << p.height << ")"
					  << std::endl;
			}
			std::cerr << "rects io size: " << rects_out.size()
			          << " " << rects_in.size() << std::endl;
#endif
			if (rects_out.size() == rects_in.size()) {
				break;
			} else {
				std::cerr << "Guessed sizes failed: " << edges[0] <<
					  '\t' << edges[1] << std::endl;
			}
			edges[axis] *= 2;
		}
#else
		bin.Init(edges[0], edges[1]);
		// bin.Insert(rects_in, rects_out, rbp::MaxRectsBinPack::RectBestShortSideFit);
		bin.Insert(rects_in, rects_out, rbp::MaxRectsBinPack::RectBestLongSideFit);
		// bin.Insert(rects_in, rects_out, rbp::MaxRectsBinPack::RectBestAreaFit);
		// bin.Insert(rects_in, rects_out, rbp::MaxRectsBinPack::RectBottomLeftRule);
		// bin.Insert(rects_in, rects_out, rbp::MaxRectsBinPack::RectContactPointRule);
		if (rects_out.size() == rects_in.size()) {
			break;
		}
		std::cerr << "Guessed sizes failed: " << edges[0] << '\t' << edges[1] << std::endl;
		if (!probe_box_size) {
			throw std::runtime_error("Fail to fit into the required size of box.");
		}
		edges[0] *= 1.25;
		edges[1] *= 1.25;
#endif
	}
	std::cerr << "Bounding sizes " << edges[0] << '\t' << edges[1] << std::endl;
	for (const auto& rect : rects_out) {
		std::cerr << "\tBox size (" << rect.width <<  ", " << rect.height
		          << ") at (" << rect.x <<  ", " << rect.y << ")\n";
	}

	size_t total_uv = 0;
	for (size_t i = 0; i < face_pair.rows(); i++) {
		if (face_pair(i, 1) >= 0)
			total_uv += 4;
		else if (face_pair(i, 0) >= 0)
			total_uv += 3;
	}
	UV.resize(total_uv, 2);
	FUV.resize(F.rows(), 3);
	size_t uv_iter = 0;
	for (size_t i = 0; i < rects_out.size(); i++) {
		const auto& rect = rects_out[i];
		int rect_in_id = reinterpret_cast<intptr_t>(rect.cookie);
		const auto& box = boxes[rect_in_id];
		Eigen::MatrixXd uv = box.rel_uv;
		if (rect.rotated) {
			// swap u and v
			uv.col(0) = box.rel_uv.col(1);
			uv.col(1) = box.rel_uv.col(0);
		}
		Eigen::Matrix<double, 1, 2> base(rect.x + margin_u, rect.y + margin_v);
		uv.rowwise() += base;
		UV.block(uv_iter, 0, uv.rows(), 2) = uv;
		if (face_pair(rect_in_id, 1) >= 0) {
			FUV(box.face_id[0], box.vertex_index[0][0]) = uv_iter + 0;
			FUV(box.face_id[0], box.vertex_index[0][1]) = uv_iter + 1;
			FUV(box.face_id[0], box.vertex_index[0][2]) = uv_iter + 3;

			FUV(box.face_id[1], box.vertex_index[1][0]) = uv_iter + 2;
			FUV(box.face_id[1], box.vertex_index[1][1]) = uv_iter + 3;
			FUV(box.face_id[1], box.vertex_index[1][2]) = uv_iter + 1;
		} else {
			FUV(box.face_id[0], box.vertex_index[0][0]) = uv_iter + 0;
			FUV(box.face_id[0], box.vertex_index[0][1]) = uv_iter + 1;
			FUV(box.face_id[0], box.vertex_index[0][2]) = uv_iter + 2;
		}
		for (int j = 0; j < uv.rows(); j++) {
			if (!in(uv.row(j).transpose(), rect)) {
				std::cerr << uv.row(j) << " not in rect " << rect_in_id << " ("
					<< rect.width <<  ", " << rect.height
					<< ") at (" << rect.x <<  ", " <<
					rect.y << ") "
					<< (rect.rotated ? "rotated": "not rotated")
					<< "\n";
			}
		}
		uv_iter += uv.rows();
	}
	if (uv_iter != UV.rows()) {
		throw std::runtime_error("San check failed. uv_iter != UV.rows()");
	}
	rec_u_size = rec_v_size = UV.maxCoeff();
	std::cerr << "Recommended sizes " << rec_u_size << '\t' << rec_v_size << std::endl;
	UV.col(0) /= rec_u_size;
	UV.col(1) /= rec_v_size;
#if 0
	std::cerr << "UV\n" << UV << std::endl;
	std::cerr << "FUV\n" << FUV << std::endl;
#endif
}

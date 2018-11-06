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
	Eigen::MatrixXd rel_uv;
	double size_u, size_v;

	double area() const {
		return size_u * size_v;
	};
};

void Mesh::PairWithLongEdge()
{
	Eigen::MatrixXi igl_tta_face;
	igl::triangle_triangle_adjacency(F, igl_tta_face);
	// Translate convention from [0,1] [1,2] [2,3] to
	// [1,2], [2,3], [0,1]
	tta_face.resize(igl_tta_face.rows(), igl_tta_face.cols());
	tta_face.col(0) = igl_tta_face.col(1);
	tta_face.col(1) = igl_tta_face.col(2);
	tta_face.col(2) = igl_tta_face.col(0);

	igl::mat_max(el,2, face_longedge, face_longedge_id);
	for (int i = 0; i < tta_face.rows(); i++) {
		adjface_longedge(i) = tta_face(i, face_longedge_id(i));
	}

	boxes.clear();

	face_pair = Eigen::MatrixXi::Constant(F.rows(), 2, -1);
	Eigen::VectorXi paired = Eigen::VectorXi::Constant(F.rows(), 1, 0);
	for (int f = 0; f < tta_face.rows(); f++) {
		int other_f = adjface_longedge(f);
		// Criterion 1: ordered (f, other_f) to avoid duplication
		if (other_f < f)
			continue;
		// Criterion 2: circular consisitancy
		if (f != adjface_longedge(other_f))
			continue;
		// Criterion 3: actually saved space
		auto unioned_bb = CreateBB(f, other_f);
		auto bb1 = CreateBB(f);
		auto bb2 = CreateBB(other_f);
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
			boxes.emplace_back(CreateBB(f));
			face_pair(boxes.size(), 0) = f;
		}
	}
}

namespace {
Eigen::Matrix<double, 4, 4> makeAffine(const Eigen::Matrix<double, 3, 4>& in)
{
	Eigen::Matrix<double, 4, 4> out;
	out.block<3,4>(0,0) = in;
	out.row(3) << 0, 0, 0, 1;
	return out;
}

Eigen::Matrix<double, 1, 4> affinePoint(const Eigen::Matrix<double, 1, 3>& vec3)
{
	Eigen::Matrix<double, 1, 4> out;
	out << vec3(0), vec3(1), vec3(2), 1.0;
	return out;
}

}

UVBox Mesh::CreateBB(int f, int other_f = -1)
{
	int vi0 = face_longedge_id(f);
	int vi1 = (vi0 + 1) % 3;
	int vi2 = (vi0 + 2) % 3;
	int v0 = F(f, vi0);
	int v1 = F(f, vi1);
	int v2 = F(f, vi2);
	Eigen::Matrix<double, 3, 4> base = Matrix<double, 3, 4>::Zero();
	base.col(0) = (V.row(v1) - V.row(v0)).normalized();
	base.col(2) = face_normals.row(f);
	base.col(1) = base.col(2).cross(base.col(0));
	base.col(3) = V.row(v1); // Shared vertex of the other_f as origin
	auto tx = makeAffine(base);
	auto itx = tx.inverse();

	UVBox box;
	Eigen::MatrixXd rel_uv; // temp relative uv
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
		Eigen::Matrix<double, 3, 4> other_b = Matrix<double, 3, 4>::Zero();
		// Other normal
		other_b.col(2) = face_normals.row(other_f);
		// Other X edge
		other_b.col(0) = (V.row(other_v2) - V.row(other_v1)).normalized();
		other_b.col(1) = other_b.col(2).cross(other_b.col(0));
		// But in our vertex
		other_b.col(3) = V.row(v1);

		assert(other_v1 == v2);
		assert(other_v2 == v1);

		Eigen::Matrix<double, 3, 4> hinge = Matrix<double, 3, 4>::Zero();
		hinge.col(0) = other_b.col(0);
		hinge.col(2) = face_normals.row(f);
		hinge.col(1) = hinge.col(2).cross(hinge.col(0));
		hinge.col(3) = V.row(v1);
		auto hinge_tx = makeAffine(hinge);

		auto other_itx = makeAffine(other_b).inverse();
		Eigen::Vector4d projected_x = hinge_tx * other_itx * V.row(other_f, other_v0);

		rel_uv.resize(4, 4);

		rel_uv.row(0) = itx * affinePoint(V.row(v0));
		rel_uv.row(1) = itx * affinePoint(V.row(v1));
		rel_uv.row(2) = itx * affinePoint(V.row(v2));
		rel_uv.row(3) = itx * affinePoint(projected_x);
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
	// TODO: sancheck: col(2,3) are close to 0.0
	Eigen::ArrayXd origin = rel_uv.colwise().minCoeff();
	rel_uv.col(0) += origin(0);
	rel_uv.col(1) += origin(1);
	Eigen::ArrayXd sizes = rel_uv.colwise().maxCoeff();

	box.rel_uv = rel_uv.block(0,0, rel_uv.rows(), 2);
	box.size_u = sizes(0);
	box.size_v = sizes(1);
}


void Mesh::Program()
{
	double total_area = 0;
	double max_edge = 0.0;
	std::vector<RectSize> rects_in;
	std::vector<Rect> rects_out;
	for (const auto& box :boxes) {
		total_area += box.area();
		max_edge = std::max(max_edge, box.size_u);
		max_edge = std::max(max_edge, box.size_v);
		rects.emplace_back(box.size_u, box.size_v);
	}
	double edges[2];
	edges[0] = edges[1] = std::max(max_edge, std::sqrt(total_area) * 1.125);
	MaxRectsBinPackReal bin;
	while (rects_out.size() != rects_in.size()) {
		for (int axis = 0; axis < 2; axis++) {
			bin.Init(edges[0], edges[1]);
			bin.Insert(rects_in, rects_out, MaxRectsBinPack::RectBestShortSideFit);
			if (rects_out.size() == rects_in.size())
				break;
			edges[axis] *= 2;
		}
	}

	size_t total_uv = 0;
	for (size_t i = 0; i < face_pair.row(); i++) {
		if (face_pair(i, 1) >= 0)
			total_uv += 4;
		else
			total_uv += 3;
	}
	UV.resize(total_uv, 2);
	size_t uv_iter = 0;
	for (size_t i = 0; i < rects_out.size(); i++) {
		const auto& rect = rects_out[i];
		Eigen::MatrixXd uv = boxes[i].rel_uv;
		if (rect.rotated) {
			// swap u and v
			uv.col(0) = boxes[i].rel_uv.col(1);
			uv.col(1) = boxes[i].rel_uv.col(0);
		}
		Eigen::Vector2d base(rect.x, rect.y);
		uv.rowwise() += base;
		UV.block(uv_iter, 0, uv.rows(), 2) = uv;
		FUV(box.face_id[0], box.vertex_index[0][0]) = uv_iter + 0;
		FUV(box.face_id[0], box.vertex_index[0][1]) = uv_iter + 1;
		FUV(box.face_id[0], box.vertex_index[0][2]) = uv_iter + 3;
		if (face_pair(i, 1) >= 0) {
			FUV(box.face_id[1], box.vertex_index[1][0]) = uv_iter + 2;
			FUV(box.face_id[1], box.vertex_index[1][1]) = uv_iter + 3;
			FUV(box.face_id[1], box.vertex_index[1][2]) = uv_iter + 1;
		}
		uv_iter += uv.rows();
	}
}

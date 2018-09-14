#include "cdmodel.h"
#include "scene.h"
#include <iostream>
#include <fcl/fcl.h>
#include <igl/per_face_normals.h>

namespace osr {
struct CDModel::CDModelData {
	using Scalar = StateScalar;
	using Vector3 = fcl::Vector3d;
	typedef fcl::OBBRSS<Scalar> BVType;
	typedef fcl::BVHModel<BVType> Model;
	Model model;

	std::unique_ptr<fcl::Box<Scalar>> bbox;
	/*
	 * fcl::Box assumes the center is at the origin, but this does not
	 * match our case in general, so we need this uncentrializer matrix to
	 * show how to move the centralized bounding box to the correct
	 * position.
	 */
	Transform uncentrializer;

	VMatrix eig_cache_vertices;
	FMatrix eig_cache_findices; // Face INDICES -> findices
	VMatrix eig_cache_fnormals;

	void cache_eig_forms()
	{
		int NV = model.num_vertices;
		int NF = model.num_tris;
		eig_cache_vertices.resize(NV, 3);
		eig_cache_findices.resize(NF, 3);
		for(int i = 0; i < NV; i++) {
			eig_cache_vertices.row(i) = model.vertices[i];
		}
		for(int i = 0; i < NF; i++) {
			eig_cache_findices(i, 0) = model.tri_indices[i][0];
			eig_cache_findices(i, 1) = model.tri_indices[i][1];
			eig_cache_findices(i, 2) = model.tri_indices[i][2];
		}
		igl::per_face_normals(eig_cache_vertices,
		                      eig_cache_findices,
		                      eig_cache_fnormals);
	}
};

CDModel::CDModel(const Scene& scene)
	:model_(new CDModelData)
{
	model_->model.beginModel();
	scene.addToCDModel(*this);
	model_->model.endModel();
	model_->model.computeLocalAABB();
	using BBOX = fcl::Box<CDModelData::Scalar>;
	const auto& aabb = model_->model.aabb_local;
	auto& uncern = model_->uncentrializer;
	uncern.setIdentity();
	uncern.translate(aabb.center());
	std::cerr << "AABB center: " << aabb.center().transpose() << std::endl;
	std::cerr << "AABB size: " << aabb.width() << ' ' << aabb.height() << ' ' << aabb.depth() << std::endl;
	model_->bbox = std::make_unique<BBOX>(aabb.width(),
			aabb.height(), aabb.depth());
	// SAN CHECK
	model_->bbox->computeLocalAABB();
	const auto& scaabb = model_->bbox->aabb_local;
	std::cerr << "SC AABB center: " << scaabb.center().transpose() << std::endl;
	std::cerr << "SC AABB size: " << scaabb.width() << ' ' << scaabb.height() << ' ' << scaabb.depth() << std::endl;
	std::cerr << "SC AABB's should match AABB's" << std::endl;

	// Cache libigl form of (V, F) to CDModelData
	model_->cache_eig_forms();
}

CDModel::~CDModel()
{
}

void CDModel::addVF(const glm::mat4& m,
		const std::vector<Vertex>& verts,
		const std::vector<uint32_t>& indices)
{
	using VType = CDModelData::Vector3;
	std::vector<VType> vertices;
	std::vector<fcl::Triangle> triangles;
	glm::dmat4 mm(m);
	// glm::dmat4 mm(1.0);
#if 1
	for (const auto& vert: verts) {
		glm::dvec4 oldp(vert.position, 1.0);
		glm::dvec4 newp = mm * oldp;
		vertices.emplace_back(newp[0], newp[1], newp[2]);
		// std::cerr << vertices.back().transpose() << '\n';
	}
#else
	vertices.emplace_back(1.0, 0.0, 0.0);
	vertices.emplace_back(0.0, 1.0, 0.0);
	vertices.emplace_back(0.0, 0.0, 1.0);
#endif
#if 1
	triangles.resize(indices.size() / 3);
	for (size_t i = 0; i < triangles.size(); i++) {
		triangles[i].set(indices[3 * i + 0],
			         indices[3 * i + 1],
			         indices[3 * i + 2]);
	}
#else
	triangles.resize(1);
	triangles[0].set(0, 1, 2);
#endif
	std::cerr << "CD First Vertex: " << vertices.front() << std::endl;
	std::cerr << "CD Mode Size: " << vertices.size() << ' ' << triangles.size() << std::endl;

	model_->model.addSubModel(vertices, triangles);
}

bool CDModel::collide(const CDModel& env,
                      const Transform& envTf,
                      const CDModel& rob,
                      const Transform& robTf)
{
	fcl::CollisionRequest<CDModelData::Scalar> req;
	fcl::CollisionResult<CDModelData::Scalar> res;
	size_t ret;

	ret = fcl::collide(&env.model_->model, envTf,
	                   &rob.model_->model, robTf,
	                   req, res);
#if 0
	std::cerr << "Collide with \n" << t1.matrix() << "\nand\n" << t2.matrix() << "\nreturns: " << ret << std::endl;
#endif
	return ret > 0;
}

bool CDModel::collideBB(const CDModel& env,
                        const Transform& envTf,
                        const CDModel& rob,
                        const Transform& robTf)
{
	fcl::CollisionRequest<CDModelData::Scalar> req;
	fcl::CollisionResult<CDModelData::Scalar> res;
#if 1
	size_t ret = fcl::collide(rob.model_->bbox.get(), robTf * rob.model_->uncentrializer,
			env.model_->bbox.get(), envTf * env.model_->uncentrializer,
			req, res);
#else
	size_t ret = fcl::collide(rob.model_->bbox.get(), robTf,
			env.model_->bbox.get(), envTf,
			req, res);
#endif
	return ret > 0;
}


bool
CDModel::collideForDetails(const CDModel& env,
                           const Transform& envTf,
                           const CDModel& rob,
                           const Transform& robTf,
                           Eigen::Matrix<int, -1, 2>& facePairs)
{
	fcl::CollisionRequest<CDModelData::Scalar> req(1UL << 24, true);
	fcl::CollisionResult<CDModelData::Scalar> res;
	size_t ret;
	ret = fcl::collide(&env.model_->model, envTf,
	                   &rob.model_->model, robTf,
	                   req, res);
	std::vector<fcl::Contact<Scalar>> cts;
	res.getContacts(cts);
	facePairs.resize(cts.size(), 2);
	for (size_t i = 0; i < cts.size(); i++) {
		const auto& ct = cts[i];
#if 0
		std::cerr << "CT " << i << " (" << ct.b1 << ", " << ct.b2 << ")"
		          << " Geo 1 OType: " << ct.o1->getObjectType()
			  << " Geo 1 NType: " << ct.o1->getNodeType()
			  << " Geo 2 OType: " << ct.o2->getObjectType()
			  << " Geo 2 NType: " << ct.o2->getNodeType()
			  << std::endl;
#endif
		facePairs.row(i) << ct.b1 , ct.b2;
	}
	return ret > 0;
}


const Eigen::Ref<Eigen::Matrix<CDModel::Scalar, -1, 3>>
CDModel::vertices() const
{
	return model_->eig_cache_vertices;
}

const Eigen::Ref<Eigen::Matrix<int, -1, 3>>
CDModel::faces() const
{
	return model_->eig_cache_findices;
}

CDModel::VMatrix
CDModel::faceNormals(const Eigen::Matrix<int, -1, 1>& fi) const
{
	size_t NF = fi.rows();
	VMatrix ret;
	ret.resize(NF, 3);
#if 0
	const auto& V = model_->eig_cache_vertices;
	const auto& F = model_->eig_cache_findices;
	for (size_t i = 0; i < N; i++) {
		Eigen::Vector3i vi = F.row(fi(i));
		ret.row(i) = (V.row(vi(1)) - V.row(vi(0))).cross(V.row(vi(2)) - V.row(vi(0))).normalized();
	}
#else
	const auto& N = model_->eig_cache_fnormals;
	for (size_t i = 0; i < NF; i++)
		ret.row(i) = N.row(fi(i));
#endif
	return ret;
}

}

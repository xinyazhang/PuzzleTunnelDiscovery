#include "cdmodel.h"
#include "scene.h"
#include <iostream>
#include <fcl/fcl.h>

namespace osr {
struct CDModel::CDModelData {
	using Scalar = double;
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

}

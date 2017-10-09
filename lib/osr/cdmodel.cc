#include "cdmodel.h"
#include "scene.h"
#include <iostream>
#include <fcl/fcl.h>

namespace osr {
struct CDModel::CDModelData {
	using Scalar = float;
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
	model_->bbox = std::make_unique<BBOX>(aabb.width()/2,
			aabb.height()/2, aabb.depth()/2);
}

CDModel::~CDModel()
{
}

void CDModel::addVF(const glm::mat4& m,
		const std::vector<Vertex>& verts,
		const std::vector<uint32_t>& indices)
{
	std::vector<fcl::Vector3f> vertices;
	std::vector<fcl::Triangle> triangles;
	vertices.resize(verts.size());
	std::transform(verts.begin(), verts.end(), vertices.begin(),
			[m](const Vertex& vert) -> fcl::Vector3f {
				glm::vec4 oldp(vert.position, 1.0f);
				glm::vec4 newp = m * oldp;
				return { newp[0],
					 newp[1],
					 newp[2] };
	                } );
	triangles.resize(indices.size() / 3);
	for (size_t i = 0; i < triangles.size(); i++) {
		triangles[i].set(indices[3 * i + 0],
			         indices[3 * i + 1],
			         indices[3 * i + 2]);
	}

	model_->model.addSubModel(vertices, triangles);
}

bool CDModel::collide(const CDModel& env,
                      const Transform& envTf,
                      const CDModel& rob,
                      const Transform& robTf)
{
	fcl::CollisionRequest<CDModelData::Scalar> req;
	fcl::CollisionResult<CDModelData::Scalar> res;

	size_t ret = fcl::collide(&rob.model_->model, robTf,
			&env.model_->model, envTf,
			req, res);
#if 0
	std::cerr << "Collide with \n" << transform.matrix() << "\nreturns: " << ret << std::endl;
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
	size_t ret = fcl::collide(rob.model_->bbox.get(), robTf * rob.model_->uncentrializer,
			env.model_->bbox.get(), envTf * env.model_->uncentrializer,
			req, res);
	return ret > 0;
}

}

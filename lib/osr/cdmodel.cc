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
};

CDModel::CDModel(const Scene& scene)
	:model_(new CDModelData)
{
	model_->model.beginModel();
	scene.addToCDModel(*this);
	model_->model.endModel();
	model_->model.computeLocalAABB();
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
				glm::vec3 newp = m * oldp;
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

}

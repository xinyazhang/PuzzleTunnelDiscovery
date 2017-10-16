#ifndef OSR_CD_MODEL_H
#define OSR_CD_MODEL_H

#include "geometry.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <vector>

namespace osr {
class Scene;
using Transform = Eigen::Transform<double, 3, Eigen::AffineCompact>;

class CDModel {
	struct CDModelData;
	std::unique_ptr<CDModelData> model_;
public:
	CDModel(const Scene& scene);
	~CDModel();

	void addVF(const glm::mat4&,
		const std::vector<Vertex>&,
		const std::vector<uint32_t>& );
	static bool collide(const CDModel& env,
			    const Transform& envTf,
			    const CDModel& rob,
			    const Transform& robTf);
	/*
	 * Collide env vs rob w.r.t. their Bounding Boxes
	 */
	static bool collideBB(const CDModel& env,
			      const Transform& envTf,
			      const CDModel& rob,
			      const Transform& robTf);
};

}

#endif

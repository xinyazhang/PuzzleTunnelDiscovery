#ifndef OSR_CD_MODEL_H
#define OSR_CD_MODEL_H

#include "geometry.h"
#include "osr_state.h"
#include <memory>
#include <vector>
#include <Eigen/Core>

namespace osr {
class Scene;

class CDModel {
	struct CDModelData;
	std::unique_ptr<CDModelData> model_;
public:
	using Scalar = StateScalar;

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

	static bool collideForDetails(
	                    const CDModel& env,
			    const Transform& envTf,
			    const CDModel& rob,
			    const Transform& robTf,
			    Eigen::Matrix<int, -1, 2>& facePairs);

	using VMatrix = Eigen::Matrix<Scalar, -1, 3>;
	using FMatrix = Eigen::Matrix<int, -1, 3>;

	const Eigen::Ref<VMatrix>
	vertices() const;

	const Eigen::Ref<FMatrix>
	faces() const;
};

}

#endif

#ifndef OMPLAUX_SCENE_BOUNDING_BOX_H
#define OMPLAUX_SCENE_BOUNDING_BOX_H

#include "geo.h"
#include "path.h"

namespace omplaux {
	void calculateSceneBoundingBox(const Geo& robot,
				  const Geo& env,
				  const Path& path,
				  double& bbmin,
				  double& bbmax);
}

#endif

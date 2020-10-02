/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
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

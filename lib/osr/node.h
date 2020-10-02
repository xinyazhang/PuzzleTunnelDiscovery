/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef OSR_NODE_H
#define OSR_NODE_H

#include <assimp/scene.h>
#include <iostream>
#include <vector>
#include <memory>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "camera.h"

namespace osr {
class Scene;

struct Node {
	std::vector<std::shared_ptr<Node>> nodes;
	std::vector<uint32_t> meshes;
	glm::mat4 xform;

	Node(aiNode* node);
	virtual ~Node();
};
}

#endif /* end of include guard: NODE_H */

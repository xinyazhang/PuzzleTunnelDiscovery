#ifndef OSR_NODE_H
#define OSR_NODE_H

#include <assimp/scene.h>
#include <iostream>
#include <vector>
#include <memory>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "camera.h"

namespace osr {
class Scene;

class Node {
	friend class Scene;
	std::vector<std::shared_ptr<Node>> nodes;
	std::vector<uint32_t> meshes;
	glm::mat4 xform;

public:
	Node(aiNode* node);
	virtual ~Node();
};
}

#endif /* end of include guard: NODE_H */

#include "scene.h"
#include "node.h"

Node::Node(aiNode* node) {
    // meshes in current nodes
    for (size_t i = 0; i < node->mNumMeshes; i++) {
        meshes.push_back(node->mMeshes[i]);
    }
    // nodes under current node
    for  (size_t i = 0; i < node->mNumChildren; i++) {
        nodes.push_back(new Node(node->mChildren[i]));
    }
    // transform
    aiMatrix4x4 m = node->mTransformation;
    float data[16] = {
        m.a1, m.b1, m.c1, m.d1,
        m.a2, m.b2, m.c2, m.d2,
        m.a3, m.b3, m.c3, m.d3,
        m.a4, m.b4, m.c4, m.d4
    };
    xform = glm::make_mat4(data);
}

Node::~Node() {

}

#include <fstream>
#include "scene.h"

Scene::Scene() {
    // do nothing
}

Scene::Scene(Scene& rhs) {
    xform = rhs.xform;
    root = rhs.root;
    bbox = rhs.bbox;
}

Scene::~Scene() {
    clear();
}

void
Scene::load(std::string filename) {
    assert(std::ifstream(filename.c_str()).good());
    clear();

    using namespace Assimp;
    Assimp::Importer importer;
    uint32_t flags = aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_PreTransformVertices;
    const aiScene* scene = importer.ReadFile(filename, flags);

    std::vector<glm::vec3> meshColors = {
        glm::vec3(1.0, 0.0, 0.0),
        glm::vec3(0.0, 1.0, 0.0),
        glm::vec3(0.0, 0.0, 1.0),
        glm::vec3(1.0, 1.0, 0.0),
        glm::vec3(1.0, 0.0, 1.0),
        glm::vec3(0.0, 1.0, 1.0),
        glm::vec3(0.2, 0.3, 0.6),
        glm::vec3(0.6, 0.0, 0.8),
        glm::vec3(0.8, 0.5, 0.2),
        glm::vec3(0.1, 0.4, 0.7),
        glm::vec3(0.0, 0.7, 0.2),
        glm::vec3(1.0, 0.5, 1.0)
    };

    // generate all meshes
    for (size_t i = 0; i < scene->mNumMeshes; i++) {
        glm::vec3 color = meshColors[i % meshColors.size()];
        meshes.push_back(new Mesh(scene->mMeshes[i], color));
    }

    // construct scene graph
    root = new Node(scene->mRootNode);

    updateBoundingBox(root, glm::mat4());
    center = center / numVertices;
}

void
Scene::updateBoundingBox(Node* node, glm::mat4 m) {
    glm::mat4 xform = m * node->xform;
    for (auto i : node->meshes) {
        Mesh* mesh = meshes[i];
        for (auto vec : mesh->vertices) {
            glm::vec3 v = glm::vec3(xform * glm::vec4(vec.position, 1.0));
            bbox << v;
            numVertices++;
            center += v;
        }
    }
    for (auto child : node->nodes) {
        updateBoundingBox(child, xform);
    }
}

void
Scene::render(GLuint program, Camera& camera, glm::mat4 m) {
    // render(program, camera, m * xform, root);
    for (auto mesh : meshes)
        mesh->render(program, camera, m * xform);
}

void
Scene::render(GLuint program, Camera& camera, glm::mat4 m, Node* node) {
    glm::mat4 xform = m * node->xform;
#if 0
    if (node->meshes.size() > 0)
        std::cout << "matrix: " << std::endl << glm::to_string(xform) << std::endl;
#endif
    for (auto i : node->meshes) {
        Mesh* mesh = meshes[i];
        mesh->render(program, camera, xform);
    }
    for (auto child : node->nodes) {
        render(program, camera, xform, child);
    }
}

void
Scene::clear() {
    xform = glm::mat4();
    center = glm::vec3();
    numVertices = 0;
    bbox = BoundingBox();
    if (root) {
        delete root;
        root = nullptr;
    }
    for (auto mesh : meshes)
        delete mesh;
    meshes.resize(0);
}

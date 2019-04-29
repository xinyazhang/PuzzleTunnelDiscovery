#include "scene.h"
#include "cdmodel.h"
#include <fstream>
#include <glm/gtx/io.hpp>

namespace {
// Code copied from assimpUtil, to ensure we can reproduce the OMPL's center
void extractVerticesAux(const aiScene *scene,
		        const aiNode *node,
			aiMatrix4x4 transform,
                        std::vector<aiVector3D> &vertices)
{
	transform *= node->mTransformation;
	for (unsigned int i = 0 ; i < node->mNumMeshes; ++i)
	{
		const aiMesh* a = scene->mMeshes[node->mMeshes[i]];
		for (unsigned int i = 0 ; i < a->mNumVertices ; ++i)
			vertices.push_back(transform * a->mVertices[i]);
	}

	for (unsigned int n = 0; n < node->mNumChildren; ++n)
		extractVerticesAux(scene, node->mChildren[n], transform, vertices);
}

// Code copied from assimpUtil
void extractVertices(const aiScene *scene, std::vector<aiVector3D> &vertices)
{
    vertices.clear();
    if ((scene != nullptr) && scene->HasMeshes())
        extractVerticesAux(scene, scene->mRootNode, aiMatrix4x4(), vertices);
}

// Code copied from assimpUtil
aiVector3D getSceneCenter(const aiScene *scene)
{
	aiVector3D center;
	std::vector<aiVector3D> vertices;
	extractVertices(scene, vertices);
	center.Set(0, 0, 0);
	for (auto & vertex : vertices)
		center += vertex;
	center /= (float)vertices.size();
	return center;
}

}

namespace osr {
Scene::Scene()
	:xform_(xform_data_)
{
	clear();
}

Scene::Scene(std::shared_ptr<Scene> other)
	:xform_(other->xform_data_), shared_from_(other)
{
	// std::cerr<< "COPY Scene FROM " << other.get() << " TO " << this << std::endl;
	root_ = other->root_;
	bbox_ = other->bbox_;
	meshes_ = other->meshes_;
	center_ = other->center_;
}

Scene::~Scene()
{
	root_.reset();
	meshes_.clear();
}


void Scene::load(std::string filename, const glm::vec3* model_color)
{
	assert(std::ifstream(filename.c_str()).good());
	clear();
	has_vertex_normal_ = true;

	using namespace Assimp;
	Assimp::Importer importer;
#if 0
	uint32_t flags = aiProcess_Triangulate | aiProcess_GenSmoothNormals |
			 aiProcess_FlipUVs | aiProcess_PreTransformVertices;
#endif
	/* Use the same flags to align with OMPL */
#if 0
	uint32_t flags = aiProcess_Triangulate            |
		         aiProcess_JoinIdenticalVertices  |
		         aiProcess_SortByPType            |
		         aiProcess_OptimizeGraph          |
		         aiProcess_OptimizeMeshes;
#else
	// OMPL updates the flags for some reason
	uint32_t flags = aiProcess_GenNormals             |
	                 aiProcess_Triangulate            |
		         aiProcess_JoinIdenticalVertices  |
			 aiProcess_SortByPType            |
			 aiProcess_OptimizeGraph;
#endif
	const aiScene* scene = importer.ReadFile(filename, flags);

	const static std::vector<glm::vec3> meshColors = {
	    glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0),
	    glm::vec3(0.0, 0.0, 1.0), glm::vec3(1.0, 1.0, 0.0),
	    glm::vec3(1.0, 0.0, 1.0), glm::vec3(0.0, 1.0, 1.0),
	    glm::vec3(0.2, 0.3, 0.6), glm::vec3(0.6, 0.0, 0.8),
	    glm::vec3(0.8, 0.5, 0.2), glm::vec3(0.1, 0.4, 0.7),
	    glm::vec3(0.0, 0.7, 0.2), glm::vec3(1.0, 0.5, 1.0)};

	// generate all meshes
	for (size_t i = 0; i < scene->mNumMeshes; i++) {
		glm::vec3 color(1.0);
		if (model_color)
			color = *model_color;
		else
			color = meshColors[i % meshColors.size()];
#if 0
		if (!scene->mMeshes[i]->HasNormals()) {
			throw std::runtime_error(filename + " does not contain per vertex normal");
		}
#else
		has_vertex_normal_ = has_vertex_normal_ && scene->mMeshes[i]->HasNormals();
#endif
		meshes_.emplace_back(new Mesh(scene->mMeshes[i], color));
	}

	// construct scene graph
	root_.reset(new Node(scene->mRootNode));

	center_ = glm::vec3(0.0f);
	vertex_total_number_ = 0;
	face_total_number_ = 0;
	updateBoundingBox(root_.get(), glm::mat4(1.0));
	// center_ = center_ / vertex_total_number_;
	// mean_of_vertices_ = center_;
	auto ompl_center = getSceneCenter(scene);
	mean_of_vertices_[0] = ompl_center[0];
	mean_of_vertices_[1] = ompl_center[1];
	mean_of_vertices_[2] = ompl_center[2];
	center_ = mean_of_vertices_;

	std::cerr.precision(20);
	std::cerr << "[Scene::load] Assimp loaded "
	          << vertex_total_number_ << " vertices "
		  << face_total_number_ << " faces "
		  << " mean of vertices is"
                  << " " << mean_of_vertices_.x
                  << " " << mean_of_vertices_.y
                  << " " << mean_of_vertices_.z
		  << std::endl;
}

void Scene::updateBoundingBox(Node* node, glm::mat4 m)
{
	glm::mat4 xform = m * node->xform;
	for (auto i : node->meshes) {
		auto mesh = meshes_[i];
		for (const auto& vec : mesh->getVertices()) {
#if 0
			glm::vec3 v =
			    glm::vec3(xform * glm::vec4(vec.position, 1.0));
#else
			glm::vec3 v = vec.position;
#endif
			bbox_ << v;
			vertex_total_number_++;
			center_ += v;
		}
		face_total_number_ += mesh->getNumberOfFaces();
	}
	for (auto child : node->nodes) {
		updateBoundingBox(child.get(), xform);
	}
}

void Scene::addToCDModel(CDModel& model) const
{
	auto visitor = [&model, this] (std::shared_ptr<const Mesh> mesh) {
		mesh->addToCDModel(xform_, model);
	};
	visitMesh(visitor);
#if 0
	for (auto mesh : meshes_) {
		mesh->addToCDModel(xform_, model);
	}
#endif
}

bool Scene::hasUV() const
{
	bool has_uv = true;
	auto visitor = [&has_uv] (std::shared_ptr<const Mesh> mesh) {
		has_uv = has_uv && mesh->hasUV();
	};
	return has_uv;
}

void Scene::visitMesh(std::function<void(std::shared_ptr<const Mesh>)> visitor) const
{
	for (auto mesh : meshes_) {
		visitor(mesh);
	}
}

void Scene::moveToCenter()
{
	std::cerr.precision(17);
	std::cerr << "Move by " << -center_.x << ' ' << -center_.y << ' ' << -center_.z << std::endl;
	translate(-center_);
}

void Scene::clear()
{
	xform_ = glm::mat4(1.0);
	center_ = glm::vec3(0.0);
	bbox_ = BoundingBox();
	root_.reset();
	meshes_.clear();
}

std::shared_ptr<const Mesh>
Scene::getUniqueMesh() const
{
	std::shared_ptr<const Mesh> target_mesh(nullptr);
	auto visitor = [&target_mesh](std::shared_ptr<const Mesh> m) {
		if (target_mesh)
			throw std::runtime_error("Scene object has multiple meshes, hence getUniqueMesh failed.");
		target_mesh = m;
	};
	visitMesh(visitor);
	if (!target_mesh)
		throw std::runtime_error("Some Scene object has no mesh, hence getUniqueMesh failed.");
	return target_mesh;
}

}

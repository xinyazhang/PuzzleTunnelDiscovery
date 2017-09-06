#ifndef OFF_SCREEN_RENDERING_TO_H
#define OFF_SCREEN_RENDERING_TO_H

#include <Eigen/Core>
#include "quickgl.h"
#include <string>
#include <ostream>
#include <memory>

namespace osr {
class Scene;

class Renderer {
public:
	Renderer();
	~Renderer();

	void setup();
	void teardown();
	void loadModelFromFile(const std::string& fn);
	void angleModel(float latitude, float longitude);
	void render_depth_to(std::ostream& fout);
	Eigen::VectorXf render_depth_to_buffer();

	int pbufferWidth = 224;
	int pbufferHeight = 224;
	float default_depth = 1000.0f;
private:
	GLuint shaderProgram = 0;
	GLuint vertShader = 0;
	GLuint fragShader = 0;
	GLuint framebufferID = 0;
	GLuint depthbufferID = 0;
	GLuint renderTarget = 0;

	std::unique_ptr<Scene> scene_;
	void render_depth();
};
}

#endif // OFF_SCREEN_RENDERING_TO_H

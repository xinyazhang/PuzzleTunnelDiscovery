#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>

class Camera {
public:
	Camera();
	const glm::mat4& get_view_matrix() const;
	void left_drag(double delta_x, double delta_y);
	void right_drag(double delta_x, double delta_y);
	void middle_drag(double delta_x, double delta_y);
	bool key_pressed(int key);

	glm::vec3 getCenter() const { return center_; }
	const glm::vec3& getCamera() const { return eye_; }
private:
	bool fps_mode_ = false;
	float camera_distance_ = 10.0;
	glm::vec3 look_ = glm::vec3(0.0f, 0.0f, -1.0f);
	glm::vec3 up_ = glm::vec3(0.0f, 1.0, 0.0f);
	glm::vec3 tangent_;
	mutable glm::vec3 eye_ = glm::vec3(0.0f, 0.0f, camera_distance_);
	mutable glm::vec3 center_;
	glm::mat3 orientation_;
	mutable glm::mat4 current_view_matrix_;

	void sync_with_orientation_matrix();
};

#endif

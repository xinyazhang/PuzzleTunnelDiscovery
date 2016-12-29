#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <GLFW/glfw3.h>
#include "camera.h"
#include <glm/gtx/io.hpp>
#include <iostream>

namespace {
	float pan_speed = 0.025f;
	float roll_speed = 0.025f;
	float rotation_speed = 0.025f;
	float zoom_speed = 0.1f;
};

Camera::Camera()
{
	tangent_ = -glm::cross(up_, look_);
	orientation_ = glm::mat3(-tangent_, up_, look_);
	sync_with_orientation_matrix();
}

const glm::mat4& Camera::get_view_matrix() const
{
	if (fps_mode_)
		center_ = eye_ + camera_distance_ * look_;
	else
		eye_ = center_ - camera_distance_ * look_;

	// Get the look direction.
	glm::vec3 z_direction = -glm::normalize(look_);

	// Get tangent direction.
	glm::vec3 x_direction = glm::normalize(tangent_);

	// Get up_direction.
	glm::vec3 y_direction = glm::normalize(up_);

	// Compute the transpose of the rotation matrix.
	glm::mat3 R_T = glm::transpose(glm::mat3(x_direction, y_direction, z_direction));

	// The view matrix has its upper 3x3 values equal to R^T.
	current_view_matrix_ = glm::mat4(R_T);

	// We get the translation to use:  -R^T * eye.
	glm::vec3 t_view = -R_T * eye_;
	current_view_matrix_[3] = glm::vec4(glm::vec3(t_view), 1.0f);

	//std::cout << result << std::endl;

	return current_view_matrix_;
}

void Camera::left_drag(double delta_x, double delta_y)
{
	glm::vec3 mouse_direction = glm::normalize(glm::vec3(delta_x, delta_y, 0.0f));
	glm::vec3 axis = glm::normalize(
			orientation_ * glm::vec3(-mouse_direction.y, mouse_direction.x, 0.0f));
	orientation_ = glm::mat3(glm::rotate(rotation_speed, axis) * glm::mat4(orientation_));
	sync_with_orientation_matrix();
}

void Camera::right_drag(double delta_x, double delta_y)
{
	if (delta_y > 0.0f)
		camera_distance_ += zoom_speed;
	if (delta_y < 0.0f)
		camera_distance_ -= zoom_speed;
	if (camera_distance_ < 0.0f)
		camera_distance_ = 0.0f;
}

void Camera::middle_drag(double delta_x, double delta_y)
{
	glm::vec2 x_axis = glm::vec2(1.0f, 0.0f);
	glm::vec2 y_axis = glm::vec2(0.0f, 1.0f);
	glm::vec2 mouse_direction = glm::normalize(glm::vec2(delta_x, delta_y));
	glm::vec3 pan_direction =
		glm::normalize(mouse_direction.x * tangent_ - mouse_direction.y * up_);

	center_ += pan_speed * pan_direction;
}

bool Camera::key_pressed(int key)
{
	float speed_factor = 1.0;
	switch (key) {
		case GLFW_KEY_C:
			fps_mode_ = !fps_mode_;
			break;
		case GLFW_KEY_LEFT:
			speed_factor = -1.0;
		case GLFW_KEY_RIGHT:
			{ 
				glm::mat3 rotation = glm::mat3(glm::rotate(speed_factor*roll_speed, look_));
				orientation_ = rotation * orientation_;
				sync_with_orientation_matrix();
			}
			break;
		case GLFW_KEY_S:
			speed_factor = -1.0;
		case GLFW_KEY_W:
			if (fps_mode_)
				eye_ += speed_factor * zoom_speed * look_;
			else
				camera_distance_ -= speed_factor * zoom_speed;
			break;
		case GLFW_KEY_A:
			speed_factor = -1.0;
		case GLFW_KEY_D:
			if (fps_mode_)
				eye_ += speed_factor * pan_speed * tangent_;
			else
				center_ += speed_factor * pan_speed * tangent_;
			break;
		case GLFW_KEY_DOWN:
			speed_factor = -1.0;
		case GLFW_KEY_UP:
			if (fps_mode_)
				eye_ += speed_factor*pan_speed * up_;
			else
				center_ += speed_factor*pan_speed * up_;
			break;
		default:
			return false;
	}
	return true;
}

void
Camera::sync_with_orientation_matrix()
{
	tangent_ = -glm::column(orientation_, 0);
	up_ = glm::column(orientation_, 1);
	look_ = glm::column(orientation_, 2);
}

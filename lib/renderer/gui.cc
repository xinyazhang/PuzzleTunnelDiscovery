/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "gui.h"
#include "camera_config.h"
#include <iostream>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/io.hpp>
#include "camera.h"

GUI::GUI(GLFWwindow* window)
	:window_(window), camera_(new Camera)
{
	glfwSetWindowUserPointer(window_, this);
	glfwSetKeyCallback(window_, KeyCallback);
	glfwSetCursorPosCallback(window_, MousePosCallback);
	glfwSetMouseButtonCallback(window_, MouseButtonCallback);

	glfwGetWindowSize(window_, &window_width_, &window_height_);
	float aspect_ = static_cast<float>(window_width_) / window_height_;
	if (orth_proj_) {
		projection_matrix_ = 
		glm::ortho(-100.0f, 100.0f, -100.0f, -100.0f);
	} else {
		projection_matrix_ =
			glm::perspective((float)(kFov * (M_PI / 180.0f)), aspect_, kNear, kFar);
	}
}

GUI::~GUI()
{
}

void GUI::keyCallback(int key, int scancode, int action, int mods)
{
	if (camera_->key_pressed(key))
		return;
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window_, GL_TRUE);
		return ;
	} else if (key == GLFW_KEY_T && action != GLFW_RELEASE) {
		transparent_ = !transparent_;
	}
}

void GUI::mousePosCallback(double mouse_x, double mouse_y)
{
	last_x_ = current_x_;
	last_y_ = current_y_;
	current_x_ = mouse_x;
	current_y_ = window_height_ - mouse_y;
	double delta_x = current_x_ - last_x_;
	double delta_y = current_y_ - last_y_;
	if (sqrt(delta_x * delta_x + delta_y * delta_y) < 1e-15)
		return;
	if (!drag_state_)
		return ;
	if (current_button_ == GLFW_MOUSE_BUTTON_LEFT)
		camera_->left_drag(delta_x, delta_y);
	else if (current_button_ == GLFW_MOUSE_BUTTON_RIGHT)
		camera_->right_drag(delta_x, delta_y);
	else if (current_button_ == GLFW_MOUSE_BUTTON_MIDDLE)
		camera_->middle_drag(delta_x, delta_y);
}

void GUI::mouseButtonCallback(int button, int action, int mods)
{
	drag_state_ = (action == GLFW_PRESS);
	current_button_ = button;
}

void GUI::updateMatrices()
{
	light_position_ = glm::vec4(camera_->getCenter(), 1.0f);
	aspect_ = static_cast<float>(window_width_) / window_height_;
	if (orth_proj_) {
		constexpr float ws = 10.0f;
		projection_matrix_ = 
		glm::ortho(aspect_ * -ws, aspect_ * ws, -ws, ws, 0.0f, 1000.0f);
	} else {
		projection_matrix_ =
		glm::perspective((float)(kFov * (M_PI / 180.0f)), aspect_, kNear, kFar);
	}
	model_matrix_ = glm::mat4(1.0f);
}

MatrixPointers GUI::getMatrixPointers() const
{
	MatrixPointers ret;
	ret.projection = &projection_matrix_[0][0];
	ret.model= &model_matrix_[0][0];
	ret.view = &(camera_->get_view_matrix()[0][0]);
	return ret;
}

glm::vec3 GUI::getCenter() const
{
	return camera_->getCenter();
}

const glm::vec3& GUI::getCamera() const
{
	return camera_->getCamera();
}

void GUI::setCameraDistance(float d)
{
	camera_->setCameraDistance(d);
}

// Delegrate to the actual GUI object.
void GUI::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	GUI* gui = (GUI*)glfwGetWindowUserPointer(window);
	gui->keyCallback(key, scancode, action, mods);
}

void GUI::MousePosCallback(GLFWwindow* window, double mouse_x, double mouse_y)
{
	GUI* gui = (GUI*)glfwGetWindowUserPointer(window);
	gui->mousePosCallback(mouse_x, mouse_y);
}

void GUI::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	GUI* gui = (GUI*)glfwGetWindowUserPointer(window);
	gui->mouseButtonCallback(button, action, mods);
}

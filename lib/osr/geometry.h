/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef OSR_GEOMETRY_H
#define OSR_GEOMETRY_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace osr {
struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
    glm::vec3 normal;
    Vertex(const glm::vec3& p, const glm::vec3& c)
	    :position(p), color(c)
    {
    }
    Vertex(const glm::vec3& p, const glm::vec3& c, const glm::vec3& n)
	    :position(p), color(c), normal(n)
    {
    }
};
}

#endif /* end of include guard: GEOMETRY_H */

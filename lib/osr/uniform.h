/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef OSR_UNIFORM_H
#define OSR_UNIFORM_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace osr {
struct UniformMatrix {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;

    UniformMatrix() {
        model = glm::mat4(1.0);
        view = glm::mat4(1.0);
        proj = glm::mat4(1.0);
    }

    UniformMatrix(UniformMatrix& rhs) {
        model   = rhs.model;
        view    = rhs.view;
        proj    = rhs.proj;
    }
};
}

#endif /* end of include guard: UNIFORM_H */

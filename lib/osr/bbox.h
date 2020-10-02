/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef OSR_BBOX_H
#define OSR_BBOX_H

#include <iostream>
#include <algorithm>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace osr {

class BoundingBox {
public:
    float left, right, top, bottom, front, back;

    BoundingBox() {
        left = right = top = bottom = front = back = 0;
    }

    virtual ~BoundingBox() {

    }

    glm::vec3 center() {
        return glm::vec3(left + right, top + bottom, front + back) / 2.0f;
    }

    float span() const
    {
	    auto spanX = right - left;
	    auto spanY = top - bottom;
	    auto spanZ = front - back;
	    return std::max(std::max(spanX, spanY), spanZ);
    }

    friend BoundingBox& operator<<(BoundingBox& bbox, glm::vec3 v) {
        bbox.right = std::max(v.x, bbox.right);
        bbox.left  = std::min(v.x, bbox.left);

        bbox.top    = std::max(v.y, bbox.top);
        bbox.bottom = std::min(v.y, bbox.bottom);

        bbox.front  = std::max(v.z, bbox.front);
        bbox.back   = std::min(v.z, bbox.back);
        return bbox;
    }

    friend std::ostream& operator<<(std::ostream& out, BoundingBox b) {
        out << "left    :   " << b.left << std::endl
            << "right   :   " << b.right << std::endl
            << "top     :   " << b.top << std::endl
            << "bottom  :   " << b.bottom << std::endl
            << "front   :   " << b.front << std::endl
            << "back    :   " << b.back << std::endl;
        return out;
    }
};

}

#endif /* end of include guard: LAVA_BBOX_H */

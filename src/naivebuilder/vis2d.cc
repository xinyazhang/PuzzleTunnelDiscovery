/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "vis2d.h"

NaiveRenderer* NaiveVisualizer::renderer_;
#if SHOW_AGGPATH
int NaiveVisualizer::aggpath_token;
#endif

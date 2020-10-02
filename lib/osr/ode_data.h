/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef ODE_DATA_H
#define ODE_DATA_H

#include <ode/ode.h>
#include "osr_state.h"

namespace osr {

class CDModel;

/*
 * Wrapper class over ODE routines
 */
struct OdeData {
	dWorldID world;
	dSpaceID space;
	dBodyID body;
	dMass m;

	static void init_ode();

	OdeData(const CDModel& robot);
	~OdeData();

	void setMass(StateScalar mass,
	             const CDModel& robot);
	void setState(const StateVector& q);
	void applyForce(const ArrayOfPoints& fpos,
	                const ArrayOfPoints& fdir,
	                const Eigen::Matrix<StateScalar, -1, 1>& fmag);
	StateVector stepping(StateScalar dt);
	void resetVelocity();
};

}
#endif

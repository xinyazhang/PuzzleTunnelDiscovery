#include "ode_data.h"
#include "cdmodel.h"

namespace osr {

void
OdeData::init_ode()
{
	static bool initialized = false;
	if (initialized)
		return ;
	dInitODE();
	dQuaternion q;
	dQSetIdentity(q);
	if (q[0] != 1.0) {
		throw std::runtime_error("ODE is not using W-first quaterion anymore!");
	}
	initialized = true;
}

OdeData::OdeData(const CDModel& robot)
{
	init_ode();

	world = dWorldCreate();
	space = dHashSpaceCreate(0);
	body = dBodyCreate(world);
	/*
	 * Note: alternatively we can pretend every robot is a solid
	 * sphere, which could simplify the concept and the process.
	 *
	 * But let's try this first.
	 */
	setMass(1.0, robot);
}

OdeData::~OdeData()
{
	dBodyDestroy(body);
	dSpaceDestroy(space);
	dWorldDestroy(world);
}

void
OdeData::setMass(StateScalar mass,
	         const CDModel& robot)
{
#if 0
	auto com = robot.centerOfMass();
	auto MI = robot.inertiaTensor();
	dMassSetParameters(&m, 1.0,
			   com(0), com(1), com(2),
			   MI(0,0), MI(1,1), MI(2,2),
			   MI(0,1), MI(0,2), MI(1,2));
#endif
	dMassSetSphere(&m, mass, 1.0); // mass is density, radius = 1.0
	dBodySetMass(body, &m);
}

void
OdeData::setState(const StateVector& q)
{
	dBodySetPosition(body, q(0), q(1), q(2));
	// Note: ODE also use w-first notation
	dBodySetQuaternion(body, &q(3));
}

void
OdeData::applyForce(const ArrayOfPoints& fpos,
                    const ArrayOfPoints& fdir,
                    const Eigen::Matrix<StateScalar, -1, 1>& fmag)
{
	int N = fpos.rows();
#if 0
	std::cerr << std::endl;
	std::cerr << "----------------------------------------------------" << std::endl;
	std::cerr << "fpos\n " << fpos << std::endl;
	std::cerr << "fdir\n " << fdir << std::endl;
	std::cerr << "fmag\n " << fmag << std::endl;
	std::cerr << "----------------------------------------------------" << std::endl;
	std::cerr << std::endl;
#endif
	for (int i = 0; i < N; i++) {
		Eigen::Matrix<StateScalar, 3, 1> force;
		force = (fmag(i) * fdir.row(i)).transpose();
		if (force.norm() == 0.0)
			continue;
		if (force.hasNaN() || fpos.hasNaN())
			continue;
		dBodyAddForceAtPos(body,
				   force(0), force(1), force(2),
				   fpos(i,0), fpos(i,1), fpos(i,2)
				  );
	}
}

StateVector
OdeData::stepping(StateScalar dt)
{
	dWorldStep(world, dt);
	auto pos = dBodyGetPosition(body);
	auto quat = dBodyGetQuaternion(body);
	StateVector ret;
	ret << pos[0], pos[1], pos[2],
	       quat[0], quat[1], quat[2], quat[3];
	return ret;
}

void
OdeData::resetVelocity()
{
	dBodySetLinearVel(body, 0, 0, 0);
	dBodySetAngularVel(body, 0, 0, 0);
}

}

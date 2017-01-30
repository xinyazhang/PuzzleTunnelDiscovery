#include "EigenCCD.h"
#include <ccd/ccd.h>
#include <ccd/quat.h> // for work with quaternions
#include <iostream>

using std::string;

/** Support function for box */
static void support(const void *obj, const ccd_vec3_t *dir, ccd_vec3_t *vec)
{
	EigenCCD *eic = (EigenCCD *)obj;
	eic->support(dir, vec);
#if 0
	// assume that obj_t is user-defined structure that holds info about
	// object (in this case box: x, y, z, pos, quat - dimensions of box,
	// position and rotation)
	ccd_vec3_t dir;
	ccd_quat_t qinv;

	// apply rotation on direction vector
	ccdVec3Copy(&dir, _dir);
	ccdQuatInvert2(&qinv, &obj->quat);
	ccdQuatRotVec(&dir, &qinv);

	// compute support point in specified direction
	ccdVec3Set(v, ccdSign(ccdVec3X(&dir)) * box->x * CCD_REAL(0.5),
			ccdSign(ccdVec3Y(&dir)) * box->y * CCD_REAL(0.5),
			ccdSign(ccdVec3Z(&dir)) * box->z * CCD_REAL(0.5));

	// transform support point according to position and rotation of object
	ccdQuatRotVec(v, &obj->quat);
	ccdVec3Add(v, &obj->pos);
#endif
}

static void center(const void *obj, ccd_vec3_t *center)
{
	EigenCCD *eic = (EigenCCD *)obj;
	eic->center(center);
}

int main(int argc, char *argv[])
{
	using std::cerr;
	using std::endl;
	string robfn = "../res/simple/mediumstick.obj";
	string envfn = "sc.obj";
	auto rob = EigenCCD::create(robfn);
	auto env = EigenCCD::create(envfn);
	EigenCCD::State state;
	state << 0.07457736701164995, -0.76490862891433631, 7.6299513303455253, -2.4175537217077703, 1.0615147052168636, -1.0185632431560658;
	rob->setTransform(state);
	ccd_t ccd;
	CCD_INIT(&ccd); // initialize ccd_t struct
	ccd_real_t depth;
	ccd_vec3_t dir, pos;

	// set up ccd_t struct
	ccd.support1       = support; // support function for first object
	ccd.support2       = support; // support function for second object
	ccd.center1       = center; // support function for first object
	ccd.center2       = center; // support function for second object
	ccd.max_iterations = 500;     // maximal number of iterations
	ccd.epa_tolerance = 1e-6;

	std::cerr << "rob: " << rob.get() << "\tenv: " << env.get() << std::endl;
	int res = ccdGJKPenetration(rob.get(), env.get(), &ccd, &depth, &dir, &pos);
	if (res == 0) {
		cerr << "Collision point: " << ccdVec3X(&pos) << ' ' << ccdVec3Y(&pos) << ' ' << ccdVec3Z(&pos) << endl;
		cerr << "PD: " << depth << endl;
		cerr << "P Direction: " << ccdVec3X(&dir) << ' ' << ccdVec3Y(&dir) << ' ' << ccdVec3Z(&dir) << endl;
	} else {
		cerr << "Not colliding" << endl;
	}
}

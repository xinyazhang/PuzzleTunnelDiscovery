#include <ode/ode.h>
#include <stdio.h>

// dynamics and collision objects
static dWorldID world;
static dSpaceID space;
static dBodyID body;
// static dGeomID geom;
static dMass m;
static dJointGroupID contactgroup;

// simulation loop
static void simLoop (int pause)
{
    const dReal *pos = nullptr;
    const dReal *R;
    pos = dBodyGetPosition(body);
    // dBodyAddForceAtPos(body, 0, 0, 1, pos[0], pos[1], pos[2] + 0.75);
    dBodyAddForceAtPos(body, 0, 0, 1, pos[0], pos[1] + 0.75, pos[2]);
    dWorldStep(world, 0.01);
    // dBodySetPosition (body,0,0,3);
    pos = dBodyGetPosition(body);
    R = dBodyGetQuaternion(body);
    printf("pos: %f %f %f\n", pos[0], pos[1], pos[2]);
    printf("R: %f %f %f %f\n", R[0], R[1], R[2], R[3]);
}

int main (int argc, char **argv)
{
    dInitODE ();
    // create world
    world = dWorldCreate ();
    space = dHashSpaceCreate (0);
    dWorldSetGravity (world,0,0,-9.8);
    dWorldSetCFM(world, 1e-5);
    // create object
    body = dBodyCreate(world);
    // geom = dCreateSphere(space,0.5);
    dMassSetSphere (&m, 1, 0.5);
    dBodySetMass(body,&m);
    // dGeomSetBody(geom,body);
    // set initial position
    dBodySetPosition (body,0,0,3);
    for (int i = 0; i < 16; i++)
	    simLoop(0);
    // clean up
    dSpaceDestroy (space);
    dWorldDestroy (world);
    dCloseODE();
    return 0;
}

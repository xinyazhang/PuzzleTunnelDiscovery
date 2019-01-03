# Interoperation between our pyOSR and Open Motion Planning Library (OMPL)

## Preface

We are not planning to code up various sampling-based algorithms from scratch.
The practical approach is to reuse existing implementations from OMPL.
Hence the interoperation between the output of pyOSR and OMPL is necessary.

## Problem 1: Configuration Space

The SE(3) configuration space can be decomposed into the rotation and transformation components.

However OMPL has its own convention to match.

### Description

We noticed the following two inconsistencies between OMPL and our representation.

#### Quaternion

OMPL always write quaterion's w coefficient last, while pyOSR usually use $w$ coefficient as the first element.

*Note: Not so sure about OMPL's choice since w is the real part of a quaternion,
and following the convention of writing complex numbers w should also be the first one.*

#### Origin

OMPL chose a different origin of the SE(3) configuration space.
It uses the mean of vertices loaded and processed by [Assimp](http://www.assimp.org/) as the origin of this configuration space,
instead of the origin of the geometry coordinate system.

For example, we loaded a geometry whose mean of vertices is at $(1,2,3)$.
The Configuration $(t,q)$ in OMPL actually generate geometry for collision detection (CD) through the following steps:
1. Translate the original geometry by $-(1,2,3)$
2. Rotate 1 by Quaternion $q$
3. Translate 2 by Vector $t$.

We called this process "centralizing" in the following texts.

To match this behaviour, we also adopted the similar centralizing method in pyOSR.
However we soon realized this method is very sensitive to geometry processing.
For example, if we refined the mesh by tessellating some part of this geometry, the mean of vertices would drift,
and the same configuration numbers would generate a totally different geometry for CD.

The function `UnitWorld.enforceRobotCenter` is introduced to eliminate such sensitives,
and we fixed this point since we remeshed our puzzle for better atlas.
Unfortunately this also means OMPL application now uses a different origin than ours for the remeshed puzzle.
Hence there are two choices for us:
1. Align the output of pyOSR to OMPL, for the remeshed puzzle
2. Align the origin of OMPL to the center we fixed during our experiments.

### Proposal

We have the following methods to address this issue.

#### Method 1
We can introduce a method (maybe called `rebasis`) to pyOSR, and translate all the samples to the new configuration space under OMPL's convention.

* PROS: We probably still need this in the future;
* CONS: It is hard to do sanity check for this approach.

#### Method 2
We can introduce a function similar to `UnitWorld.enforceRobotCenter`, and call it **BEFORE** calling `AppBase::setup()`
*Note: `AppBase` is inherited by `app::SE3RigidBodyPlanning` *.

* PROS: Few sanity check needed. This is much less error-prone than Method 1.
* CONS: Intrusive modifications to OMPL.

### Plan

Try Method 1 really quickly with sanity check through RRT sample injection.
If it doesn't work and failed to be fixed within hours then proceed to Method 2.

### Resolution and Conclusion

We implemented Method 1 at `UnitWorld::translateUnitStateToOMPLState`, and have great success.

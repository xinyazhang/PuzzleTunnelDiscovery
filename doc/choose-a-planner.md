Choose A Planner for our NN-assisted Sampler
---

# Scope Discussion
We are planning to limit our choices within OMPL [Open Motion Planning
Library](http://ompl.kavrakilab.org/).

There might be more advanced algorithms not included in OMPL like
[retraction-based RRT (RRRT)](https://ieeexplore.ieee.org/document/4543785).
Is there any way to avoid implement (potentially) everything ever published?

Note: RRRT claims *It took 4,130.5s. There were 103,121 nodes in the resulting
tree*. We may not have to implement RRRT again.

# Algorithm Discussion

We are trying to exploit the NN as an effective, generalizable sampler. This
means we are trying to apply our approach to different puzzles or the same
puzzle with different rigid transformations. Hence this is more likely to be a
single-query planner. We can remove multi-query from our design goals safely.

At the same time, it makes little sense to figure out the shortest path while
a feasible path is already very difficult to find. Hence we can also remove
optimization from our designs.

The last thing we need to consider is the integration. Assisted by the NN we
can generate a set of samples that are presumably in the narrow tunnel. For
PRM base algorithms the solution is straightforward: just add them
as milestones at the beginning of the algorithm. However for tree-based
algorithm like RRT and its derivatives it is not very clear how to "inject" them
to the tree: an early injection leads to rejections since the narrow tunnel is
unlikely visible to the initial state or the goal state; a late injection may
also lead to rejections due to "sample interferences" (TODO: definition).

Summary: we need a single-query algorithm without optimization, and also
handles sample injection in a reasonable way. Thus we propose Rapid-exploring
Random Forest (RRF). This is a slightly modified version over RRTConnect algorithm by
extending the number of trees from two to arbitrary number of trees (i.e.
forest), while the root nodes of the trees become the initial state, goal
state and sampled states. Additionally RRF maintains the connectivity among
tress, and stops if the initial tree connected to the goal tree directly or
indirectly through other trees.


*Now it's time to modify RRTConnect.cpp in OMPL to implement our RRF algorithm...*

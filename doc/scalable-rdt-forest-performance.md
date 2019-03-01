% Performance Report about Scalable RDT Forest Algorithm

# Background

## Rapidly exploring Dense Tree (RDT) algorithm

RDT algorithm is a straightforward variant from the classical Rapidly exploring Random Tree (RRT) algorithm.

In RRT algorithm, if we cannot connect the newly sampled milestone to the closet milestone in the existing tree, we simply dropped this new milestone.
Unfortunately, this design makes RRT very hard to expand within a narrow tunnel, and its variant RDT was propose to deal with this issue.
Compared with RRT, RDT only makes a small change: while the tree cannot connect to the newly sampled milestone,
it connects to the last valid state returned from the collision detection algorithm instead of simply terminating the current iteration.

Hence we choose RDT instead of RRT as the building block of our scalable forest-based algorithm.

## Scalable RDT Forest

We use a forest based algorithm to address disentanglement puzzles.
The forest initially consists of the initial state, the goal state,
and a set of sampled narrow tunnel configurations as the roots of the trees.
The narrow tunnel configurations are predicted through the following steps:
1. Predict distributions over surfaces of puzzle pieces through one or more deep neural networks
2. Sample a pair of points over the surfaces
3. Apply rigid transformations so that the puzzle pieces "touch" with each other at the sampled surface points, with normals aligned.
4. Enumerate the relative rotations and collect collision free configurations
5. Repeat 2-4 until sufficient number of configurations are collected

To cover all possible narrow tunnels, a large number of configurations needs to be collected.
Even if we apply post processing steps after the sampling, the number will still be considerable.
Therefore our forest algorithm has to be scalable up to thousands of trees.

Here we propose our *Scalable RDT Forest* algorithm. This algorithm consists of four steps:
1. Pre-sample a list of milestones in the configuration space of current problem. We call it *shared list*
2. When running RDT algorithm from each root in the forest, substitute the independently sampled milestones with the ones from the *shared list*
3. Hence, we can detect the connectivity between two roots by checking if their corresponding trees connected to the same milestone in the *shared list*
4. Now we can tell if the initial state connects to the goal state through the disjoint set algorithm with the connectivity information from Step 3
5. Furthermore, the solution path can be generated from the tree structure information (if recorded in Step 2)

In our experiment, this algorithm scaled up to 1220 trees in a single forest, with 4 Mi ($4\times 2^{20}) samples.

# Experiment analysis

## Experiment setup

We are solving the dual puzzle, whose OMPL configuration file is `res/dual/dual-g9.cfg`.

This experiment was performed on HTCondor cluster hosted by department of computer science, UT Austin, in the spring of 2019.
We did not record the processor model, nor put a limit where to run those jobs.
Therefore different performance numbers may be measure from multiple runs.

For performance perspective, the following analysis should be considered as a rough overall throughput measurement,
given our current algorithm implementation, instead of a set of carefully controlled micro-benchmarks.

## Raw data

See `pdsrdt-g9mescreened-4m.csv`

## Analysis

Run `hist_csv.py` and `p2d_csv.py`

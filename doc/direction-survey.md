A collection of things we have and can foresee (Sep 2018).
---

# What we have now

## 2D (Unlikely get published but just for the record)

* Boundary Method (potential-based) to generate navigation function for 0D-2D
* Finite Element Method (charge-based) to generate navigation function for 0D-2D
    - Leaky at corners
    - Charge-based representation may require Fourier basis rather than piecewise
      linear basis
* Collision Obstacle (C-Obs) construction algorithm for 2D-2D
    - Can also construct the dual of C-Obs, i.e. Collision Free (C-Free)
      geometry from a define bounding box.
* Heat kernel method to simulate heat flow on C-Free
    - Also addressed periodicity
    - *If generalized to 3D-3D with Monte Carlo methods, this can be used as a (possibly better) narrow tunnel locator*

## 3D

* C-Obs/C-Free Clearance based method
    - Effectively reimplementing [A Simple Path Non-Existence Algorithm using C-obstacle Query](https://link.springer.com/chapter/10.1007/978-3-540-68405-3_17)
    - Extended to 3D-3D case, but this method itself does not scale well.
* Reinforcement Learning. Partitioned into multiple components.
    - Pre-training of the estimation network: classify two consecutive RGB-D
      images into 12 classes of discrete actions
        + Done with > 98% accuracy.
        + We use the intermediate output of the estimation network as the
          feature extractor.
    - Intrinsic curiosity from the error of predicted feature vector and
      actual feature vector. The feature vector is predicated from the current
      feature vector and the action to take. The feature extractor is shared
      with the estimation network.
        + No success here, the convergence speed is too slow (>100 iterations
            per step) to use this
          method as the memory mechanism.
        + At the same time, joint training of the estimation and prediction
          networks show that we are unable to train a model which can:
            1. Estimate the action accurately 
            2. Predict the feature vector with low L2 error.
        + Possible explanation: the feature vector space is too large for our
          current predictor.
    - Actor-Critic Reinforcement Learning
        + No success here, for some reason we don't know yet this method
          cannot even train a model for a specific 2D puzzle. Possibly due to
          sparse positive rewards.
        + Need to take a look at Atari 2600 games for similar games and see
          how other deal with this problem.
    - Re-implement Deep Q Network and train everything (including the feature
      extractor) from scratch to overfit a known solution.
        + Limited success on 2D and discrete actions
        + Requires a known solution path, but we cannot translate the continuous path to a discrete
          path. The translator always fails after stepping into the narrow
          tunnel. Hence we are forced to use continuous path.
        + Failed on 3D continuous action, due to large training error.
    - *Note: we are gradually moving from "solving from scratch" to "solving
      through imitation" in this section.*
* NN assisted sampler
    - Slightly improved rigid puzzle solver
        + Simplified version of [An Efficient Retraction-based RRT Planner](https://ieeexplore.ieee.org/document/4543785/)
        + Used to find a solution of puzzle more effectively
    - Narrow tunnel segments/vertices locator
        + Segments on the solution path of low visibility vertices 
        + Visibility is calculated through Monte Carlo method, from PRM samples.
        + Helper scripts for distributed execution
    - Deep Sampler
        + Input: RGB-D image
        + Output: delta-state from current state to the nearest tunnel vertex.
        + Evaluation: fairly close but most generated samples are in C-Obs.
        + *WIP: Push samples to C-Free*

### RL Environment for 3D Rigid Disentanglement Puzzles

* Programmatically this requires:
    - State arithmetics
    - Geometry and State normalization
        + Geometry objects may have different scales.
        + All geometries are scaled into a unit cube for generalization.
    - Collision detection (CD)
        + Through [The Flexible Collision Library](https://github.com/flexible-collision-library/fcl)
        + *Could be added: continuous collision detection*
    - State transition with CD
    - GPU Rendering
        + RGB-D output
        + Multi-view
        + SSH friendly

*Note: most of these functions can be achieved with [bullet physics](https://pybullet.org/) or [gazebo](http://gazebosim.org/).
Maybe there are still missing functions there (like CD for concave objects in bullet physics, or State normalization), but the overall framework is done.*

# What we could have in the next few months

* NN assisted sampler
    - *WIP: Push samples to C-Free*
    - Generalization
        + Currently we fixed the camera and the environment
    - Fine turning of the NN
        + Maybe with different architectures
    - A modified classical planner with NN assisted sampler
    - More testing cases
        + Various alpha variants
        + Elk puzzle
            * TODO: polish the elk puzzle geometry?
* *Anything I missed?*
* *More ideas?*

# What we could have in the next year

*Note: brainstorming rather than solid plans*

* RL Environment of Rope-Ring Puzzles
    - Generalize the current RL environment to non-rigid bodies.
    - Proper solution of input mechanism.
        + Puzzle construction
            * "Bent" the ring to form the puzzle
        + Puzzle solution
        + Potential Issues: accuracy
        + Might be classified as AR/VR
    - *Support virtual markers to assist RL?*
* Clearer view about exploiting Quantum mechanics (QM)
    - We need piecewise quadratic to preserve momentum
    - Use QM in collision detection
        + e.g. coplanar cases will be properly defined in QM
* *More ideas?*

# Works with More Engineering Efforts

* ViZDoom: A Doom-based AI research platform for visual reinforcement learning
    - https://ieeexplore.ieee.org/abstract/document/7860433/
    - 2016 IEEE Conference on Computational Intelligence and Games (CIG)
* The Arcade Learning Environment: An Evaluation Platform For General Agents
    - aka ALE or "Atari 2600 games"
    - Journal of Artificial Intelligence Research 47, 2013
    - also as Extended Abstract on IJCAI'15 Proceedings of the 24th International Conference on Artificial Intelligence
        + https://dl.acm.org/citation.cfm?id=2832830
* StarCraft II Learning Environment
    - Only preprint version at [arXiv](https://arxiv.org/abs/1708.04782), Aug 2017
    - DeepMind (and Blizzard) may not be seeking for a formal publication.
* MuJoCo: A physics engine for model-based control
    - https://ieeexplore.ieee.org/abstract/document/6386109/
    - 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems
* Rig animation with a tangible and modular input device
    - http://igl.ethz.ch/projects/rig-animation-input-device/
    - https://dl.acm.org/citation.cfm?id=2925909
    - Research paper about an input device
    - ACM SIGGRAPH 2016

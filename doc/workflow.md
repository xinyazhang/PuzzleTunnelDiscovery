# Solving a Novel Puzzle from Scratch

## Preprocessing the Puzzle

1. ~~Use meshlab to add per-vertex normal to the mesh~~
    + Not necessary for our algorithm. This is a limitation from our current
      pyOSR implementation.
    + Update: the requirement of vertex normal has been relaxed by `5dc6b81c255145b4aa92ba652c4111db9a198ec3`
2. Use TetWild to get the delaunay triangulated surface of the input mesh
3. Use `objautouv` to automatically assign UV coordinates to the mesh vertices
    + Not necessary if the geometry already has UV coordinates

## Step 1: Find a solution from the simplified version

~~1. Open OMPL GUI and prepare the puzzle visually.
    * Note: need to save the configuration to some place
2. cd `demos/SE3RigidBodyPlanning/`
3. Create a new source file from the template Alpha-1.0.cpp
4. Hard code the problem configuration into the new source file
    * Need to pick up the bounding box carefully
5. modify `demos/CMakeLists.txt` to include the new source file
6. Compile and test the binary locally
7. PRM + ReRRT
    + Run PRM locally to collect samples
    + Run ReRRT on condor to collect the solution path
        - It is better to run ReRRT N times for a relatively short period.
8. Visually verify the feasibility of the solution path~~

Now `se3solver.py` should be used to solve this problem.

## Step 2: Prepare the PRM data for Monte-Carlo visibility calculator

### Method 1 (Classical)

This is the original method we used when experimenting with the alpha puzzle

1. Blend the PRM and RRT with `mtblender`
    + PRM data comes from the output file of the binary (specified by an
    	optional argument)
    + RRT data comes from the stdout of the binary
2. Generate the ground truth with `RL/rl-precalcmap.py`
    + Well we do not really need this anymore, since we are not using Q
      learning at all
    + **Make sure the `import aniconf12 as aniconf` in the python file is
      modified to use the novel puzzle**

### Method 2

Just translate the PRM data to an NPZ file, with the following convention:

+ Use 'V' to store the vertices
+ Do NOT translate the vertices to unitary states
+ Make sure w component of the quaternion is placed first.
+ *NOTE: We do not have script for this right now*

### Method 3 (Recommended)

In testing with the "Dual" Puzzle we found PRM failed to generates states that connected to the initial state.
Hence we introduce a mixed strategy that uses the planning data from both PRM
and RRT to estimate the visibility of vertices from the solution path.

1. Run `mapmerger` on RRT planning data and a PRM file, and redirect the
   output to some text file
2. Run `prm2gt.py` on the output of `mapmerger`

## Step 3: Locate the narrow tunnel vertex in the solution path

1. Add the new puzzle definition file (denoted as PUZZLE.py)
2. Modify `condor-visibility-matrix2.py` so it can recognize this puzzle
    + More specifically, you need to import PUZZLE.py in `condor-visibility-matrix2.py`
    + and then add a new entry in the `_name2module` dictionary.
3. Calculate the visibility matrix segments in HTCondor
    + Use `condor-visibility-matrix2.py info` to check the task size
    + and then use `condor-visibility-matrix2.py calc` to compute the segments
      of the visibility matrix
2. Assembly the visibility matrix with `asvm.py`
3. Verify the visibility of vertices in the solution path through
   `tunnel-finder.py info`
4. Pick up the narrow tunnel vertices with `tunnel-finder.py`
    + use `extract` command to pick narrow tunnel vertices automatically
    + or use `pick` command to do it manually, usually after verifying the
      states visually in OMPL GUI.

`tunnel-finder.py` should store the narrow tunnel file to some location.
Keep this file and add the file path to `PUZZLE.py` as `tunnel_v_fn`.

## Step 4: Generate the texture image (a.k.a atlas/chart) for deep learning

*Note: here we use texture image, atlas and chart interchangeably*

0. (IMPORTANT) Modify `condor_touch_configuration.py` so that `PUZZLE.py` is
   imported as aniconf
   + by default we import the alpha puzzle 1.2 as `aniconf`
1. Run `condor_touch_configuration.py run` to sample touch configurations
   + It is recommend to run Step 1,2,3 on a distributed system like HTCondor.
   + Instruction for parallel execution: the total number of tasks (denoted as
     `N1`) should be the multiplication of total number of narrow tunnel vertices (which can
     be found through `condor_touch_configuration.py show`)
2. After finishing 1. Run `condor_touch_configuration.py isect` to get the
   intersecting geometry for each touch configuration.
   + Instruction for parallel execution: the total number of tasks should be
     `N1 * (<Touch Batch Size>/<Geo Batch Size>)` to cover all samples from Step 1.
3. After finishing 2. Run `condor_touch_configuration.py uvproj` to project
   faces of the intersecting geometry to UV coordinates of the coplanar faces
   from Puzzle geometry.
   + Instruction for parallel execution: the total number of tasks should
     follow the rules of Step 2.
4. After finishing 3. Run `condor_touch_configuration.py uvrender` to render
   the surface triangles (in barycentric coordinates) to the atlax.
   + As indicated by the name ("render"), this steps uses OpenGL for the task.
     Hence cannot be run in systems without GPU support. However, headless
     rendering (i.e. GPU without X.org) is supported.
   + Note: `uvrender` uses OpenGL shader to cull non-overlapping regions,
     otherwise the result from `uvproj` may fill the atlas incorrectly.
5. After finishing 4. Run `condor_touch_configuration.py uvmerge` to get the
   unified probability distribution chart (PDC) over the mesh surface.

## Step 5: Train the NN for recognition

1. Copy and rename the merged `.png` file generated from Step 4, and make sure
   the image file has the same prefix with the `.obj` file. Hence pyOSR can
   recognize this texture file.
2. Edit the `create_dataset` function in `hg/datagen.py` to add new dataset(s).
3. Copy, rename and edit the `.cfg` file to use the new dataset in a new
   checkpoint directory.
   * Optionally, change the probability to apply different types of augmentations.
4. Train the network with `python2 hg_train.py newdataset.cfg`
   + `hg_train.py` is symlinked to `src/hg/train_launcher.py`.
5. Copy, rename and edit another `.cfg` file to evaluate the dataset.
   + Add `do_testing` under the `Train` Section.
   + Use a larger `batch_size` for better efficiency. 16-32 is recommended.
   + Change `epoch_size` to determine the number of batches to generate
   + `nEpochs` has no effects in testing.
   + The location of prediction files would be printed when finished.

## Step 6: Use the prediction for sample generation

0. Change the `import .. as aniconf` in `condor_touch_configuration.py` to
   make sure the new dataset is used.
1. Copy the prediction atlas to the working directory with
   `condor_touch_configuration.py useatlas`.
2. Run `condor_touch_configuration.py atlas2prim` to unpack the
   primitive information to the chart.
3. Run `condor_touch_configuration.py sample` to sample C-Free samples
   w.r.t. the weights from atlas.
   * In practice, use `sample_enumaxis` subcommand to sample the C-free in a
     more effective manner

## Step 7: Screen the generated samples

1. Locate the file (denoted as `C.npz` in this step) of C-free key points sampled in the last step.
2. Check the task partition with `condor-visibility-matrix2.py info`.
   + Use `C.npz` as `prm` and `path` arguments
   + Use 'ReTouchQ' as --prmkey and --pathkey
   + Pick up a proper block size so the screening is partition into tasks.
     Consider the number of samples, a negative number may be used to allocate multiple rows to one task.
3. Run `condor-visibility-matrix2.py calc` (preferably on HTCondor)
   + Be sure --puzzleconf is set
4. Use 'asvm.py' to assemble the partition visibility matrix into a single one
5. Use `condor_touch_configuration.py screen --method 0` to merge samples
   according to visibility matrix
6. *TODO*: introduce --method 1 which remove samples that connected to
   multiple disentangled states.

## Step 8: RDT Forest with predefined sample set

1. Under ompl.app, run `se3solver.py presample` to generate the **p**re**d**efined **s**ample set (PDS)
2. Use the screened samples from last step as roots of the trees and run RDT algorithm with `--samset` in parallel over these roots.
3. Run `pds_edge.py` to collect the edges in PDS
   * Must use `ls -v` or equivalent to supply the list of files. `pds_edge.py` does
     not sort file names.
4. *(Optional)* Run `disjoint_set.py` to verify if there is feasible solution
   in a cheaper manner
5. Run `forest_dijkstra.py` to get the NON-OPTIMIZED path from tree 0 (initial tree) to tree 1 (goal tree).

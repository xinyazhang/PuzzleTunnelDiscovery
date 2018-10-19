# Overview

This file documents the different pipelines that may be used in *Accelerated
Disentanglement Through Imitation*. The purpose of these pipelines is to
generate samples (presumably) in narrow tunnel of puzzles in the same family
by training a (or more) neural network over a template puzzle and its
corresponding solution.

# Pipeline 1: *Relocation Regression*

## Key point detection

We use Monte-Carlo method to calculate the visibility of vertices in the known solution path and choose
the vertices with visibility lower than a user-picked threshold $\gamma$ as the key points of this template solution.

### *Terminology: visibility*.
Visible volume of sample `q` is defined as the set of samples from the configuration space
in which all the sample connects to `q` with a predetermined local planner.

For our cases we use linear local planner (linear in Euclidean $R^3$ and
SLERP in $SO(3)$)

Visibility of a sample is defined as the visible volume divided by the
configuration space volume.

## Train a relocation network through regression.

Input: a rendered image from randomly generated configuration `q`.
Ground Truth: the required transformation (translation + rotation in axis
angle) from `q` to the nearest key point.

## Sample generation

We pick a novel puzzle, and feed the image rendered from randomly generated configuration `p`.
Suppose the output from the neural network is `dp`.

The pipeline output is `dp + p`, where `+` is defined as applying the transformation `dp` over `p`.

# Pipeline 2: *Relocation Regression + Classification*

*Overview*: An outstanding drawback of Pipeline 1 is the regression does not work well for
$SO(3)$ elements. In Pipeline 2 we replace the regression part into a
classification-based stage.

## Key point detection

SAME

## Train a relocation network through regression + classification.

We partition `q` into $R^3$ component and $SO(3)$ component. For $R^3$ we
keep using the regression network as Pipeline 1 does. For $SO(3)$ component we
switch to the following architecture.

We train an *Augmented Autoencoder* (AAE) so only $SO(3)$ information is preserved in the
bottleneck feature vector. Afterwards we also collect a codebook of
(Rotation transformation, feature vector) pairs for later use.

*Note here we do not put the randomly generated $SO(3)$ into the codebook but
rather substitute it with the required rotation from the configuration to the
nearest key point.*

## Sample generation

$R^3$ component is generated as described in Pipeline 1.

$SO(3)$ component is done by
1. Generate the feature vector through the encoder part of AAE.
2. Look up the feature vector in the codebook to find the corresponding `dp`.

# Pipeline 3: *Tunnel Constructor Locator Network*

*Overview*: Pipeline 2 suffers from the generalisability (*NOT* tested but we
guessing so). In Pipeline 3 we try to deal with this issue.

## Key point detection

SAME

## Tunnel constructor detection

First we want to detect the crucial parts in the puzzle that defines the narrow
tunnel. We use the following algorithm to achieve this:

1. We find a set of key points
2. For each key point `q`, we calculate the neighboring boundaries $\partial\Omega$
   in collision-free configuration space.
3. We can translate the $\partial\Omega$ back to the colliding primitives in
   the puzzle.
   * Note: we can define this set of primitives as our *tunnel constructor*
       concept, but intuitively these primitives are usually heavily
       fragmented. Hence the next step is introduced.
4. Primitive closure: we project these colliding primitives into the median
   axis, and define tunnel constructor as the set of primitives share the same
   region of median axis as of the colliding primitives.

## Train a relocation network through regression + classification.

Almost the same as Pipeline 2, but this time the ground truth becomes the
Tunnel constructor region.

We also add noises and random patches onto the input image. By doing this we
tried to let AAE only memory the tunnel constructor region.

## Sample generation

SAME as Pipeline 2

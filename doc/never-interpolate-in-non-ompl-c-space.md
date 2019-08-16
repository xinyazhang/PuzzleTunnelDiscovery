% Never Interpolate in Configuration Spaces other than OMPL
% Xinya Zhang

Suppose we are interpolating the OMPL configurations from $(T_0, R_0)$ to $(T_1, R_1)$ as $(T_\tau, R_\tau)$, where $\tau \in [0, 1]$.

The interpolation of rotation component is trivial since in all three configuration spaces (C-Spaces) the rotation component is the same.
However the translation part would be tricky.

In OMPL configuration, we have $T_\tau = \tau T_1 + (1-\tau) T_0$, where $\tau\in[0,1]$.

Suppose the Vanilla configuration space is denoted with tilde like $\tilde{T_0}, \tilde{T_1}$. We can get

$$\tilde{T_0} = T_0 - R_0 O$$
$$\tilde{T_1} = T_1 - R_1 O$$

Here $O$ denotes the center of OMPL geometry.

Now we can see the results from the following paths disagree:

1. Interpolate OMPL keys, and then translate to Vanilla
2. Translate keys to Vanilla, and then interpolate

The first result gives $\tilde{T_\tau} = \tau T_1 + (1-\tau) T_0 - R_\tau O$, and the second gives
$\tilde{T_\tau} = \tau (T_1 - R_1 O) + (1-\tau) (T_0 - R_0 O) = \tau T_1 + (1-\tau) T_0 - (\tau R_1 + (1-\tau) R_0) O$.

These two definitions are equivalent iff $R_\tau \equiv (\tau R_1 + (1-\tau) R_0)$, which is not true.

Since our planners are running in OMPL C-space,
only the trajectory in OMPL C-Space is guaranteed to be valid.

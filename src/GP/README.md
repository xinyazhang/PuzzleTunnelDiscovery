1. Visibility calculation (esp. for HTCondor platform)
        a) condor-visibility-matrix.py: calculate the visibility matrix,
           supports distributed computing.
        b) asvm.py: assembly parts of visibility matrix calculated
           by a)
        c) condor_visibility_mc.py: distributed calculate the visibility
           matrix of collision free samples from:
                i)  numerical methods in RL/rlsampler.py
                ii) saforce based methods by condor_saforce.py
        d) assaforcevm.py: assembly parts of visibility matrix from c)
2. Tunnel Finder
        a) tunnel-vertex-locator.py: locate the tunnel vertices from the 
           visibility matrix from 1.a) and 1.b)
        b) visibility-finder.py: non-distributed version of 1.a)+1.b),
           for all samples
        c) tunnel-segment-finder.py: non-distributed version of 1.a)+1.b),
           only for path vertices.
        d) tunnel-finder: non-distributed version to located tunnels
           from all samples. DO NOT USE IT -- too many false positives.
3. Surface Area Force (saforce)
        a) saforce.py: simulation
        b) sancheck_saforce.py: visualize the saforce process
        c) condor_saforce.py: run on distributed system like HTCondor.

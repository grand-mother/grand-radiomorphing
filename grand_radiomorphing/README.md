**Possible optimisation of radio morphing procedure:**

(1) Currently all traces (of all planes etc) are read-in and scaled accordingly to shower parameters. 
Then the positions are translated, (2) the "clostest" neighbores get found (hardcoded underlying pattern). 
The interpolation for the electric field trace at the desired antenna position is performed.

(1) in scaling.py and (2) in core.py (and interpolation.py)
This is optimal for a huge number of desired antenna positions for which more or less all simulated positions are 
needed for the interpolation.


To optimise the code for fewer desired positions one could think about following procedure:
1) Isometry (Translation) of reference position accordingly new shower parameters
2) Identify the 2 surrounding antenna planes + the 4 "closest" neighbours and only scale their electric field traces
3) Peform the interpolation for the desired positions
It requires a slight restructering of the code, but all elements are already available. 

Whether it is worth it, it is a question of the target of using radio morphing. One could save a lot of CPU if only those 
electric field traces are scaled which positions are actually surrounding a desired one. As an additional time saving step,
one could check whether the required "closest" neighbour position was already "scaled" for a previous desired position. 
But the new procedure would allow to use an even denser antenna grid in teh reference shower since the scaling procedure 
is still the bottleneck regarding optimisition of the method and cost a large fraction of the total run time. 

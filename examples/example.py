'''
    Welcome to RADIO MORPHING
    start the run with: python example.py 
'''



#!/usr/bin/env python
from os.path import split, join, realpath
import sys

# Expand the PYTHONPATH and import the radiomorphing package
root_dir = realpath(join(split(__file__)[0], ".."))
sys.path.append(join(root_dir))
import grand_radiomorphing

def run():
    # Settings of the radiomorphing
    data_dir = join(root_dir, "examples", "data")
    sim_dir = join(data_dir, "GrandEventADetailed2") # folder containing your refernence shower simulations
    out_dir = join(data_dir, "InterpolatedSignals") # folder which will contain radio morphed traces afterwards
    antennas = join(out_dir, "antpos_desired.dat") # list of antenna positions you would like to simulate, stored in out_dir in the best case

    shower = {
        "primary" : "electron",        # primary (electron, pion)
        "energy" : 0.96,               # EeV
        "zenith" : 89.5,               # deg (GRAND frame)
        "azimuth" : 0.,                # deg (GRAND frame)
        "injection_height" : 2000.,    # m (injection height in the local coordinate system)
        "altitude" : 2000. }   # m (alitude with respect to sealevel, not necessarily eqivalent to injection height)

    # Perform the radiomorphing
    grand_radiomorphing.process(sim_dir, shower, antennas, out_dir)

    sys.exit(0)


if __name__ == "__main__":
    run()

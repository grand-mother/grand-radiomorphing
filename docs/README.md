# GRAND Radiomorphing
_Produce radio signals with Radio Morphing_

Welcome to Radio Morphing!

These people made that tool amazing:
W. Carvalho, K. de Vries, O. Martineau, K. Kotera V. Niess, M. Tueros, A. Zilles

Details of the methods can be found in [arXiv:1811:01750](https://arxiv.org/abs/1811.01750).

The usage of Radio Morphing with python2.7 is possible, but discouraged since maintenance of python2.7 will stop soon. 

## Description
This is a preliminary version of the radio morphing Python package.

This Python package allows you to calculates the radio signal of an air shower with desired shower parameters which will be detected by an radio array within a fraction of the time common air-simulation tools will need.

This full parametrisation of the radio signal based on a simulated reference shower was developed for the GRAND sensitivity study, to detect the hot-spots on the world where the trigger-rate for horizontal neutrino-induced air showers is enhanced.

## Installation

_GRAND packages require python3.7. If can be installed from the
[tarball](https://www.python.org/downloads) on Linux or with brew on OSX._

The latest stable version of this package can be installed from [PyPi][PYPI]
using [pip][PIP], e.g. as:
```bash
pip3 install --user grand-radiomorphing
```

Alternatively one can also install the latest development commit directly from
[GitHub][GITHUB], as:
```bash
pip3 install --user git+https://github.com/grand-mother/grand-radiomorphing.git@master
```


### Run the example
To run the example, execute 
`python examples/example.py`

Following information have to be handed over in that file:
The desired shower parameter as the primary type (electron or pion), its primary energy, its injection height and the injection altitude. Both are needed since for very inclined showers and depending on the definition of the coordinate origin they must not be equivalent due to Earth's curvature. 
In addition, one has to hand over the direction of propagation (zenith and azimuth of the shower in [GRAND conventions](https://github.com/grand-mother/simulations/blob/master/GRANDAngularConventions.pdf)).
Be aware of that the coordinate system is defined so that the x-axis is pointing towards the magnetic North and the y-axis to West!

The script will read in the antenna positions and electric field traces of the example reference shower given in `examples/data/GrandEventADetaild2` and write the output electric field traces for the desired antenna positions defined in `antpos_desired.dat` to `examples/data/InterpolatedSignals`

### The example reference shower
The example reference shower is an air shower induced by an electron of an energy of 1EeV and an height of 2000m above sealevel. The propagation direction is given by a zenith of 89.5deg (upward-going) and an azimuth of 0deg.

## Documentation
See the [example.py](examples/example.py) script for an example of usage.

The basis of the calclulation is an simulated reference shower. At the moment just results of ZHAireS simulations can be read in. A direct usage of CoREAS output will be integrated asap.

The internal coordinate system used in radiomorphing is defined the injection point of the shower which is given by (0,0,height). The altitude wrt the sealevel of the injection has to be handed over as well for the scaling part of the method. 
In comparison to ZHAireS or CoREAS simulations, Radio Morphing expects the direction of propagation in  [GRAND conventions](https://github.com/grand-mother/simulations/blob/master/GRANDAngularConventions.pdf) as input.

 
One important aspect is that the magnetic field configuration are hardcoded for the GRAND example location (Ulastai) in [scaling.py](https://github.com/grand-mother/radiomorphing/blob/afc77779bb0acc09e3458e9e5f0c0e68b77456f9/lib/python/grand-radiomorphing/scaling.py#L287-L291). If needed that values (as well as the reference shower) have to be adapted.


### Input overview
to be handed over as in [example.py](https://github.com/grand-mother/radiomorphing/blob/master/examples/example.py)

**set of shower parameters** 
- primary type (electron/pion), 
- energy of the particle inducing the shower in EeV, 
- propagation direction of shower: zenith in deg (GRAND conv) and azimuth in deg (GRAND conv), 
- injection height in m == z component/height wrt sealvel of the injection point in m, used to define the injection position as (0,0,injectionheight) as reference
- altitude == actual height in m of injection above sealevel which can differ from the injectionheight in ase of Earth's curvature and differing original coordinate system for the desired antenna positions

**desired antenna positions**: list of desired antenna positions handed over in x (along North-South, pointing to Magnetic North), y (along East-West), z (vertical, r.t. sealevel), since positions must be given in the reference frame defined by injection point, for example saved like  [antpos_desired.dat](https://github.com/grand-mother/grand-radiomorphing/blob/master/examples/data/InterpolatedSignals/antpos_desired2.dat)
 
**path reference shower**: the reference shower is simulated in star-shape-patterned planes (see folder GrandEventADetailed2, -> ask me for the 16 planes)


### Output
- out_dir: folder in which output shall be stored
- a#.trace: electric field traces for EW, NS, and UP component, time in ns, electric field strength in muV/m, #=ID number of antenna, following the index of antennas in the list-file of the desired antenna positions



### Setting up own study/Producing your own reference shower
To produce the reference shower, antenna position following a star-shape pattern in shower coordinates  (vxB, vxvxB) have to be calculated. Each star-shape pattern forms one plane and should be positioned in a fixed distance to the Xmax of the simulated shower. Several of these planes in different fixed distances to Xmac should be simulated. 
Therefor, one can use [ZHAireS](https://arxiv.org/abs/1107.1189) or [CoREAS](https://arxiv.org/abs/1301.2132v1) as long as the following procedure is respected:

- the resulting electric field traces for each plane should be stored in a single subfolder, while a `MasterIndex`-file shall summarise the information about the parameter set of the shower (in ZHAireS convention!) as well as the names of the subfolder and the distances to Xmax of the each plane. See [example](https://github.com/grand-mother/grand-radiomorphing/blob/master/examples/data/GrandEventADetailed2/MasterIndex).

- electric field traces have to be stored as:

  `a#.trace`: time in ns, E(magnetic North-South) in muV/m, E(East-West) in muV/m, E(Vertical) in muV/m,
  while `#`is the wild card for the antenna ID, starting with 0.
  
- simulated antenna positions have to be stored as:

  `antpos.dat`: x (along North-South, pointing towards magnetic North) in m, y (along East-West) in m, z (height wrt sealevel) in m
  
  Here, the antenna positions should be defined in the reference frame where the injection of the shower takes place at  
  (0m,0m,injectionheight in m)
 
 An example reference shower is given in [GRANDevent](https://github.com/grand-mother/grand-radiomorphing/tree/master/examples/data/GrandEventADetailed2).

 


## Possible future projects
 - hand over magnetic field configuration as parameter
 - include CoREAS output directly as possible input 
 - decouple Askaryan and Geomagnetic component
 

## License

The GRAND software is distributed under the LGPL-3.0 license. See the provided
[`LICENSE`][LICENSE] and [`COPYING.LESSER`][COPYING] files.


[COPYING]: https://github.com/grand-mother/radiomorphing/blob/master/COPYING.LESSER
[GITHUB]: https://github.com/grand-mother/radiomorphing
[LICENSE]: https://github.com/grand-mother/radiomorphing/blob/master/LICENSE
[PIP]: https://pypi.org/project/pip
[PYPI]: https://pypi.org/project/grand-radiomorphing

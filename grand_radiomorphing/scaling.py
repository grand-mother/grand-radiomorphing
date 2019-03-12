'''
Performs the scaling of given electric field traces and the isometry 
of the antenna positions for a reference shower according to the parameters of a target shower
'''
from __future__ import print_function

import os
import sys
import numpy as np
from os.path import split, join, realpath

# Expand the PYTHONPATH and import the radiomorphing package
root_dir = realpath(join(split(__file__)[0], ".."))
sys.path.append(join(root_dir, "grand_radiomorphing"))
import utils
import frame
from frame import UVWGetter, XYZGetter, get_rotation  # , GetUVW
from utils import getCerenkovAngle, load_trace


def _getAirDensity(h):
    '''Returns the air density at a specific height, usng an isothermal model as in ZHAireS

    Parameters:
    ---------
        h: float
            height in meters

    Returns:
    -------
        rho: float
            air density in g/cm3
    '''

    rho_0 = 1.225  # ; % kg/m3
    M = 0.028966  # ;  %kg/mol
    g = 9.81  # ; %ms-2
    T = 288.  # ; % K
    R = 8.32  # ;
    rho = rho_0*np.exp(-g*M*h/(R*T))
    return rho

###################################


def _getXmax(primarytype, energy, zen2):
    '''Returns the average xmax value for the primary in g/cm2

    Parameters:
    ---------
        primarytype: str
            for now it just excepts 'electron' or 'pion'
        energy: float
            energy of the primary given in EeV
        zen2: float
            zenith angle of primary in radian

    Returns:
    -------
      Xmax: float
         shower maximum in g/cm2
    '''

    # type of primary (electron or pion, energy in EeV, zen2 (GRAND) in rad
    if primarytype == 'electron':  # aprroximated by gamma shower
        a = 82.5  # g/cm2
        c = 342.5  # g/cm2
    if primarytype == 'pion':  # aprroximated by proton
        a = 62.5  # g/cm2
        c = 357.5  # g/cm2
    Xmax = a*np.log10(energy*10**6.)+c  # E/EeV* 10**6. to be in TeV

    # print "for  ", primarytype, " of energy ", energy,  "EeV: Xmax in g/cm2 : ", Xmax

    return Xmax  # /abs(np.cos(np.pi-zen2))


def _dist_decay_Xmax(zen2, injh2, Xmax_primary):  # zen2: zenith of target shower
    '''Returns the distance from injection to Xmax along the shower trajectory in m and its height above sealevel in meters

    Parameters:
    ---------
        zen2: float
            zenith angle of primary in radian
        injh2: float
            injectionheight of particle in m
        Xmax_primary: float
            Xmax value in g/cm2

    Returns:
    -------
        h: float
            height of shower maximum above sealevel in meters
        ai: float 
            distance from injection to Xmax along the shower trajectory in meters
    '''

    # % Using isothermal Model as in ZHAireS
    rho_0 = 1.225*0.001  # ; % kg/m3 to 0.001g/cm3: 1g/cm3=1000kg/m3, since X given in g/cm2
    M = 0.028966  # ;  %kg/mol - 1000g/mol
    g = 9.81  # ; %ms-2
    T = 288.  # ; % K
    R = 8.32  # ; J/K/mol , J=kg m2/s2

    hD = injh2
    Xmax_primary = Xmax_primary  # * 10. # g/cm2 to kg/m2: 1g/cm2 = 10kg/m2
    gamma = np.pi-zen2  # counterpart of where it goes to
    Re = 6370949  # m, Earth radius
    X = 0.
    i = 0.
    h = hD
    ai = 0
    step = 10
    while X < Xmax_primary:
        i = i+1
        ai = i*step  # 100. #m
        hi = -Re+np.sqrt(Re**2. + ai**2. + hD**2. + 2.*Re*hD - 2 *
                         ai*np.cos(gamma) * (Re+hD))  # cos(gamma)= + to - at 90dg
        deltah = abs(h-hi)  # (h_i-1 - hi)= delta h
        h = hi  # new height
        X = X + rho_0*np.exp(-g*M*hi/(R*T)) * step * \
            100.  # (deltah*100) s # Xmax in g/cm2, density in g/cm3, h: m->100cm,

    # print "Xmax height in m", h, " distance along axis in m ", ai
    return h, ai  # Xmax_height in m, Xmax_distance along axis in m


def _scalingfactors(E1, az1, zen1, injh1, E2, az2, zen2, injh2, phigeo, thetageo, altitude, primary):
    '''Returns factors to scale the amplitude 

    Parameters:
    ---------
        E1: float
            primary energy of reference shower in EeV
        az1: float
            azimuth angle of primary of reference shower in radian
        zen1: float
            zenith angle of primary of reference shower in radian
        injh1: float
            injectionheight of particle for reference shower in meters
        E2: float
            primary energy of target shower in EeV
        az2: float
            azimuth angle of primary of target shower in radian
        zen2: float
            zenith angle of primary of target shower in radian
        injh2: float
            injectionheight of particle for target shower in meters
        phigeo: float
            angles defining direction of magnetic field    
        thetageo: float
            angles defining direction of magnetic field    
        altitude: float
            usually same as injh2, but there could be exceptions
        primary: str
            primary for target shower, for now it just excepts 'electron' or 'pion' (for now primary of reference fixed to 'electron')

    Returns:
    -------
        kStretch: float
            Streching factor for antenna grid
        kE: float
            scaling factor to account for energy
        kAz: float
            scaling factor to account for geomagnetic angle
        kHeight: float
            scaling factor to account for injection height
    '''

    # print "altitude scaling ", altitude
    # 2: target shower, 1: generic shower
    # Energy scaling
    # % Energy
    kE = E2/E1  # both in 1e18eV

    # Azimuth scaling
    # % Azimuth
    Bgeo = [np.cos(phigeo)*np.sin(thetageo), np.sin(phigeo)
            * np.sin(thetageo), np.cos(thetageo)]
    vref = [np.cos(az1)*np.sin(zen1), np.sin(az1)*np.sin(zen1), np.cos(zen1)]
    vxB_ref = np.cross(vref, Bgeo)
    vxB_ref = np.linalg.norm(
        vxB_ref)/(np.linalg.norm(vref)*np.linalg.norm(Bgeo))
    v = [np.cos(az2)*np.sin(zen2), np.sin(az2)*np.sin(zen2), np.cos(zen2)]
    vxB = np.cross(v, Bgeo)
    vxB = np.linalg.norm(vxB)/(np.linalg.norm(v)*np.linalg.norm(Bgeo))
    kAz = vxB/vxB_ref

    h_ref = injh1
    h = altitude  # injh2 # actual altitude wrt sealevel at decay position of target position
    # %############## Height+Zenith, distance injection point to xmax rougly 8000m
    primary1 = 'electron'
    # approximation based on values from plots for gamma (=e) and protons (=pi) # g/cm2
    Xmax_primary1 = _getXmax(primary1, E1, zen1)
    Xmax_height1, Xmax_distance1 = _dist_decay_Xmax(zen1, injh1, Xmax_primary1)

    # hx_ref = h_ref+Xmax_hor1*np.tan(0.5*np.pi-zen1) #   % Height at reference shower Xmax
    # % Height at reference shower Xmax
    hx_ref = h_ref+Xmax_distance1*np.sin(0.5*np.pi-zen1)
    ac_ref = getCerenkovAngle(hx_ref)
    rho_ref = _getAirDensity(hx_ref)

    # approximation based on values from plots for gamma (=e) and protons (=pi) # g/cm2
    Xmax_primary2 = _getXmax(primary, E2, zen2)
    Xmax_height2, Xmax_distance2 = _dist_decay_Xmax(zen2, injh2, Xmax_primary2)
    # hx = h+Xmax_hor2*np.tan(0.5*np.pi-zen2)#   % Height at target shower Xmax
    # % Height at target shower Xmax
    hx = h+Xmax_distance2*np.sin(0.5*np.pi-zen2)

    ac = getCerenkovAngle(hx)
    rho = _getAirDensity(hx)
    kStretch = float(ac)/float(ac_ref)  # % Stretch factor for the antenna grid
    kRho = np.sqrt(rho_ref/rho)
    kHeight = kRho/kStretch
    kAmp = kE*kAz*kHeight

    # print 'kStretch ', kStretch, ' kAmp ', kAmp,  ' kE ', kE, ' KAz ', kAz, ' kHeight ', kHeight

    return kStretch, kE, kAz, kHeight


def _scalingpulse(dist1, E1, az1, zen1, injh1, E2, az2, zen2, injh2, primary, phigeo, thetageo, l,  positions, path, altitude):
    '''Returns factors to scale the amplitude 

    Parameters:
    ---------
        dist1: float
            distance of plane with respect to Xmax for the reference shower in meters
        E1: float
            primary energy of reference shower in EeV
        az1: float
            azimuth angle of primary of reference shower in radian
        zen1: float
            zenith angle of primary of reference shower in radian
        injh1: float
            injectionheight of particle for reference shower in meters
        dist2: float
            distance of plane with respect to Xmax for the target shower in meters
        E2: float
            primary energy of target shower in EeV
        az2: float
            azimuth angle of primary of target shower in radian
        zen2: float
            zenith angle of primary of target shower in radian
        injh2: float
            injectionheight of particle for target shower in meters
        primary: str
            primary for target shower, for now it just excepts 'electron' or 'pion', (for now primary of reference fixed to 'electron')
        phigeo: float
            angles defining direction of mangetic field   
        thetageo: float
            angles defining direction of mangetic field    
        positions: numpy array
            list of antenna positions in the given plane
        path: str
            path to folder containing the files of the reference shower
        altitude: float
            usually same as injh2, but there could be exceptions


    Returns:
    -------
        txt1: numpy array
            scaled electric field traces
        stretch2[l]: numpy array
            antenna position after isometry
    '''


# SCALING factorss
    kStretch, kE, kAz, kHeight = _scalingfactors(
        E1, az1, zen1, injh1, E2, az2, zen2, injh2, phigeo, thetageo, altitude, primary)
    kAmp = kE*kAz*kHeight
    if l == 0: # just print them for the first position
        print('kStretch ', kStretch, ' kAmp ', kAmp,  ' kE ',
              kE, ' KAz ', kAz, ' kHeight ', kHeight)

  ###############################################################################
  # scaling electric fields amplitude
  ################################################

    try:
        # read in full traces of antenna l: 0:time in ns, 1,2,3:  efield
        txt1 = load_trace(path, l)
    except IOError:
        print("antenna ID ", str(int(l)), " file doesn't exist")
        sys.exit()

    # rotation matrix
    # Convert efield to shower coordinates to apply the scaling
    # R = get_rotation(zen1, az1, phigeo, thetageo)# original
    # EshowerA = np.dot(txt1[:,1:], R) # original

    # Sciling, kHeight includes 1/kStretch
    #EshowerA.T[0] *= kE * kHeight
    #EshowerA.T[1] *= kE * kAz * kHeight
    #EshowerA.T[2] *= kE * kHeight

    # Backtrafo of efield from shower coord ( after scaling and/or stretching using the target angles
    # Rt = get_rotation(zen2, az2, phigeo, thetageo).T # original
    # v2 = Rt[:,0]# original
    # txt1[:,1:] = np.dot(EshowerA, Rt)# original
    ##############


# slow way
    inc = thetageo

    az = az1
    zen = zen1
    B = np.array([np.cos(phigeo)*np.sin(inc),
                  np.sin(phigeo)*np.sin(inc), np.cos(inc)])
    B = B/np.linalg.norm(B)

    v = np.array([np.cos(az)*np.sin(zen), np.sin(az)*np.sin(zen), np.cos(zen)])
    v = v/np.linalg.norm(v)
    # print v

    # np.array([v[1]*B[2]-v[2]*B[1],v[2]*B[0]-v[0]*B[2],v[0]*B[1]-v[1]*B[0]]) # crossproduct
    vxB = np.cross(v, B)
    vxB = vxB/np.linalg.norm(vxB)
    # np.array([v[1]*vxB[2]-v[2]*vxB[1],v[2]*vxB[0]-v[0]*vxB[2],v[0]*vxB[1]-v[1]*vxB[0]])# crossproduct
    vxvxB = np.cross(v, vxB)
    vxvxB = vxvxB/np.linalg.norm(vxvxB)

    # rotation to showeframe
    EshowerA = np.zeros([len(txt1.T[1]), 3])
    EshowerA.T[0] = txt1.T[1] * v[0] + txt1.T[2]*v[1] + txt1.T[3]*v[2]
    EshowerA.T[1] = txt1.T[1] * vxB[0] + txt1.T[2]*vxB[1] + txt1.T[3]*vxB[2]
    EshowerA.T[2] = txt1.T[1] * vxvxB[0] + \
        txt1.T[2]*vxvxB[1] + txt1.T[3]*vxvxB[2]

    # Scaling, kHeight includes 1/kStretch
    EshowerA.T[0] *= kE * kHeight
    EshowerA.T[1] *= kE * kAz * kHeight
    EshowerA.T[2] *= kE * kHeight

    # print "az2, zen2 ", az2, zen2

    # angles target shower
    v2 = np.array([np.cos(az2)*np.sin(zen2), np.sin(az2)
                   * np.sin(zen2), np.cos(zen2)])
    v2 = v2/np.linalg.norm(v2)
    # np.array([v[1]*B[2]-v[2]*B[1],v[2]*B[0]-v[0]*B[2],v[0]*B[1]-v[1]*B[0]]) # crossproduct
    vxB2 = np.cross(v2, B)
    vxB2 = vxB2/np.linalg.norm(vxB2)
    # np.array([v[1]*vxB[2]-v[2]*vxB[1],v[2]*vxB[0]-v[0]*vxB[2],v[0]*vxB[1]-v[1]*vxB[0]])# crossproduct
    vxvxB2 = np.cross(v2, vxB2)
    vxvxB2 = vxvxB2/np.linalg.norm(vxvxB2)

    # Backtrafo of efield from shower coord (1,2,3) after scaling 
    # and/or stretching using the target angles
    txt1.T[1] = EshowerA.T[0] * v2[0] + EshowerA.T[1] * \
        vxB2[0] + EshowerA.T[2]*vxvxB2[0]
    txt1.T[2] = EshowerA.T[0] * v2[1] + EshowerA.T[1] * \
        vxB2[1] + EshowerA.T[2]*vxvxB2[1]
    txt1.T[3] = EshowerA.T[0] * v2[2] + EshowerA.T[1] * \
        vxB2[2] + EshowerA.T[2]*vxvxB2[2]


###############################
# stretching of positions
###############################

    # default parametes of star shape simulation
    angles = 8  # 8 rays in start shape pattern hard coded
    rings = int(len(positions[:, 1])/angles)
    beta = (360./angles)/180.*np.pi


################################
  # Calculating the new stretched antenna positions in the star shape
    offinz = np.mean(positions[:, 2])
    offiny = np.mean(positions[:, 1])
    offinx = np.mean(positions[:, 0])
    pos = np.zeros([len(positions[:, 1]), 3])

    # rotate into shower coordinates for preparation to get the 
    # strechted antenna position to compare to
    GetUVW = UVWGetter(offinx, offiny, offinz, zen1, az1, phigeo, thetageo)
    for i in np.arange(0, len(positions[:, 1])):
        pos[i, :] = GetUVW(positions[i, :], )

    # substitue pos by pos and add if condition here
    if kStretch != 1.:
        r = np.linalg.norm(pos[6] - pos[5])  # first rays are formed
        step = 1
        # print r
        # get the distance between the antenna
        # first rings are formed
        if np.linalg.norm(pos[6]-pos[5]) != 0.5*np.linalg.norm(pos[7]-pos[5]):
            r = np.linalg.norm(pos[16] - pos[8])
            step = 8
            # print r

        r = r*kStretch  # just STRETCHING the radius of the rings

        # default parametes of star shape simulation
        #rings = 15
        #angles= 8
        beta = (360./angles)/180.*np.pi

        # first rays are formed, depends on how star shapepattern was set up
        if np.linalg.norm(pos[6]-pos[5]) == 0.5*np.linalg.norm(pos[7]-pos[5]):
            for n in range(0, angles):  # second rings
                for m in range(0, rings):  # first rays
                    pos[n*rings+m, 1] = (m+1)*r * np.cos(n*beta)
                    pos[n*rings+m, 2] = (m+1)*r * np.sin(n*beta)
                    pos[n*rings+m, 0] = 0.

        # first rings are formed
        if np.linalg.norm(pos[6]-pos[5]) != 0.5*np.linalg.norm(pos[7]-pos[5]):
            for m in range(0, rings):  # sencond rays
                for n in range(0, angles):  # first rings

                    pos[m*angles+n, 1] = (m+1)*r * np.cos(n*beta)  # vxB
                    pos[m*angles+n, 2] = (m+1)*r * np.sin(n*beta)  # vxvxB
                    pos[m*angles+n, 0] = 0.  # along v

# CALCULATION OF NEW POSITION VECTOR

    # the new/target position vector of the star shape plane
    # approximation based on values from plots for gamma (=e) and protons (=pi) # g/cm2
    Xmax_primary = _getXmax(primary, E2, zen2)
    #print("xmax value " , Xmax_primary)
    # 8000.# d_prime: distance from decay point to Xmax
    Xmax_height, Xmax_distance = _dist_decay_Xmax(zen2, injh2, Xmax_primary)
    # ==: decay position as defined in zhaires sim, from DANTOn files
    decay = np.array([0., 0., injh2])

    # new position vector:
    # x2= decay - v2 * (Xmax_distance+ dist1) # to account for going from Zhaires to GRAND conv
    # to account for going from Zhaires to GRAND conv
    x2 = decay + v2 * (Xmax_distance + dist1)

  # Backtrafo to XYZ
    # Now the new 'stretched' positions are calculated in the xyz components, backrotation
    stretch2 = np.zeros([len(pos[:, 1]), 3])

    # backtrafo of positions
    GetXYZ = XYZGetter(x2[0], x2[1], x2[2], zen2, az2, phigeo, thetageo)
    for m in range(0, len(pos[:, 1])):
        stretch2[m, :] = GetXYZ(pos[m])

    # print ' scaling done , positions ', stretch2[l]
    return txt1, stretch2[l]


################################################################################

def _scale_run(sim_dir, run, primary, E1, zen1, az1, injh1, dist1, E2, zen2, az2, injh2, altitude):
    """Scale the simulated traces of a run to the shower parameters

    Parameters:
    ---------

        E1: float
            primary energy of reference shower in EeV
        az1: float
            azimuth angle of primary of reference shower in radian
        zen1: float
            zenith angle of primary of reference shower in radian
        injh1: float
            injectionheight of particle for reference shower in meters
        dist2: float
            distance of plane with respect to Xmax for the target shower in meters
        E2: float
            primary energy of target shower in EeV
        az2: float
            azimuth angle of primary of target shower in radian
        zen2: float
            zenith angle of primary of target shower in radian
        injh2: float
            injectionheight of particle for target shower in meters
        altitude: float
            usually same as injh2, but there could be exceptions

    Returns:
    ---------
    stores the scaled electric field traces and the stretched antenna positions after the isometry for the target shower

    """

    # TODO: implement the magnetic field strength as an argument
    phigeo = 2.72*np.pi / 180.  # 182.66#; (ie pointing 2.66 degrees East from full
    # North) from simulations inputfile %
    # In both EVA & Zhaires, North = magnetic North
    thetageo = (180.-27.05)*np.pi/180.  # 152.95*np.pi/180. #27.05*np.pi/180.
    # (pointing down)-62.95

    path = os.path.join(sim_dir, run)  # Path to the simulation run which
    # shall later be rescaled or so

    def print_parameters(E, dist, zen, az, injh):
        """Formated print of the shower parameter values
        """
        parameters = (
            "Energy: {:.2E} EeV".format(E),
            "distance from Xmax: {:.2f} m".format(dist),
            "zenith: {:.2f} deg".format(zen),
            "azimuth: {:.2f} deg".format(az),
            "height: {:.2f} m".format(injh))
        print(", ".join(parameters))

    if dist1 == 4000.:  # to reduce printout
        # Print the reference parameter values
        # print("# Reference shower", run)
        print_parameters(E1, dist1, zen1, az1, injh1)

        # Print the target parameter values
        #print("Target shower parameters")
        print_parameters(E2, dist1, zen2, az2, injh2)

    # Convert the angles from degrees to radians
    zen1, az1, zen2, az2 = map(np.deg2rad, (zen1, az1, zen2, az2))

    # read-in positions of reference shower
    posfile = os.path.join(path, "antpos.dat")
    positions = np.loadtxt(posfile)
    pos_new = np.zeros(positions.shape)

    ############################################################################

    # Create the output directory if it doesnt exist
    directory = os.path.join(sim_dir, "scaled_" + run)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # print(directory)

    end = positions.shape[0]
    # loop over all antenna positions, outer positions should be skipped
    # since then the interpolation doesnt work anymore (4 neighbours needed
    for l in np.arange(0, end):
        # always hand over all need parameters,1 3D pulse, and all antenna
        # positions
        txt3, pos_new[l, :] = _scalingpulse(dist1, E1, az1, zen1, injh1, E2, az2,
                                            zen2, injh2, primary, phigeo,
                                            thetageo, l,  positions, path, altitude)

        # Writing to file for later use
        name3 = os.path.join(directory, "a{:}.trace".format(l))
        with open(name3, "w+") as FILE:
            for i in range(0, len(txt3.T[0])):
                args = (txt3.T[0, i], txt3.T[1, i], txt3.T[2, i], txt3.T[3, i])
                try:
                    print("%.2f	%1.3e	%1.3e	%1.3e" % args, end='\n', file=FILE)
                except SyntaxError:
                    print >> FILE, "%.2f	%1.3e	%1.3e	%1.3e" % args

    # print "scaled traces saved like this: {:}/a0.trace".format(directory)

    # Save as well the posiion file somewhere if you scale the complete star
    # shape pattern
    posfile_new = os.path.join(directory, "antpos.dat")
    with open(posfile_new, "w+") as file_ant:
        for i in range(0, end):
            args = (pos_new[i, 0], pos_new[i, 1], pos_new[i, 2])
            try:
                print("%.3f	%.3f	%.3f" % args, end='\n', file=file_ant)
            except SyntaxError:
                print >> file_ant, "%.3f	%.3f	%.3f" % args


def scale(sim_dir, primary, energy, zenith, azimuth, injection_height, altitude):
    """Scale all simulated traces to the shower parameters

    Parameters
    ---------
        sim_dir: str
            path to older containing the files of the reference shower
        primary: str
            primary for target shower, for now it just excepts 'electron' or 'pion'
        energy: float
            primary energy of target shower in EeV
        zenith: float
            zenith angle of primary of target shower in radian
        azimuth: float
            azimuth angle of primary of target shower in radian
        injection_height: float
            injectionheight of particle for target shower in meters
        altitude: float
            usually same as injh2, but there could be exceptions

    Returns:
    --------
    -
    Start the scaling and isometry process of the simulated reference shower acoording to target shower parameters

    """

    # Loop over runs
    steerfile_sim = os.path.join(sim_dir, "MasterIndex")
    with open(steerfile_sim, "r") as f:
        for line in f:
            # Unpack the run settings
            args = line.split()
            run = args[0]
            if not os.path.exists(os.path.join(sim_dir, run)):
                continue
            E1, zen1, az1, injh1, dist1 = map(float, args[2:])

            # Conversion from Aires to GRAND convention
            zen1 = 180. - zen1
            az1 = 180. + az1

            # Scale this run
            _scale_run(sim_dir, run, primary, E1, zen1, az1, injh1, dist1,
                       energy, zenith, azimuth, injection_height, altitude)

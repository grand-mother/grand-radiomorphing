''' Start the interpolation of the signal at a desired antenna position 
on the basis of the rescaled electric field traces and the antenna position after the isometry.
#
Notes: 
This script does the preperation for a interpolation of a complete pulse at any antenna position you desire.
Therfore, you have to hand over a list of antenna position you would like to have, a file containing the 
simulations which should be use (names of the planes) and a path whre to find these simlations
the script calculates all the prjections on the exiting planes which are needed, and hand the traces 
and positions over to the interpolation script which performs the interpoaltion alwys in between two positions
whether you wanna use filtered traces is set in this script by hand at the beginning
Tt returns files (t, Ex,Ey,Ez) in a folder, named InterpoaltedSignals, if it exists. 
'''
from __future__ import print_function
import os
import numpy as np
from interpolation import interpolate_trace
from frame import UVWGetter
from scaling import scale
from utils import getCerenkovAngle, load_trace
import operator

from utils import getn, get_integratedn, mag


def _ProjectPointOnLine(a, b, p):
    ap = p-a
    ab = b-a
    nrm = np.dot(ab, ab)
    if nrm <= 0.:
        print(a, b)
    point = a + np.dot(ap, ab) / nrm * ab
    return point


def _ProjectPointOnPlane(a, b, d, p):
    n = np.cross(a, b)
    n = n/np.linalg.norm(n)
    t = (np.dot(d, n)-np.dot(p, n))/np.dot(n, n)
    point = p + t*n
    return point


def interpolate(path0, path1, path2, zenith=None, azimuth=None, injection_height=None, scaled=True):
    """Interpolate all traces from the (rescaled) closest neighbours
        - first the closest neighbours to a desired antenna positions get identified,
            traces and positions are handed over to the interpolation function,
            calculated signal traces get stored.          

    Args:
        path0 (str): path to file with desired antenna positions
        path1 (str): path to the simulations
        path2 (str): path to the folder for final traces
        zenith (float): zenith angle of the morphed shower, in degrees
        azimuth (float): azimuth angle of the morphed shower, in degrees
        scaled (bool): flag for interpolating from a non scaled shower
    returns:
        stores the interpolated signal trace in a data file 
    """

    # Check the consistency of the arguments
    if scaled:
        if (zenith is None) or (azimuth is None):
            raise ValueError("missing zenith or azimuth")
    else:
        if (zenith is not None) or (azimuth is not None):
            raise ValueError(
                "zen / az is not supported for non scaled showers")

    # switch on details print out DISPLAY=1
    # DISPLAY=0

    #####
    # add request whether f1 and f2 as freqeuncies are handed over
    # if not: full=1, oherwise full =0
    full = 1  # fullband =1 == no filtering, raw pulses
    f1 = 60.e6  # MHz
    f2 = 200.e6  # MHz
    ######

    phigeo = 2.72*np.pi / 180.  
    # 182.66#; (ie pointing 2.66 degrees East from full North) # phigeo= 0 from simulations inputfile % In both EVA & Zhaires, North = magnetic North
    thetageo = (180.-27.05)*np.pi / 180.  
    # 152.95*np.pi/180. #27.05*np.pi/180. #; (pointing down)-62.95

    # Hand over a list file including the antenna positions you would like to have.
    # positions[:,0]:height, positions[:,1]:x,positions[:,2]:y
    positions = np.loadtxt(path0)
    # if DISPLAY==1:
    #print('desired positions: ')
    #print(positions, len(positions))
    if len(positions) <= 1:
        print("Files of desired positions has to consist of at least two positions, Bug to be fixed")

    # Get the simulation settings
    steerfile_sim = os.path.join(path1, "MasterIndex")
    sims = []
    with open(steerfile_sim, "r") as f:
        for line in f:
            # Unpack the run settings
            args = line.split()
            run = args[0]
            if not os.path.exists(os.path.join(path1, run)):
                continue
            if not sims:
                # Get the settings of the reference shower
                zen, az, _, dist1 = map(float, args[3:])
                # Conversion from Aires to GRAND convention
                zen = np.deg2rad(180. - zen)
                az = np.deg2rad(180. + az)
            sims.append(run)

    if scaled:
        # Override zenith and azimuth with the morphed shower settings
        zen, az = zenith, azimuth  # in deg
        az = np.deg2rad(az)
        zen = np.deg2rad(zen)

    # scaled traces shall be read in
    if scaled:
        path1 = os.path.join(path1, "scaled_")

    # GET THE NEIGHBOURS
    # Here the closests Neighbours should be found....

    # Finding the Neighbours:  In principal one would like to check which star shape pattern are the closest etc.
    # it reads in all star shape pattern positions from the simulations you hand over
    positions_sims = np.zeros([len(sims), 120, 3])
    print("Attention: read-in fixed to 120 antennas max. - to be fixed at some point")
    for i in np.arange(0, len(sims)):  # loop over simulated antenna positions

        posfile = path1 + str(sims[i]) + '/antpos.dat'
        #if DISPLAY == 1:
            #print(posfile)
        positions_sims[i, :, :] = np.loadtxt(posfile)

    # if DISPLAY==1:
        #print("Antenna files loaded")

    for b in range(len(positions)):
        # print "##############  begin of interpolation at desired position ", b, ' at ',  positions[b]
        # desired positions has to get projected on all the planes to get the orthogonal distances to the planes
        # then one can check in between which planes the points is by choosing the two smallest distances
        # first find the two planes which are the closest
        dist_value = np.zeros(len(sims))  # 1dim distance array
        for i in range(len(sims)):
            PointPrime = _ProjectPointOnPlane(
                positions_sims[i, 10] - positions_sims[i, 11],
                positions_sims[i, 40] - positions_sims[i, 41],
                positions_sims[i, 1], positions[b])
            dist_value[i] = np.linalg.norm(positions[b] - PointPrime)
        # sort distances from min to max value and save the id
        dist_plane = np.argsort(dist_value)

        # if DISPLAY==1:
        #print('nearest neighbour planes found')
        #print(dist_plane[0], dist_plane[1])

        # reconstruct a link between desired position and Xmax
        # since xmax positions is not given in coordinates you reconstruct its
        # positions by defining a line from a plane and this line has a length of
        # the given distance of the simulated plane to Xmax: dist1 which is the
        # distance of the first plane in the simulations file
        # dist1 belongs to positions_sims[0,:], normal should always be the same

        sz = np.sin(zen)
        v = np.array((np.cos(az) * sz, np.sin(az) * sz, np.cos(zen)))
        p = np.array((np.mean(positions_sims[0, :, 0]),
                      np.mean(positions_sims[0, :, 1]),
                      np.mean(positions_sims[0, :, 2])))  # center of plane 1
        Xmax_pos = p + dist1 * v  # assuming that Xmax is "before" the planes

        # if DISPLAY==1:
        #print('shower direction')
        # print(v)

        #print(dist1, np.linalg.norm(v))
        #print('postion Xmax, position desired')
        #print(Xmax_pos, positions[b])

        # now you can construct a line given by Xmax_pos and your disired antenna
        # positions. the Intersection points of this line with the planes gives
        # you then the position for the pulseshape interpolation
        # plane is given by (point-p_0)*n=0, line given by
        # point=s*(Xmax_pos-positions)+ Xmax_pos
        nrm = 1. / np.dot(Xmax_pos - positions[b], v)
        # intersection Point on plane dist_plane[0]
        s0 = np.dot(positions_sims[dist_plane[0], 10] - Xmax_pos, v) * nrm
        Inter_plane0 = s0 * (Xmax_pos - positions[b]) + Xmax_pos
        # intersection Point on plane dist_plane[1]
        s1 = np.dot(positions_sims[dist_plane[1], 10] - Xmax_pos, v) * nrm
        Inter_plane1 = s1 * (Xmax_pos - positions[b]) + Xmax_pos

        # AZ 15 March 2018
        # fix for antenna positions before or beyond the simulated planes, Xmax, Interpoints and antenna position along line of sight -> check for distances
        if (np.linalg.norm(Inter_plane0-Xmax_pos) > np.linalg.norm(positions[b]-Xmax_pos)) and (np.linalg.norm(Inter_plane1-Xmax_pos) > np.linalg.norm(positions[b]-Xmax_pos)):
            print("########  desired antenna position beyond 79km from Xmax.... antenna at desired position ",
                  b, ' at ',  positions[b], " skipped")
            continue
        if (np.linalg.norm(Inter_plane0-Xmax_pos) < np.linalg.norm(positions[b]-Xmax_pos)) and (np.linalg.norm(Inter_plane1-Xmax_pos) < np.linalg.norm(positions[b]-Xmax_pos)):
            print("########  desired antenna position beyond 79km from Xmax.... antenna at desired position ",
                  b, ' at ',  positions[b], " skipped")
            continue

        ##############
        # DISPLAY=1
        # if DISPLAY==1:
            #import matplotlib.pyplot as plt
            #import pylab
            # from mpl_toolkits.mplot3d import Axes3D  ## just if 3D plotting is needed

            # Plot to check whether its working correctly
            #fig = plt.figure(1, facecolor='w', edgecolor='k')
            #ax = fig.add_subplot(111, projection='3d')
            #ax.scatter(positions_sims[dist_plane[0],:,0], positions_sims[dist_plane[0],:,1], positions_sims[dist_plane[0],:,2], c='red', marker='o', label="surrounding planes")
            #ax.scatter(positions_sims[dist_plane[1],:,0], positions_sims[dist_plane[1],:,1], positions_sims[dist_plane[1],:,2], c='red', marker='o')
            ###ax.scatter(positions_sims[dist_plane[2],:,0], positions_sims[dist_plane[2],:,1], positions_sims[dist_plane[2],:,2], c='red', marker='o')
            ###ax.scatter(positions_sims[dist_plane[3],:,0], positions_sims[dist_plane[3],:,1], positions_sims[dist_plane[3],:,2], c='red', marker='o')
            # ax.scatter(line[:,0],line[:,1],line[:,2],c='green', marker='o', lw = 0)# c='green', marker='+', s=80)
            # ax.scatter(line_ortho[:,0],line_ortho[:,1],line_ortho[:,2], c='black', marker='o', lw = 0 )# c='green', marker='+', s=80)
            #ax.scatter(positions[b,0], positions[b,1], positions[b,2], c='blue', marker='x', label='desired position', s=80 )
            # ax.scatter(test[0], test[1], test[2], c='green', marker='x', s=80) # orthogonal projection of point as test
            #ax.scatter(Xmax_pos[0], Xmax_pos[1], Xmax_pos[2], c='green', marker='x', label='Xmax positions' , s=80)
            #ax.scatter(Inter_plane0[0], Inter_plane0[1], Inter_plane0[2], c='green', marker='o', label='projection on planes' , s=80)
            #ax.scatter(Inter_plane1[0], Inter_plane1[1], Inter_plane1[2], c='green', marker='o', s=80 )
            #plt.legend(loc='upper right')
            #pylab.tight_layout(0.4, 0.5,1.0)

            # plt.show()

            # plt.close()

    # Find the right neighbour part

        def get_neighbours(plane, Inter_plane):
            """Rotate into shower coordinates and find the 4 closest neighbours
            """
            # Get the frame transform
            offinz = np.mean(positions_sims[dist_plane[plane], :, 2])
            offiny = np.mean(positions_sims[dist_plane[plane], :, 1])
            offinx = np.mean(positions_sims[dist_plane[plane], :, 0])
            pos = np.zeros((len(positions_sims[dist_plane[plane], :, :]), 3))
            GetUVW = UVWGetter(offinx, offiny, offinz,
                               zen, az, phigeo, thetageo)

            def set_index(d, i):
                """Set the indices of the neighbours antennas.
                """
                d[0] = i     # antenna which radius is smaller and angle smaller
                d[1] = i + 1  # antenna which radius is smaller and angle larger
                # antenna which radius is larger and angle smaller, based on 8 angles
                d[2] = i + 8
                d[3] = i + 9

            def compute_angle(a, b):
                nrm = 1. / (np.linalg.norm(a) * np.linalg.norm(b))
                c = np.clip(np.dot(a, b) * nrm, -1., 1.)
                angle = np.arccos(c)
                return angle

            Inter = GetUVW(Inter_plane)
            # just the first simulated antenna as reference axis
            pos[0, :] = GetUVW(positions_sims[dist_plane[plane], 0, :])

            radius = np.linalg.norm(Inter)
            angle = compute_angle(Inter, pos[0, :])

            d = np.zeros(4, dtype=int)
            set_index(d, 0)

            # loop over the simulated antenna positions
            for i in np.arange(1, len(positions_sims[dist_plane[plane], :, :])):
                pos[i, :] = GetUVW(positions_sims[dist_plane[plane], i, :])
                #pos[i,:] = GetUVW(positions_sims[dist_plane[plane],i,:], offinx,offiny,offinz, zen, az, phigeo, thetageo)

                # skip that events, radoius too small
                if radius < np.linalg.norm(pos[i, :]):
                    continue

                angle1 = compute_angle(pos[i, :], pos[0, :])

                if Inter[2] < 0.:
                    if pos[i, 2] <= 0:
                        # look for clostest alpha, pos[0,:] reference antenna, angle slightly larger than angle1
                        if angle < angle1:
                            set_index(d, i)

                if Inter[2] > 0.:
                    if pos[i, 2] >= 0:
                        # look for clostest alpha, pos[0,:] reference antenna, angle slightly larger than angle1
                        if angle > angle1:
                            set_index(d, i)

            # be sure that now antenna outside the starhape pattern covered area is chosen
            if radius > np.linalg.norm(pos[-1, :]):
                set_index(d, 130)

            return Inter, pos, d
    ##################

        # Intersection point, simulated positions in shower coordinates, indizes of neighbours
        Inter_0, pos_0, d0 = get_neighbours(0, Inter_plane0)
        Inter_1, pos_1, d1 = get_neighbours(1, Inter_plane1)

        # print d0, d1

        if (d0 > 120).any() or (d1 > 120).any():
            print("########  desired antenna position outside region in which interpolation works, no 4 neighbours.... antenna at desired position ",
                  b, ' at ',  positions[b], " skipped")
            continue
        try:
            valid = pos_0[d0[3]]
        except IndexError:
            print("########  desired antenna position outside region in which interpolation works, no 4 neighbours.... antenna at desired position ",
                  b, ' at ',  positions[b], " skipped")
            continue
        try:
            valid = pos_1[d1[3]]
        except IndexError:
            print("######## desired antenna position outside region in which interpolation works, no 4 neighbours.... antenna at desired position ",
                  b, ' at ',  positions[b], " skipped")
            continue

        # if DISPLAY==1:
            #import matplotlib.pyplot as plt
            #import pylab

            # Plot to check whether its working correctly
            #fig = plt.figure(1, facecolor='w', edgecolor='k')
            #ax = fig.add_subplot(111, projection='3d')
            #ax.scatter(positions_sims[dist_plane[0],:,0], positions_sims[dist_plane[0],:,1], positions_sims[dist_plane[0],:,2], c='red', marker='o', label="surrounding planes")
            #ax.scatter(positions_sims[dist_plane[1],:,0], positions_sims[dist_plane[1],:,1], positions_sims[dist_plane[1],:,2], c='red', marker='o')
            ###ax.scatter(positions_sims[dist_plane[2],:,0], positions_sims[dist_plane[2],:,1], positions_sims[dist_plane[2],:,2], c='red', marker='o')
            ###ax.scatter(positions_sims[dist_plane[3],:,0], positions_sims[dist_plane[3],:,1], positions_sims[dist_plane[3],:,2], c='red', marker='o')
            # ax.scatter(line[:,0],line[:,1],line[:,2],c='green', marker='o', lw = 0)# c='green', marker='+', s=80)
            # ax.scatter(line_ortho[:,0],line_ortho[:,1],line_ortho[:,2], c='black', marker='o', lw = 0 )# c='green', marker='+', s=80)
            #ax.scatter(positions[b,0], positions[b,1], positions[b,2], c='blue', marker='x', label='desired position', s=80 )
            # ax.scatter(test[0], test[1], test[2], c='green', marker='x', s=80) # orthogonal projection of point as test
            #ax.scatter(Xmax_pos[0], Xmax_pos[1], Xmax_pos[2], c='green', marker='x', label='Xmax positions' , s=80)
            #ax.scatter(Inter_plane0[0], Inter_plane0[1], Inter_plane0[2], c='green', marker='o', label='projection on planes' , s=80)
            #ax.scatter(Inter_plane1[0], Inter_plane1[1], Inter_plane1[2], c='green', marker='o', s=80 )

            # ax.scatter( pos_0[d0[0],1], pos_0[d0[0],2], c='blue', marker='x')#, s=80)#pos_0[d0[0],0],
            # ax.scatter( pos_0[d0[1],1], pos_0[d0[1],2], c='blue', marker='x')#, s=80)#pos_0[d0[1],0],
            # ax.scatter( pos_0[d0[2],1], pos_0[d0[2],2], c='blue', marker='x')#, s=80)#pos_0[d0[2],0],
            # ax.scatter( pos_0[d0[3],1], pos_0[d0[3],2], c='blue', marker='x')#, s=80) #  pos_0[d0[3],0],

            #ax.scatter( positions_sims[dist_plane[0],d0[0],0], positions_sims[dist_plane[0],d0[0],1], positions_sims[dist_plane[0],d0[0],2], c='blue', marker='x', s=80)
            #ax.scatter( positions_sims[dist_plane[0],d0[1],0], positions_sims[dist_plane[0],d0[1],1], positions_sims[dist_plane[0],d0[1],2], c='blue', marker='x', s=80)
            #ax.scatter( positions_sims[dist_plane[0],d0[2],0], positions_sims[dist_plane[0],d0[2],1], positions_sims[dist_plane[0],d0[2],2], c='blue', marker='x', s=80)
            #ax.scatter( positions_sims[dist_plane[0],d0[3],0], positions_sims[dist_plane[0],d0[3],1], positions_sims[dist_plane[0],d0[3],2], c='blue', marker='x', s=80)

            #plt.legend(loc='upper right')
            #pylab.tight_layout(0.4, 0.5,1.0)

            # plt.show()

            # plt.close()

        # DISPLAY=1

        # if DISPLAY==1 and len(pos_0)>0.:# and Inter_0[2]>0:
            #import matplotlib.pyplot as plt
            #import pylab

            # print "Check plot"
            # print d0, d1
            # Plot to check whether its working correctly
            #fig2 = plt.figure(2, facecolor='w', edgecolor='k')
            # ax2 = fig2.add_subplot(111)#, projection='3d')
            # ax2.scatter( pos_0[:,1], pos_0[:,2], c='red', marker='o', label="surrounding planes")#pos_0[:,0],
            ###ax.scatter(positions_sims[dist_plane[1],:,0], positions_sims[dist_plane[1],:,1], positions_sims[dist_plane[1],:,2], c='red', marker='o')
            # ax2.plot(pos_0[:,1], pos_0[:,2])#pos_0[:,0],

            # ax2.scatter( pos_0[d0[0],1], pos_0[d0[0],2], c='blue', marker='x')#, s=80)#pos_0[d0[0],0],
            # ax2.scatter( pos_0[d0[1],1], pos_0[d0[1],2], c='blue', marker='x')#, s=80)#pos_0[d0[1],0],
            # ax2.scatter( pos_0[d0[2],1], pos_0[d0[2],2], c='blue', marker='x')#, s=80)#pos_0[d0[2],0],
            # ax2.scatter( pos_0[d0[3],1], pos_0[d0[3],2], c='blue', marker='x')#, s=80) #  pos_0[d0[3],0],

            # ax2.scatter(Xmax_pos[0], Xmax_pos[1], Xmax_pos[2], c='green', marker='x', label='Xmax positions' )#, s=80)

            # ax2.scatter( Inter_0[1], Inter_0[2], c='green', marker='+', label='projection on planes', s=80)#Inter_0[0],
            #plt.legend(loc='upper right')
            #plt.tight_layout(0.4, 0.5,1.0)
            # plt.axis('equal')

            #plt.xlabel(r"vxB", fontsize=16)
            #plt.ylabel(r"vxvxB", fontsize=16)

            # ax3 = fig2.add_subplot(132)#, projection='3d')
            # ax3.scatter( pos_0[:,0], pos_0[:,1], c='red', marker='o', label="surrounding planes")#pos_0[:,0],
            ###ax.scatter(positions_sims[dist_plane[1],:,0], positions_sims[dist_plane[1],:,1], positions_sims[dist_plane[1],:,2], c='red', marker='o')
            # ax3.plot(pos_0[:,0], pos_0[:,1])#pos_0[:,0],

            # ax3.scatter( pos_0[d0[0],0], pos_0[d0[0],1], c='blue', marker='x', s=80)#pos_0[d0[0],0],
            # ax3.scatter( pos_0[d0[1],0], pos_0[d0[1],1], c='blue', marker='x', s=80)#pos_0[d0[1],0],
            # ax3.scatter( pos_0[d0[2],0], pos_0[d0[2],1], c='blue', marker='x', s=80)#pos_0[d0[2],0],
            # ax3.scatter( pos_0[d0[3],0], pos_0[d0[3],1], c='blue', marker='x', s=80) #  pos_0[d0[3],0],
            # ax3.scatter( Inter_0[0], Inter_0[1], c='green', marker='o', label='projection on planes' , s=80)#Inter_0[0],

            ##plt.xlabel(r"v", fontsize=16)
            ##plt.ylabel(r"vxv", fontsize=16)

            # ax3 = fig2.add_subplot(133)#, projection='3d')
            # ax3.scatter( pos_0[:,0], pos_0[:,2], c='red', marker='o', label="surrounding planes")#pos_0[:,0],
            ###ax.scatter(positions_sims[dist_plane[1],:,0], positions_sims[dist_plane[1],:,1], positions_sims[dist_plane[1],:,2], c='red', marker='o')
            # ax3.plot(pos_0[:,0], pos_0[:,2])#pos_0[:,0],

            # ax3.scatter( pos_0[d0[0],0], pos_0[d0[0],2], c='blue', marker='x', s=80)#pos_0[d0[0],0],
            # ax3.scatter( pos_0[d0[1],0], pos_0[d0[1],2], c='blue', marker='x', s=80)#pos_0[d0[1],0],
            # ax3.scatter( pos_0[d0[2],0], pos_0[d0[2],2], c='blue', marker='x', s=80)#pos_0[d0[2],0],
            # ax3.scatter( pos_0[d0[3],0], pos_0[d0[3],2], c='blue', marker='x', s=80) #  pos_0[d0[3],0],
            # ax3.scatter( Inter_0[0], Inter_0[2], c='green', marker='o', label='projection on planes' , s=80)#Inter_0[0],

            ##plt.xlabel(r"v", fontsize=16)
            ##plt.ylabel(r"vxvxB", fontsize=16)

            # plt.show()

        # if DISPLAY==1:
            # print '\n cloest antennas on ecach plane, Plane 1 and Plane 2'
            # print d0[0], d0[1], d0[2], d0[3]
            # print d1[0], d1[1], d1[2], d1[3]

            # Plot to check whether its working correctly
            #fig = plt.figure(1, facecolor='w', edgecolor='k')
            #ax = fig.add_subplot(111, projection='3d')
            #ax.scatter(positions_sims[dist_plane[0],:,0], positions_sims[dist_plane[0],:,1], positions_sims[dist_plane[0],:,2], c='red', marker='o', label="surrounding planes")
            #ax.scatter(positions_sims[dist_plane[1],:,0], positions_sims[dist_plane[1],:,1], positions_sims[dist_plane[1],:,2], c='red', marker='o')
            ##ax.plot(positions_sims[dist_plane[1],:,0], positions_sims[dist_plane[1],:,1], positions_sims[dist_plane[1],:,2])

            ###ax.scatter(positions_sims[dist_plane[2],:,0], positions_sims[dist_plane[2],:,1], positions_sims[dist_plane[2],:,2], c='red', marker='o')
            ###ax.scatter(positions_sims[dist_plane[3],:,0], positions_sims[dist_plane[3],:,1], positions_sims[dist_plane[3],:,2], c='red', marker='o')
            # ax.scatter(line[:,0],line[:,1],line[:,2],c='green', marker='o', lw = 0)# c='green', marker='+', s=80)
            # ax.scatter(line_ortho[:,0],line_ortho[:,1],line_ortho[:,2], c='black', marker='o', lw = 0 )# c='green', marker='+', s=80)
            #ax.scatter(positions[b,0], positions[b,1], positions[b,2], c='blue', marker='o', label='desired position', s=80 )
            #ax.scatter(positions_sims[dist_plane[0],d0[0]][0], positions_sims[dist_plane[0],d0[0]][1], positions_sims[dist_plane[0],d0[0]][2], c='blue', marker='x', s=80)
            #ax.scatter(positions_sims[dist_plane[0],d0[1]][0], positions_sims[dist_plane[0],d0[1]][1], positions_sims[dist_plane[0],d0[1]][2], c='blue', marker='x', s=80)
            #ax.scatter(positions_sims[dist_plane[0],d0[2]][0], positions_sims[dist_plane[0],d0[2]][1], positions_sims[dist_plane[0],d0[2]][2], c='blue', marker='x', s=80)
            #ax.scatter(positions_sims[dist_plane[0],d0[3]][0], positions_sims[dist_plane[0],d0[3]][1], positions_sims[dist_plane[0],d0[3]][2], c='blue', marker='x', s=80)
            #ax.scatter(positions_sims[dist_plane[1],d1[0]][0], positions_sims[dist_plane[1],d1[0]][1], positions_sims[dist_plane[1],d1[0]][2], c='blue', marker='x', s=80)
            #ax.scatter(positions_sims[dist_plane[1],d1[1]][0], positions_sims[dist_plane[1],d1[1]][1], positions_sims[dist_plane[1],d1[1]][2], c='blue', marker='x', s=80)
            #ax.scatter(positions_sims[dist_plane[1],d1[2]][0], positions_sims[dist_plane[1],d1[2]][1], positions_sims[dist_plane[1],d1[2]][2], c='blue', marker='x', s=80)
            #ax.scatter(positions_sims[dist_plane[1],d1[3]][0], positions_sims[dist_plane[1],d1[3]][1], positions_sims[dist_plane[1],d1[3]][2], c='blue', marker='x', s=80)

            #ax.scatter(Xmax_pos[0], Xmax_pos[1], Xmax_pos[2], c='green', marker='x', label='Xmax positions' , s=80)
            #ax.scatter(Inter_plane0[0], Inter_plane0[1], Inter_plane0[2], c='green', marker='o', label='projection on planes' , s=80)
            #ax.scatter(Inter_plane1[0], Inter_plane1[1], Inter_plane1[2], c='green', marker='o', s=80 )
            #plt.legend(loc='upper right')
            #plt.tight_layout(0.4, 0.5,1.0)

            # plt.show()

# PulseShape Interpolation part

        # if DISPLAY==1:
            #print('\n\n PLANE1')
        # PLANE 1
        # Get the pulseshape for the projection on line 1
            #print(' Projection 1 ')
            #print('\n Interpolate x')

        point_online1 = _ProjectPointOnLine(
            positions_sims[dist_plane[0], d0[0]], positions_sims[dist_plane[0], d0[1]], Inter_plane0)  # Project Point on line 1
        # if DISPLAY==1:
        #print(positions_sims[dist_plane[0],d0[0]], positions_sims[dist_plane[0],d0[1]], point_online1)

        def get_traces(plane, d, i, j):
            """Get the traces for antennas d[i], d[j] in the given plane
            """
            if full == 1:
                directory = path1 + str(sims[dist_plane[plane]])
                ti = load_trace(directory, d[i])
                tj = load_trace(directory, d[j])
            else:
                directory = path1 + str(sims[dist_plane[plane]])
                suffix = "_{:}-{:}MHz.dat".format(str(f1*1E-06), str(f2*1E-06))
                ti = load_trace(directory, d[i], suffix)
                tj = load_trace(directory, d[j], suffix)
            return ti, tj

        # the interpolation of the pulse shape is performed
        txt0, txt1 = get_traces(0, d0, 0, 1)
        xnew1, tracedes1 = interpolate_trace(txt0.T[0], txt0.T[1], positions_sims[dist_plane[0], d0[0]], txt1.T[0],
                                             txt1.T[1], positions_sims[dist_plane[0], d0[1]], point_online1, upsampling=None, zeroadding=True)

        # Get the pulseshape for the projection on line 2
        point_online2 = _ProjectPointOnLine(
            positions_sims[dist_plane[0], d0[2]], positions_sims[dist_plane[0], d0[3]], Inter_plane0)  # Project Point on line 2
        #if DISPLAY == 1:
            #print('\n\n Projection 2 ')
            #print(positions_sims[dist_plane[0], d0[2]],
                  #positions_sims[dist_plane[0], d0[3]], point_online2)

        # the interpolation of the pulse shape is performed
        txt2, txt3 = get_traces(0, d0, 2, 3)
        xnew2, tracedes2 = interpolate_trace(txt2.T[0], txt2.T[1], positions_sims[dist_plane[0], d0[2]], txt3.T[0],
                                             txt3.T[1], positions_sims[dist_plane[0], d0[3]], point_online2, upsampling=None, zeroadding=True)

        # if DISPLAY==1:
        #print('\n interpolation plane 1')
        # Get the pulse shape of the desired position (projection on plane0) from projection on line1 and 2
        # print ' Pulse Shape '
        xnew_planex0, tracedes_planex0 = interpolate_trace(
            xnew1, tracedes1, point_online1, xnew2, tracedes2, point_online2, Inter_plane0, zeroadding=True)

        # if DISPLAY==1:
        #print('\n Interpolate y')
        # Get the pulseshape for the projection on line 1
        #print(' Projection 1 ')
        # the interpolation of the pulse shape is performed
        xnew1, tracedes1 = interpolate_trace(txt0.T[0], txt0.T[2], positions_sims[dist_plane[0], d0[0]], txt1.T[0],
                                             txt1.T[2], positions_sims[dist_plane[0], d0[1]], point_online1, upsampling=None, zeroadding=True)

        # Get the pulseshape for the projection on line 2
        # if DISPLAY==1:
        #print('\n\n Projection 2 ')
        # the interpolation of the pulse shape is performed
        xnew2, tracedes2 = interpolate_trace(txt2.T[0], txt2.T[2], positions_sims[dist_plane[0], d0[2]], txt3.T[0],
                                             txt3.T[2], positions_sims[dist_plane[0], d0[3]], point_online2, upsampling=None, zeroadding=True)

        # if DISPLAY==1:
        #print('\n interpolation plane 1')
        # Get the pulse shape of the desired position (projection on plane0) from projection on line1 and 2
        # print ' Pulse Shape '
        xnew_planey0, tracedes_planey0 = interpolate_trace(
            xnew1, tracedes1, point_online1, xnew2, tracedes2, point_online2, Inter_plane0, zeroadding=True)

        # if DISPLAY==1:
        #print('\n Interpolate z')
        # Get the pulseshape for the projection on line 1
        #print(' Projection 1 ')
        # the interpolation of the pulse shape is performed
        xnew1, tracedes1 = interpolate_trace(txt0.T[0], txt0.T[3], positions_sims[dist_plane[0], d0[0]], txt1.T[0],
                                             txt1.T[3], positions_sims[dist_plane[0], d0[1]], point_online1, upsampling=None, zeroadding=True)

        # Get the pulseshape for the projection on line 2
        # if DISPLAY==1:
        #print('\n\n Projection 2 ')
        # the interpolation of the pulse shape is performed
        xnew2, tracedes2 = interpolate_trace(txt2.T[0], txt2.T[3], positions_sims[dist_plane[0], d0[2]], txt3.T[0],
                                             txt3.T[3], positions_sims[dist_plane[0], d0[3]], point_online2, upsampling=None, zeroadding=True)

        # if DISPLAY==1:
        #print('\n interpolation plane 1')
        # Get the pulse shape of the desired position (projection on plane0) from projection on line1 and 2
        # print ' Pulse Shape '
        xnew_planez0, tracedes_planez0 = interpolate_trace(
            xnew1, tracedes1, point_online1, xnew2, tracedes2, point_online2, Inter_plane0, zeroadding=True)

        # if DISPLAY==1:
        #print('\n\n PLANE2')
        # PLANE 2
        #print('\n Interpolate x')
        # Get the pulseshape for the projection on line 1
        #print(' Projection 1 ')

        point_online12 = _ProjectPointOnLine(
            positions_sims[dist_plane[1], d1[0]], positions_sims[dist_plane[1], d1[1]], Inter_plane1)  # Project Point on line 1
        # if DISPLAY==1:
        #print(positions_sims[dist_plane[1],d1[0]], positions_sims[dist_plane[1],d1[1]], point_online12)
        # the interpolation of the pulse shape is performed
        txt0, txt1 = get_traces(1, d1, 0, 1)
        xnew1, tracedes1 = interpolate_trace(txt0.T[0], txt0.T[1], positions_sims[dist_plane[1], d1[0]], txt1.T[0],
                                             txt1.T[1], positions_sims[dist_plane[1], d1[1]], point_online12, upsampling=None, zeroadding=True)

        # Get the pulseshape for the projection on line 2
        # if DISPLAY==1:
        #print('\n\n Projection 2 ')
        point_online22 = _ProjectPointOnLine(
            positions_sims[dist_plane[1], d1[2]], positions_sims[dist_plane[1], d1[3]], Inter_plane1)  # Project Point on line 2
        # if DISPLAY==1:
        #print(positions_sims[dist_plane[1],d1[2]], positions_sims[dist_plane[1],d1[3]], point_online22)

        # the interpolation of the pulse shape is performed
        txt2, txt3 = get_traces(1, d1, 2, 3)
        xnew2, tracedes2 = interpolate_trace(txt2.T[0], txt2.T[1], positions_sims[dist_plane[1], d1[2]], txt3.T[0],
                                             txt3.T[1], positions_sims[dist_plane[1], d1[3]], point_online22, upsampling=None, zeroadding=True)

        # if DISPLAY==1:
        #print('\n interpolation plane 2')
        # Get the pulse shape of the desired position (projection on plane1) from projection on line1 and 2
        # print ' Pulse Shape '
        xnew_planex1, tracedes_planex1 = interpolate_trace(
            xnew1, tracedes1, point_online12, xnew2, tracedes2, point_online22, Inter_plane1, zeroadding=True)

        # if DISPLAY==1:
        #print('\n Interpolate y')
        # Get the pulseshape for the projection on line 1
        #print(' Projection 1 ')
        # the interpolation of the pulse shape is performed
        xnew1, tracedes1 = interpolate_trace(txt0.T[0], txt0.T[2], positions_sims[dist_plane[1], d1[0]], txt1.T[0],
                                             txt1.T[2], positions_sims[dist_plane[1], d1[1]], point_online12, upsampling=None, zeroadding=True)

        # Get the pulseshape for the projection on line 2
        # if DISPLAY==1:
        #print('\n\n Projection 2 ')
        # the interpolation of the pulse shape is performed
        xnew2, tracedes2 = interpolate_trace(txt2.T[0], txt2.T[2], positions_sims[dist_plane[1], d1[2]], txt3.T[0],
                                             txt3.T[2], positions_sims[dist_plane[1], d1[3]], point_online22, upsampling=None, zeroadding=True)

        # if DISPLAY==1:
        #print('\n interpolation plane 2')
        xnew_planey1, tracedes_planey1 = interpolate_trace(
            xnew1, tracedes1, point_online12, xnew2, tracedes2, point_online22, Inter_plane1, zeroadding=True)

        # if DISPLAY==1:
        #print('\n Interpolate z')
        # Get the pulseshape for the projection on line 1
        #print(' Projection 1 ')
        # the interpolation of the pulse shape is performed
        xnew1, tracedes1 = interpolate_trace(txt0.T[0], txt0.T[3], positions_sims[dist_plane[1], d1[0]], txt1.T[0],
                                             txt1.T[3], positions_sims[dist_plane[1], d1[1]], point_online12, upsampling=None, zeroadding=True)

        # Get the pulseshape for the projection on line 2
        # if DISPLAY==1:
        #print('\n\n Projection 2 ')
        # the interpolation of the pulse shape is performed
        xnew2, tracedes2 = interpolate_trace(txt2.T[0], txt2.T[3], positions_sims[dist_plane[1], d1[2]], txt3.T[0],
                                             txt3.T[3], positions_sims[dist_plane[1], d1[3]], point_online22, upsampling=None, zeroadding=True)

        # if DISPLAY==1:
        #print('\n interpolation plane 2')
        xnew_planez1, tracedes_planez1 = interpolate_trace(
            xnew1, tracedes1, point_online12, xnew2, tracedes2, point_online22, Inter_plane1, zeroadding=True)

        # if DISPLAY==1:
        #print('\n\n final interpolation')
        xnew_desiredx, tracedes_desiredx = interpolate_trace(
            xnew_planex0, tracedes_planex0, Inter_plane0, xnew_planex1, tracedes_planex1, Inter_plane1, positions[b], zeroadding=True)

        xnew_desiredy, tracedes_desiredy = interpolate_trace(
            xnew_planey0, tracedes_planey0, Inter_plane0, xnew_planey1, tracedes_planey1, Inter_plane1, positions[b], zeroadding=True)

        xnew_desiredz, tracedes_desiredz = interpolate_trace(
            xnew_planez0, tracedes_planez0, Inter_plane0, xnew_planez1, tracedes_planez1, Inter_plane1, positions[b], zeroadding=True)

        # if DISPLAY==1:
        #print(' length of time traces: ', len(txt2.T[0]), len(xnew_desiredx))


##############################
        # print ' interpolated signal belonging to positions in ' +str(path0) +' saved as '

        # lop over b as number of desired positions
        if full == 1:
            name = path2 + '/a'+str(b)+'.trace'
            # print name
        else:
            name = path2 + "/a"+str(b)+'_'+str((f1*1e-6)) + \
                '-' + str((f2*1e-6)) + 'MHz.dat'


# would save trace without timing correction
        #FILE = open(name, "w+" )
        # for i in range( 0, len(xnew_desiredx) ):
            #print >>FILE,"%3.2f %1.5e %1.5e %1.5e" % (xnew_desiredx[i], tracedes_desiredx[i], tracedes_desiredy[i], tracedes_desiredz[i])
        # FILE.close()


#################################
# correct timing of radiomorphhing
# get time of 0 transition between peak and peak, assimung taht this time coresponds to Xmax (sign fipe thr increase and decrease of shower)
# correct this by decay point to Xmax position / c
# plus xmax to antenna position / c/n (integrated n over path)
# => correct zero transition for that resulting time: find bin where min and where max, find inbetween the zero crossing
# get the diff between that time and the calculated and correct the complete timing for that difference: xnew_desiredx - diff

# decay= (0,0, height)
# Xmax_pos
# positions[b]

# since zero crossing takes place in the same timebin for all three components, get the bin from the component with the largest signal.

        try:
            # Ex
            if (max(tracedes_desiredx)-min(tracedes_desiredx)) > (max(tracedes_desiredy)-min(tracedes_desiredy)) and (max(tracedes_desiredx)-min(tracedes_desiredx)) > (max(tracedes_desiredz)-min(tracedes_desiredz)):
                trace = tracedes_desiredx
    # Ey
            if (max(tracedes_desiredy)-min(tracedes_desiredy)) > (max(tracedes_desiredx)-min(tracedes_desiredx)) and (max(tracedes_desiredy)-min(tracedes_desiredy)) > (max(tracedes_desiredz)-min(tracedes_desiredz)):
                trace = tracedes_desiredy
    # Ez
            if (max(tracedes_desiredz)-min(tracedes_desiredz)) > (max(tracedes_desiredy)-min(tracedes_desiredy)) and (max(tracedes_desiredz)-min(tracedes_desiredz)) > (max(tracedes_desiredx)-min(tracedes_desiredx)):
                trace = tracedes_desiredz

            ind_min, value_ref0 = min(
                enumerate(tracedes_desiredx), key=operator.itemgetter(1))
            ind_max, value_ref1 = max(
                enumerate(tracedes_desiredx), key=operator.itemgetter(1))
            if ind_max > ind_min:
                a = ind_min
                c = ind_max
            else:
                a = ind_max
                c = ind_min
            zero_crossings = np.where(np.diff(np.signbit(trace[a:c])))[
                0]  # old bin for crossing

            if len(zero_crossings) == 1:
                zero_cross = zero_crossings[0]
            if len(zero_crossings) == 2:
                zero_cross = zero_crossings[0]
            if len(zero_crossings) > 2:
                zero_cross = int(np.mean(zero_crossings))

            # if it has problems to find zero crossing if electric field strength is very low
            if len(zero_crossings) == 0:
                # print value_ref0, value_ref1
                zero_cross = int(abs(a-c)/2.)

            # from utils import get_integratedn(zen2, height_Xmax, height_ant)
            decay = np.array([0, 0, injection_height])

            dist_decay_Xmax = mag(decay-Xmax_pos)
            dist_Xmax_ant = mag(Xmax_pos-positions[b])

            # light velocity
            c = 299792458*1.e-9  # m/ns

            n = get_integratedn(zen, injection_height, positions[b])

            newtime = dist_decay_Xmax/c + dist_Xmax_ant/c*n

            time_diff = newtime-xnew_desiredx[zero_cross]

            xnew_desiredx = xnew_desiredx + \
                time_diff * np.ones(len(xnew_desiredx))

        except UnboundLocalError:
            print("---- trace couldn't be defined, time not corrected")

        FILE = open(path2 + '/a'+str(b)+'.trace', "w+")
        for i in range(0, len(xnew_desiredx)):
            try:
                print("%3.2f %1.5e %1.5e %1.5e" % (
                    xnew_desiredx[i], tracedes_desiredx[i], tracedes_desiredy[i], tracedes_desiredz[i]), end='\n', file=FILE)
            except:
                print >>FILE, "%3.2f %1.5e %1.5e %1.5e" % (
                    xnew_desiredx[i], tracedes_desiredx[i], tracedes_desiredy[i], tracedes_desiredz[i])

        FILE.close()


#################################
        # if DISPLAY==1:
        # PLOTTING

        #import matplotlib.pyplot as plt
        #plt.plot(xnew_planey0, np.real(tracedes_planey0), 'g:', label= "plane 0")
        #plt.plot(xnew_planey1, np.real(tracedes_planey1), 'b:', label= "plane 1")
        #plt.plot(xnew_desiredy, np.real(tracedes_desiredy), 'r-', label= "desired")

        #plt.xlabel(r"time (s)", fontsize=16)
        #plt.ylabel(r"Amplitude muV/m ", fontsize=16)
        # plt.legend(loc='best')

        # plt.show()

def process(sim_dir, shower, antennas, out_dir):
    """Rescale and interpolate the radio traces for all antennas 
        - start the Radio Morphing procedure

    Args:
        sim_dir (str): path to the simulated traces
        shower (dict): properties of the requested shower
        antennas (str): path the requested antenna positions
        out_dir (str): path where the output traces should be dumped
    """
    # Rescale the simulated showers to the requested one
    # print "ATTENTION scaling commented"
    scale(sim_dir, **shower)

    # interpolate the traces.
    interpolate(antennas, sim_dir, out_dir,
                shower["zenith"], shower["azimuth"], shower["injection_height"])

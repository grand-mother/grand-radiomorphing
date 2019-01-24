'''Script to perform an interpolation between to electric field traces at a desired position
(called by core.py)

It needs as input antenna position 1 and 2, their traces (filtered or not) in one component, their time , and the desired antenna position
and returns the trace ( in x,y,z coordinate system) and the time from the desired antenna position
Zeroadding and upsampling of the signal are optional functions

IMPORTANT NOTE:
The interpolation of the phases includes the
interpolation of the signal arrival time. A linear interpolation implies a plane radio
emission wave front, which is a simplification as it is hyperbolic in shape. However, the wave front can be estimated as a plane between two simu-
lated observer positions for a sufficiently dense grid of observers, as then parts of
the wave front are linear on small scales.

This script bases on the diploma thesis of Ewa Holt (KIT, 2013) in the context of AERA/AUGER. It is based on the interpolation of the amplitude and the pahse in the frequency domain. 
This can lead to misidentifiying of the correct phase. We are working on the interplementaion on a more robust interpolation of the signal.
Feel free to include it if you have some time to work on it. The script is completely modular so that single parts can be substitute easily.
'''

import numpy
from scipy import signal
from utils import getn
import operator


#################################################################
# Not needed at the moment, removed later
# def rfftfreq(n, d=1.0, nyquist_domain=1):
# '''calcs frequencies for rfft, exactly as numpy.fft.rfftfreq, lacking that function in my old numpy version.
# Arguments:
# ---------
#n: int
# Number of points.
#d: float
# Sample spacing, default is set to 1.0 to return in units of sampling freq.

# Returns:
# -------
# f: array of floats
# frequencies of rfft, length is n/2 + 1
# '''
# if n % 2 == 0:
#f = numpy.array([n/2 - i for i in range(n/2,-1,-1)]) / (d*n)
# else:
#f = numpy.array([(n-1)/2 + 1 - i for i in range(n/2,-1,-1)]) / (d*n)
# if nyquist_domain is 1 you're done and return directly
# if nyquist_domain != 1:
# if nyquist_domain even, mirror frequencies
#if (nyquist_domain % 2) == 0: f = f[::-1]
#sampling_freq = 1./d
#fmax = 0.5*sampling_freq
#f += (nyquist_domain-1)*fmax
# return f
####################################

def unwrap(phi, ontrue=None):
    """Unwrap the phase to a strictly decreasing function.

    Arguments:
    ----------
        phi: numpy array, float
            phase of the signal trace
    Returns:
    ----------
        phi_unwrapped: numpy array, float
            unwarpped phase of the signal trace
    """

    phi_unwrapped = numpy.zeros(phi.shape)
    p0 = phi_unwrapped[0] = phi[0]
    pi2 = 2. * numpy.pi
    l = 0
    for i0, p1 in enumerate(phi[1:]):
        i = i0 + 1
        if p1 >= p0:
            l += numpy.floor_divide(p1 - p0, pi2) + 1
        phi_unwrapped[i] = p1 - l * pi2
        p0 = p1
        if ontrue is not None:
            print(i, phi[i], phi[i-1], l, phi_unwrapped[i], abs(phi[i] - phi[i-1]),
                  abs(phi[i] - phi[i-1] + numpy.pi), abs(phi[i] - phi[i-1] - numpy.pi), l)
    return phi_unwrapped


def interpolate_trace(t1, trace1, x1, t2, trace2, x2, xdes, upsampling=None,  zeroadding=None, ontrue=None, flow=60.e6, fhigh=200.e6):
    """Interpolation of signal traces at the specific position in the frequency domain

    Arguments:
    ----------
            t1: numpy array, float
                time in ns of antenna 1
            trace1: numpy array, float
                single component of the electric field's amplitude of antenna 1
            x1: numpy array, float
                position of antenna 1
            t2: numpy array, float
                time in ns of antenna 2
            trace2: numpy array, float
                single component of the electric field's amplitude of antenna 2
            x2: numpy array, float
                position of antenna 2
            xdes: numpy arry, float
                antenna position for which trace is desired, in meters
            upsampling: str
                optional, True/False, performs upsampling of the signal, by a factor 8
            zeroadding: str
                optional, True/False, adds zeros at the end of the trace of needed
            ontrue: str
                optional, True/False, just a plotting command
            flow, fhigh: floats
                optional, define the frequency range for plotting, if desired (DISPLAY=True/False)


    Returns:
    ----------
        xnew: numpy array, float
            time for signal at desired antenna position in ns
        tracedes: numpy array, float
            interpolated electric field component at desired antenna position
    """
    DISPLAY = False

    # hand over time traces of one efield component -t1=time, trace1=efield- and the position x1 of the first antenna, the same for the second antenna t2,trace2, x2.
    # xdes is the desired antenna position (m) where you would like to have the efield trace in time
    # if necessary you have to do an upsampling of the trace: upsampling=On
    # onTrue=On would give you printings to the terminal to check for correctness
    # flow= lower freq in Hz, fhigh=higher freq in Hz, not necessarily needed

    factor_upsampling = 1
    if upsampling is not None:
        factor_upsampling = 8
    c = 299792458.e-9  # m/ns

    # calculating weights: should be done with the xyz coordinates
    # since in star shape pattern it is mor a radial function connection the poistion of same signal as linear go for that solution.
    # if lines ar on a line, it will give the same result as before
    tmp1 = numpy.linalg.norm(x2 - xdes)
    tmp2 = numpy.linalg.norm(x1 - xdes)
    tmp = 1. / (tmp1 + tmp2)
    weight1 = tmp2 * tmp
    weight2 = tmp1 * tmp

    if numpy.isinf(weight1):
        print("weight = inf")
        print(x1, x2, xdes)
        weight1 = 1.
        weight2 = 0.
    if numpy.isnan(weight1):
        print('Attention: projected positions equivalent')
        weight1 = 1.
        weight2 = 0.
    epsilon = numpy.finfo(float).eps
    if (weight1 > 1. + epsilon) or (weight2 > 1 + epsilon):
        print("weight larger 1: ", weight1, weight2, x1, x2, xdes, numpy.linalg.norm(
            x2-x1), numpy.linalg.norm(x2-xdes), numpy.linalg.norm(xdes-x1))
    if weight1 + weight2 > 1 + epsilon:
        print("PulseShape_Interpolation.py: order in simulated positions. Check whether ring or ray structure formed first")
        print(weight1, weight2, weight1 + weight2)

    # get refractive indey at the antenna positions
    n1 = getn(x1[2])
    n2 = getn(x2[2])

    #################################################################################
    # linearly interpolation of the phases

    # first antenna
    # upsampling if necessary
    if upsampling is not None:
        trace1 = signal.resample(trace1, len(trace1)*factor_upsampling)
        t1 = numpy.linspace(t1[0], t1[-1], len(trace1)
                            * factor_upsampling, endpoint=False)

    if zeroadding is True:
        max_element = len(trace1)  # to shorten the array after zeroadding
        xnew = numpy.linspace(t1[0], 1.01*t1[-1],
                              int((1.01*t1[-1]-t1[0])/(t1[2]-t1[1])))
        xnew = xnew*1.e-9  # ns -> s
        zeros = numpy.zeros(len(xnew)-max_element)
        f = trace1
        f = numpy.hstack([f, zeros])
    if zeroadding is None:
        f = trace1
        xnew = t1*1.e-9

    fsample = 1./((xnew[1]-xnew[0]))  # Hz

    freq = numpy.fft.rfftfreq(len(xnew), 1./fsample)
    FFT_Ey = numpy.fft.rfft(f)

    Amp = numpy.abs(FFT_Ey)
    phi = numpy.angle(FFT_Ey)
    phi_unwrapped = unwrap(phi, ontrue)

    #############################

    # second antenna
    ## t in ns, Ex in muV/m, Ey, Ez
    # NOTE: Time binning always 1ns

    # upsampling if needed
    if upsampling is not None:
        trace = signal.resample(trace2, len(trace2)*factor_upsampling)
        trace2 = trace
        t2 = numpy.linspace(t2[0], t2[-1], len(trace2)
                            * factor_upsampling, endpoint=False)

    if zeroadding is True:
        # get the same length as xnew
        xnew2 = numpy.linspace(
            t2[0], t2[0] + (xnew[-1]-xnew[0])*1e9, len(xnew))
        xnew2 = xnew2*1.e-9
        f2 = trace2
        f2 = numpy.hstack([f2, zeros])
    if zeroadding is None:
        f2 = trace2
        xnew2 = t2*1e-9  # ns -> s
    fsample2 = 1./((xnew2[1]-xnew2[0]))  # *1.e-9 to get time in s

    freq2 = numpy.fft.rfftfreq(len(xnew2), 1./fsample2)
    FFT_Ey = numpy.fft.rfft(f2)

    Amp2 = numpy.abs(FFT_Ey)
    phi2 = numpy.angle(FFT_Ey)
    phi2_unwrapped = unwrap(phi2, ontrue)

# Get the pulsh sahpe at the desired antenna position

    # get the phase

    # getnumpy.zeros([len(phi2)]) the angle for the desired position
    phides = weight1 * phi_unwrapped + weight2 * phi2_unwrapped
    if ontrue is not None:
        print(phides)
    #if DISPLAY:
        #phides2 = phides.copy()

    # re-unwrap: get -pi to +pi range back and check whether phidesis inbetwwen
    phides = numpy.mod(phides + numpy.pi, 2. * numpy.pi) - numpy.pi

    #################################################################################
    # linearly interpolation of the amplitude

    #Amp, Amp2
    # Since the amplitude shows a continuous unipolar shape, a linear interpolation is sufficient

    Ampdes = weight1 * Amp + weight2 * Amp2
    #if DISPLAY:
        #Ampdes2 = Ampdes.copy()

# inverse FFT for the signal at the desired position
    Ampdes = Ampdes.astype(numpy.complex64)
    phides = phides.astype(numpy.complex64)
    #if DISPLAY:
        #phides2 = phides2.astype(numpy.complex64)
    Ampdes *= numpy.exp(1j * phides)

    tracedes = (numpy.fft.irfft(Ampdes))
    tracedes = tracedes.astype(float)

    # PLOTTING

    if DISPLAY:
        import matplotlib.pyplot as plt
        import pylab

        fig1 = plt.figure(1, dpi=120, facecolor='w', edgecolor='k')
        plt.subplot(311)
        plt.plot(freq, phi, 'ro-', label="first")
        plt.plot(freq2, phi2, 'bo-', label="second")
        plt.plot(freq2, phides, 'go--', label="interpolated")
        #plt.plot(freq2, phi_test, 'co--', label= "real")
        plt.xlabel(r"Frequency (Hz)", fontsize=16)
        plt.ylabel(r"phase (rad)", fontsize=16)
        plt.xlim(flow, fhigh)

        #pylab.legend(loc='upper left')

        plt.subplot(312)
        ax = fig1.add_subplot(3, 1, 2)
        plt.plot(freq, phi_unwrapped, 'r+')
        plt.plot(freq2, phi2_unwrapped, 'bx')
        plt.plot(freq2, phides2, 'g^')
        #plt.plot(freq2, phi_test_unwrapped, 'c^')
        plt.xlabel(r"Frequency (Hz)", fontsize=16)
        plt.ylabel(r"phase (rad)", fontsize=16)
        # plt.show()
        # plt.xlim([0,0.1e8])
        # plt.xlim([1e8,2e8])
        # plt.ylim([-10,10])
        # ax.set_xscale('log')
        plt.xlim(flow, fhigh)

        plt.subplot(313)
        ax = fig1.add_subplot(3, 1, 3)
        plt.plot(freq, Amp, 'r+')
        plt.plot(freq2, Amp2, 'bx')
        plt.plot(freq2, Ampdes2, 'g^')
        #plt.plot(freq2, Amp_test, 'c^')
        plt.xlabel(r"Frequency (Hz)", fontsize=16)
        plt.ylabel(r"Amplitude muV/m/Hz ", fontsize=16)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        plt.ylim([1e1, 10e3])
        plt.xlim(flow, fhigh)

        plt.show()

    if zeroadding is True:
        # hand over time of first antenna since interpolation refers to that time
        return xnew[0:max_element]*1.e9, tracedes[0:max_element]

    if upsampling is not None:
        return xnew[0:-1:8]*1.e9, tracedes[0:-1:8]
    else:
        xnew = numpy.delete(xnew, -1)
        return xnew*1.e9, tracedes  # back to ns

import numpy

def load_trace(directory, index, suffix=".trace"):
    """Load data from a trace file
    """
    path = "{:}/a{:}{:}".format(directory, index, suffix)
    with open(path, "r") as f:
        return numpy.array([list(map(float, line.split())) for line in f])

def getn(h):
    """Get the refractive index

       Reference:
        Zhaires (see email M. Tueros 25/11/2016)
    """
    # h in meters
    return 1. + 325E-06 * numpy.exp(-0.1218E-03 * h)

def getCerenkovAngle(h):
   """Get the Cerenkov angle
   """
   return numpy.arccos(1. / getn(h))




def get_integratedn(zen2, injh2, position):
    
    # assumption coordinate system so that tau decay at (0.,0, injectionheight)
    # calculation of integrated n implemented similar as in Zhaires
    
    Re= 6370949 # m, Earth radius
    ########
    # line of sight
    ux= position[0] -0.
    uy= position[1] -0.
    uz= position[2] -injh2
    
    nint= 10000 # so many sub tracks alng line of sight
    # vector k along the line of sight
    kx=ux/nint
    ky=uy/nint
    kz=uz/nint
    
    #current positions, start with injh as first point of emission
    currpx=0.
    currpy=0.
    currpz=injh2
    currh=currpz # just in that case here, since particle injected directly induce a shower
    
    
    
    ns=325E-06
    kr=-0.1218E-03
    
    #print "inhh, antenna height ", injh2, position[2]
    summe=0.
    for i in range(0,nint):
        nextpx=currpx+kx
        nextpy=currpy+ky
        nextpz=currpz+kz
        
        nextR=numpy.sqrt( nextpx*nextpx +nextpy*nextpy )
        nexth= ( numpy.sqrt((( injh2 - nextpz  ) + Re) * (( injh2  - nextpz  ) + Re) + nextR*nextR) - Re) /1e3
        
        if (abs(currh-nexth)>1e-10 ):
            summe=summe+ (  numpy.exp(kr*nexth) -   numpy.exp(kr*currh)  )/ (kr*( nexth - currh) )
        else:
            summe=summe+ numpy.exp(kr*currh)
        
        currpx=nextpx
        currpy=nextpy
        currpz=nextpy
        currR=nextR
        currh=nexth
        
        
    avn= ns*summe/nint
    n= 1.+ avn
    
    return  n # integrated n



def mag(x):
    return numpy.sqrt(x.dot(x))

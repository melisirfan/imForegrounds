"""
Created on Wed 11th Sep 2019

@author: mirfan
"""
import healpy as hp
import math
import numpy as np
import scipy.integrate as defint
from physical_constants import CST

def battye(sjy):
    """empirical point source model of battye 2013 for intergrated flux """
    
    oooh = np.log10(sjy)
    
    sumbit = 2.593 * oooh**0 + 9.333 * 10**-2 * oooh**1. -4.839 * 10**-4 * oooh**2. \
                + 2.488 * 10**-1 * oooh**3. + 8.995 * 10**-2 * oooh**4. + \
                    8.506 * 10**-3 * oooh**5.

    inte = (10.**sumbit) * sjy**(-2.5) * sjy
    
    return inte
    
def pois(sjy):
    """empirical point source model of battye 2013 for poisson power spec """
    
    oooh = np.log10(sjy)
    
    sumbit = 2.593 * oooh**0 + 9.333 * 10**-2 * oooh**1. -4.839 * 10**-4 * oooh**2. \
                + 2.488 * 10**-1 * oooh**3. + 8.995 * 10**-2 * oooh**4. + \
                    8.506 * 10**-3 * oooh**5.

    inte = (10.**sumbit) * sjy**(-2.5) * sjy**(2.0)
    
    return inte
    
def numcount(sjy):
    """empirical point source model og battye 2013 for source count """
    
    oooh = np.log10(sjy)
    
    sumbit = 2.593 * oooh**0 + 9.333 * 10**-2 * oooh**1. -4.839 * 10**-4 * oooh**2. \
                + 2.488 * 10**-1 * oooh**3. + 8.995 * 10**-2 * oooh**4. + \
                    8.506 * 10**-3 * oooh**5.

    inte = (10.**sumbit) * sjy**(-2.5) 
    
    return inte
    
def smooth_map(intens, fwhm_am, goalFWHM):
    """ This function smooths the maps and sets nside """

    if goalFWHM != fwhm_am:
        source_fwhm = fwhm_am/60.
        fwhm = math.sqrt((goalFWHM/60.)**2 - source_fwhm**2)
        intens = hp.smoothing(intens, fwhm=math.radians(fwhm))

    return intens

#Based on Eq 36 from https://arxiv.org/pdf/1209.0343.pdf 
#and https://www.research.manchester.ac.uk/portal/files/67403180/FULL_TEXT.PDF p99
# life tip: don't forget 10^-26 conversion from Jy into Wm-2Hz-1

def make_ps(nside, freqs, Smax, resol, beta, deltbeta):
    """
        make extragalactic point source maps 

        INPUTS: nside - desired nside for output maps
                freqs - desired frequencies for output maps 
                Smax - the maximum flux value allowed in Jy
                resol - array of resolution of maps
                beta - spectral index of point sources
                deltabeta - standard deviation of beta

        OUTPUTS:
                array of point source maps
    """
    
    ell = np.arange(nside*3) + 1.0
    npix = 12 * nside * nside
    pixarea = (np.degrees(4 * np.pi) * 60.) / (npix)
    lenf = len(freqs)
    pixn = hp.pixelfunc.nside2resol(nside, arcmin=True)
    cfact = CST["light"]**2 / (2 * CST["kbolt"] * (1.4e9)**2) * 10.**-26

    ######### first to make the point source map at 1.4 GHz ################
    # Get the mean temperature 
    intvals = defint.quad(lambda sjy: battye(sjy), 0., Smax)
    tps14 = cfact * (intvals[0] - intvals[1])
    
    #Get the clustering contribution
    Clclust = 1.8 * 10**-4 * ell**-1.2 * tps14**2
    clustmap = hp.sphtfunc.synfast(Clclust, nside, new=True)

    #Get the poisson contribution
    #under 0.01 Jy poisson contributions behave as gaussians
    Clpoislow = np.zeros((len(ell)))
    val = 0
    for ii in np.arange(1e-6, 0.01, (0.01-1e-6)/ len(ell)):
        intvals = defint.quad(lambda sjy: pois(sjy), 0., ii)
        Clpoislow[val] = cfact**2 * (intvals[0] - intvals[1])
        val += 1
    poislowmap = hp.sphtfunc.synfast(Clpoislow, nside, new=True)
    
    shotmap = np.zeros((npix))
    #over 0.01 Jy you need to inject sources into the sky
    if Smax > 0.01:
        for ii in np.arange(0.01, Smax, (Smax - 0.01)/10.):
            #N is number of sources per steradian per jansky
            numbster = defint.quad(lambda sjy: numcount(sjy), ii-1e-3, ii+1e-3)[0]
            numbsky = int(4 * np.pi * numbster * ii)
            tempval = cfact * defint.quad(lambda sjy: battye(sjy), 0.01, ii)[0] / pixarea
            print numbsky, tempval
            randind = np.random.choice(range(npix), numbsky)
            shotmap[randind] = tempval

    map14 = np.array([smooth_map(tps14 + poislowmap + clustmap + shotmap, pixn, \
            resol[ii]) for ii in range(lenf)])
    #########################################################################
    
    ######### scale up to different frequencies ################
    alphas = np.array([smooth_map(np.random.normal(beta, scale=deltbeta**2, \
                size=npix), pixn, resol[ii]) for ii in range(lenf)])
    maps = np.array([map14[freval] * (freqs[freval]/1400.)**(alphas[freval]) for freval in range(lenf)])
    #########################################################################
    return maps
    
#running example for MeerKLASS
freqs = np.array(range(893, 1160, (1160 - 893)/(20-1))) #in MHz
freqs = freqs.astype(float)
beams = np.degrees((1.22 * CST["light"]) / (freqs * 1e6 * 13.5)) * 60.
myps = make_ps(256, freqs, 0.1, beams, -2.7, 0.2)
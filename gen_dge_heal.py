"""
Created on Wed 2nd Oct 2019

@author: mirfan
"""

import numpy as np
import math
import healpy as hp
    
def smooth_map(intens, fwhm_am, goalFWHM):
    """ Smoothing the maps from fwhm_am (arcmin) to goalFWHM (arcmin) """

    if goalFWHM != fwhm_am:
        source_fwhm = fwhm_am/60.
        fwhm = math.sqrt((goalFWHM/60.)**2 - source_fwhm**2)
        intens = hp.smoothing(intens, fwhm=math.radians(fwhm))

    return intens
    
def planckcorr(freq_ghz):
    """ Takes in frequency in GHZ and produces factor to be applied to temp """
    
    freq = freq_ghz * 10.**9.
    factor = CST["plancks"] * freq / (CST["kbolt"] * CST["cmb_temp"])
    correction = (np.exp(factor)-1.)**2. / (factor**2. * np.exp(factor))
    
    return correction

CST = {"kbolt": 1.3806488e-23, "light": 2.99792458e8, "plancks": 6.626e-34, "cmb_temp": 2.73}

def generate(nside, ffp10loc, vstart, vend, space, freeInd=-2.13, dishrad=13.5, \
        omegat=5.e3, obshr=4.e3, dishes=64, Tinst=20):
        """
        determine the ff, synchrotron and noise contributions for a given MHZ frequency range

        INPUTS: nside - desired nside for output maps
                ffp10loc - location of PSM maps 
                vstart - beginning of frequency range (MHz)
                vend - end of frequency range (MHz)
                space - number of frequency bands
                freeInd - spectral index of free-free emission (for temperature not flux)
                dishrad - radius of dish
                omegat - square degrees of the sky covered
                obshr - observation time in hours
                dishes - number of dishes in use
                Tinst = receiver system temp

        OUTPUTS:
                map["TotalFlux"] - [n_obs X npix] matrix of foregrounds plus noise at beam FWHMs
                map["TotalSignal"] - [n_obs X npix] matrix of foregrounds at beam FWHMs
                map["Noise"] - [n_obs X npix] matrix of thermal noise per pixel
                map["SyncFine"] - [n_obs X npix] matrix of synchrotron emission at highest (lowest number) resolution 
                map["FreeFine"] - [n_obs X npix] matrix of free-free emission at highest (lowest number) resolution 
                map["BetaSy"] - vector of synchrotron spectral index at highest (lowest number) resolution 
                beams - array of beam FWHMs (arcmin) 
                freqs - array of frequencies (MHz)
    """
    
    #admin
    freqs = np.array(range(vstart, vend, (vend - vstart)/(space-1))) #in MHz
    freqs = freqs.astype(float)
    lenf = len(freqs)
    npix = 12 * nside * nside
    beams = np.degrees((1.22 * CST["light"]) / (freqs * 1e6 * dishrad)) * 60.

    s217loc = ffp10loc + 'COM_SimMap_synchrotron-ffp10-skyinbands-217_2048_R3.00_full.fits'
    f217loc = ffp10loc + 'COM_SimMap_freefree-ffp10-skyinbands-217_2048_R3.00_full.fits'
    s353loc = ffp10loc + 'COM_SimMap_synchrotron-ffp10-skyinbands-353_2048_R3.00_full.fits'

    #Convert maps from Tcmb to Trj
    sync217 = hp.fitsfunc.read_map(s217loc, field=0, nest=False) / planckcorr(217)
    sync353 = hp.fitsfunc.read_map(s353loc, field=0, nest=False) / planckcorr(353)
    free217 = hp.fitsfunc.read_map(f217loc, field=0, nest=False) / planckcorr(217)
    
    #get rid of unphysical values
    free217[np.where(free217 < 0.)[0]] = np.percentile(free217, 3)

    syncInd = np.log(sync353/ sync217) / np.log(353./217.)

    syncA = sync217 * ((freqs[0]/1000.)/217.)**(syncInd)
    freeA = free217 * ((freqs[0]/1000.)/217.)**(freeInd)
    
    #have the oracle values at finest resolution for frequency range
    mapres = hp.pixelfunc.nside2resol(2048, arcmin=True)
    fina = np.min(beams)
    syncA = smooth_map(syncA, mapres, fina)
    freeA = smooth_map(freeA, mapres, fina)
    
    #need to fill in small scale structure for sync ind map which is MAMD's map at 5deg
    els = np.array(range(4000)) + 1.0
    cl5deg = hp.sphtfunc.anafast(np.random.normal(0.0, np.std(syncInd), 12*2048*2048), lmax=4000)
    # power spectra taken from https://arxiv.org/pdf/astro-ph/0408515.pdf
    Cls = cl5deg[0] * (1000./els)**(2.4)
    smallmap = hp.sphtfunc.synfast(Cls, 2048)
    syncInd = smooth_map(syncInd+smallmap, mapres, fina)

    # downgrade
    syncA = hp.pixelfunc.ud_grade(syncA, nside)
    freeA = hp.pixelfunc.ud_grade(freeA, nside)
    syncInd = hp.pixelfunc.ud_grade(syncInd, nside)

    #noise calc from https://core.ac.uk/download/pdf/52405915.pdf, section 2.1.1
    # Tsys from https://pos.sissa.it/215/019/pdf
    Tsky = 60. * (300./freqs)**2.55
    Tsys = 1.1 * Tsky + Tinst
    omegap = 1.13 * (np.radians(beams/60.))**2
    allomega = 4. * np.pi * (omegat / 41253.)
    ttot = obshr * 60. * 60.
    deltaf = ((vend - vstart)/(space-1)) * 1.e6
    denom = deltaf * ttot * (omegap / allomega) * dishes
    rmsnoise = Tsys/ np.sqrt(denom)
    noise = np.array([np.random.normal(0.0, rmsnoise[ii], npix) for ii in range(lenf)])
    
    signal = np.zeros((lenf, npix))
    flux = np.zeros((lenf, npix))
    synctrue = np.zeros((lenf, npix))
    freetrue = np.zeros((lenf, npix))

    for ii in range(0, len(freqs)):
    
        syncmap = syncA * (freqs[ii]/freqs[0])**(syncInd)
        freemap = freeA * (freqs[ii]/freqs[0])**(freeInd)
    
        synctrue[ii, :] = syncmap
        freetrue[ii, :] = freemap

        signal[ii, :] = smooth_map(syncmap + freemap, fina, beams[ii])

        flux[ii, :] = signal[ii,:] + noise[ii, :]
        
    MAPS = {"TotalFlux": flux, "TotalSignal": signal, "Noise":noise, "SyncFine":synctrue, \
            "FreeFine":freetrue, "BetaSy":syncInd}

    return MAPS, beams, freqs

#running example for MeerKLASS
pre = '/Users/mirfan/Documents/SkyMaps/FFP10/'
maps, beams, freqs = generate(256, pre, 893, 1160, 20, freeInd=-2.13, dishrad=13.5, \
                        omegat=4.e3, obshr=4.e3, dishes=64, Tinst=25)
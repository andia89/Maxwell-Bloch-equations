import numpy as np
from scipy.special import gamma
from _helper_functions import cauchy, tsallis, find_nearest


def create_spins(gcoll, q, delta, gammaq, nbins, holes=False, inhomo=True, hyperfine=True, spin_width = None):
    
    #delta: ws- wc. this is positive for values "left" of the cavity peak
    
    hfcoupl = 2.3
    if spin_width:
        f = np.linspace(-spin_width/2, spin_width/2, nbins)
    else:
        f = np.linspace(-gammaq, gammaq, nbins)
    if hyperfine:
        pdf = tsallis(f, q, gammaq/2/np.pi) + tsallis(f-hfcoupl, q, gammaq/2/np.pi)+tsallis(f+hfcoupl, q, gammaq/2/np.pi)
    else:
        pdf = tsallis(f, q, gammaq/2/np.pi)

    if holes:
        frequency_exclude = gcoll/2/np.pi
        exclude_width = 0.2 #FWHM

        indf = find_nearest(f, frequency_exclude)
        scale = pdf[indf]

        toex = cauchy(f, frequency_exclude, exclude_width, -scale)+cauchy(f, -frequency_exclude, exclude_width, -scale)
        #toex2 = cauchy(f, 0, exclude_width, -pdf[int(len(f)/2)])
        pdf = pdf+toex

    pdf = np.clip(pdf, 0, max(pdf) )   

    spdf = sum(pdf)
    pdf = pdf/spdf    
    
    f += delta/2/np.pi
    
    if not inhomo:
        pdf = np.zeros(np.shape(pdf))
        pdf[int(nbins/2)] = 1

    spins = f #draw samples according to distribution
    gs = np.sqrt(pdf)*gcoll
    
    return spins, np.asarray(gs, dtype=np.complex128), pdf

def create_spins_inhomo_g(gcoll, g_list, q, delta, gammaq, nbins, holes=False, inhomo=True, hyperfine=True, spin_width=None):
    # g_list should be a list of coupling strengths normalized to one
    gs = np.empty((len(g_list)*nbins), dtype=np.complex128)
    spins = np.empty(np.shape(gs))
    for i, g in enumerate(g_list):
        spins_t, gt, pdf = create_spins(gcoll, q, delta, gammaq, nbins, holes=holes, inhomo=inhomo, hyperfine=hyperfine, spin_width=spin_width)
        gs[i*nbins:(i+1)*nbins] = g*gt
        spins[i*nbins:(i+1)*nbins] = spins_t
    return spins, gs, pdf
    
    
def create_spins_nocav(q, delta, gammaq, nbins, holes=False, inhomo=True, hyperfine=True, spin_width=None):
    #delta: ws- wc 
    
    hfcoupl = 2.3

    if spin_width:
        f = np.linspace(-spin_width/2, spin_width/2, nbins)
    else:
        f = np.linspace(-gammaq, gammaq, nbins)
    if hyperfine:
        pdf = tsallis(f, q, gammaq/2/np.pi) + tsallis(f-hfcoupl, q, gammaq/2/np.pi)+tsallis(f+hfcoupl, q, gammaq/2/np.pi)
    else:
        pdf = tsallis(f, q, gammaq/2/np.pi)

    if holes:
        frequency_exclude = gcoll/2/np.pi
        exclude_width = 0.2 #FWHM

        indf = find_nearest(f, frequency_exclude)
        scale = pdf[indf]

        toex = cauchy(f, frequency_exclude, exclude_width, -scale)+cauchy(f, -frequency_exclude, exclude_width, -scale)
        #toex2 = cauchy(f, 0, exclude_width, -pdf[int(len(f)/2)])
        pdf = pdf+toex

    pdf = np.clip(pdf, 0, max(pdf) )   

    spdf = sum(pdf)
    pdf = pdf/spdf    
    
    f += delta/2/np.pi
    
    if not inhomo:
        pdf = np.zeros(np.shape(pdf))
        pdf[int(nbins/2)] = 1

    spins = f #draw samples according to distribution
    
    return spins, pdf






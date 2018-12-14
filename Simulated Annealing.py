#!/usr/bin/env python
# coding: utf-8

# In[1]:

import logging
import numpy as np
from scipy import integrate
from scipy.special import gamma
from scipy.optimize import fsolve, basinhopping
import matplotlib.pyplot as plt
#import qutip.ipynbtools
from ipywidgets import FloatProgress
#import qutip.parallel
from scipy import signal
import scipy.io
import gc
import matplotlib.colors as colors
from IPython.display import clear_output, Javascript, Markdown
import datetime
import math
import uuid
import os
import time
#import ipyparallel
#import qutip.ui
import random
import signal
#%qtconsole


from Pulse import *
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())    # if this logger is muted
from Toolkit import Tk_file


# based on Maxwell-Bloch equations like in doi:10.1126/sciadv.1701626
# optimal control approach in doi:10.1007/s11664-014-3228-9
# todo:
"""
Questions:
- what does init_steady()?
- cavity field a looks weird during rectangular pulse (no!)
- no nice Rabi possible for high power and low Q cavity?
  -> have to satisfy Omega(t_pi) ~ kappa ? (kappa !> Omega) 
- does this work in meantime with HF substrucutre
  (vs Nöbauer paper: polarize N)
- why is the fundamental Fourier frequency of OC pulse the Rabi frequency?
  + its harmonics?
"""

# todo:
# write save method with params

class InputParams:
    def __init__(self):
        # collective coupling
        self.gcoll = 1  #10.7*np.pi #HWHM
        # cavity quality factor
        self.quali = 250#949
        # cavity f_res
        self.fr = 3.4892e3
        # cavity bandwidth
        self.kappa = None #HWHM
        # detuning cavity to spin central frequency ws-wc, unused!
        self.delta = 0* 2*np.pi
        # detuning drive to cavity wc-wp
        self.deltac = 0* 2*np.pi
        # transverse spin relaxation rate
        self.gperp = 1e-9*  2*np.pi*0.09 #HWHM
        # longitudinal spin relaxation rate
        self.gpar = 1e-9*   2*np.pi*.001#0.001 #HWHM
        # q of spin pdf, q-Gaussian (see tsallis)
        self.q = 1.39
        # width of spin pdf, q-Gaussian
        self.gammaq = 1e-9* 2*np.pi*9.4 #FWHM
        # bins in spin pdf
        self.nbins = 701#701#5000#20
        #g0 = self.gcoll/sqrt(nspins)

        self.pumptime = 200e-3      # us
        self.decaytime = 300e-3
        self.dt = 0.5e-3
        self.numsteps = None

        self.holes = False
        self.sim_inhom = True

        # todo: think of better OO structure
        # stuff that gets initialized
        self.spins = None
        self.gs = None
        self.pdf = None

    def init(self):
        # stuff that is calculated from input params
        self.kappa = np.pi * self.fr / self.quali  # HWHM
        self.numsteps = int(self.pumptime / self.dt)

        init_spin_pdf()

        logger.info("Saving input params for new init.")
        filename = 'mbeSim_params.json'
        filename = Tk_file._addTimestamp(filename)
        Tk_file.saveToFile(self, path_out + '\\' + filename, 'json')
        # plot_spin_pdf(param)


def init_output_dir():
    import os
    from datetime import datetime
    date = datetime.now().strftime("%Y%m%d")
    path_out = os.getcwd() + '\\output\\' + str(date)
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    return path_out

def init():
    init_logging()
    param.init()

    pass
    # only for calibration, doesn't init sim params
    #init_steady()
    #init_power()

def init_logging():
    # Here! -> All messages from inits done during 'import' not shown
    logger = logging.getLogger()  # get root logger
    # add new handler with own format
    consoleHandler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)  # to avoid debug from pistage to root logger
    format_1 = logging.Formatter("[%(levelname)7s] - [%(asctime)s] - %(message)s  (%(filename)s::%(funcName)s)",
                                 "%Y-%m-%d %H:%M:%S")
    format_2 = logging.Formatter("[%(levelname)7s] - %(message)s ", "%Y-%m-%d %H:%M:%S")
    consoleHandler.setFormatter(format_2)
    logger.addHandler(consoleHandler)

    loggersStr = [str(loggers) for loggers in logging.Logger.manager.loggerDict]
    logging.debug("Root logger set active.")

    # deactivate all loggers:
    for loggerStr in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(loggerStr)
        logger.propagate = False

    # activate logger of this file
    logger = logging.getLogger(__name__)
    logger.propagate = True

def odeintz(func, z0, t, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    # Disallow Jacobian-related arguments.
    _unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = integrate.odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict
    else:
        z = result.view(np.complex128)
        return z

def tsallis(f, q, width):
    # Q Gaussian distribution
    delta = (width)/2*np.sqrt(2*q-2)/np.sqrt(2**q-2)/np.sqrt(2)
    norm = np.sqrt(q-1)*gamma(1/(q-1))/(np.sqrt(2*np.pi)*delta*gamma((3-q)/(2*(q-1))))
    val = (1+(q-1)*f**2/delta**2/2)**(1/(1-q))
    return norm*val

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

def find_nearest_sorted(array,value):
    """much faster for sorted arrays"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def cauchy(x, pos, fwhm, ampl):
    return 1/((x-pos)**2+0.25*fwhm**2)*0.25*fwhm**2*ampl


def init_spin_pdf():

    # todo: what about HFS? -> see '_create_pdf.ipynb'

    holes = param.holes #set to true if you want to include holes in your simulation
    inhomo = param.sim_inhom #do you want inhomogneous broadening

    f = np.linspace(-param.gammaq/2, param.gammaq/2, param.nbins)
    param.pdf = tsallis(f, param.q, param.gammaq/2/np.pi)


    if holes:
        frequency_exclude = param.gcoll/2/np.pi
        exclude_width = 0.2 #FWHM

        indf, freq = find_nearest(f, frequency_exclude)
        scale = param.pdf[indf]

        toex = cauchy(f, frequency_exclude, exclude_width, -scale)+cauchy(f, -frequency_exclude, exclude_width, -scale)
        #toex2 = cauchy(f, 0, exclude_width, -param.pdf[int(len(f)/2)])
        param.pdf = param.pdf+toex

    param.pdf = np.clip(param.pdf, 0, max(param.pdf) )

    spdf = sum(param.pdf)
    param.pdf = param.pdf/spdf

    if not inhomo:
        param.pdf = np.zeros(np.shape(param.pdf))
        param.pdf[int(param.nbins/2)] = 1

    param.spins = f #draw samples according to distribution
    param.gs = np.sqrt(param.pdf)*param.gcoll       # single g

    logger.info("Setting spin couplings for pdf. g_coll: {:.2f} MHz, Max. g_single: {:.2f} MHz".format(param.gcoll, np.max(param.gs)))


def plot_spin_pdf(param_in):

    f = np.linspace(-param_in.gammaq / 2, param_in.gammaq / 2, param_in.nbins)

    plt.plot(f, param_in.pdf)

    plt.xlabel('f (MHz)')   # todo: really freq or w units?
    plt.ylabel(r'$p$')
    plt.title(r'Spin pdf (Q-Gaussian)')


def func(a, eta, deltdrive):
    spins2 = 2*np.pi*(param.spins)+deltdrive
    sn = a*param.gs**2*param.gpar*(param.gperp-1j*spins2 )
    sd = param.gpar*param.gperp**2+4*np.abs(a)**2*param.gs**2*param.gperp+param.gpar*spins2**2
    s = sn/sd
    
    ret = 1j*a*deltdrive + eta + a*param.kappa + np.sum(s)
    return ret.real, ret.imag

def a_steady(eta, deltadrive, init):

    def equations(x):
        ar, ai = x
        return func(ar+1j*ai, eta, deltadrive)
    ar, ai = fsolve(equations, init)

    return ar, ai


# In[9]:
def init_steady():
    initsteady = (0.1,0.1)
    fstart = 50
    deltaarr = np.linspace(-fstart*2*np.pi,fstart*2*np.pi, 2000)
    etasteady = 0.00001
    losasteady = np.zeros(np.shape(deltaarr))
    losareal = np.zeros(np.shape(deltaarr))
    losaimag = np.zeros(np.shape(deltaarr))

    newinit = initsteady
    for i, delt in enumerate(deltaarr):
        ar, ai = a_steady(etasteady, delt, newinit)
        newinit = (ar, ai)
        losasteady[i] = ai**2+ar**2
        losareal[i] = ar
        losaimag[i] = ai

    fig = plt.figure()
    plt.plot(deltaarr/2/np.pi, 10*np.log10(losasteady/0.0000000001))
    #deltaarr[find(losasteady == max(losasteady))]/2/pi
    # todo: losasteady never used???


#@jit
def mbes(Y, t, eta):
    # this is as optimized as possible using numpy slicing so calling this function SHOULD be very fast
    ret = np.zeros(param.nbins * 2 + 1, dtype=complex)

    ret[0] = -param.kappa*Y[0]-1j*param.deltac*Y[0]+np.sum(param.gs*Y[1::2])+eta
    ret[1::2] = -(param.gperp+1j*param.spins*2*np.pi)*Y[1::2]+param.gs*Y[2::2]*Y[0]
    ret[2::2] = -param.gpar*(1+Y[2::2])-2*param.gs*(Y[1::2]*np.conj(Y[0])+np.conj(Y[1::2])*Y[0])
    return ret

#%%timeit gives
#10000 loops, best of 3: 45 µs per loop
#for 800 spins it becomes ~70µs. This is probably not the bottleneck


#this function is a lot slower, so be careful when using it
#@jit
def mbes_soc(Y, t, ilist, qlist, tlist):
    idx = find_nearest_sorted(tlist, t)
    ret = np.zeros(param.nbins * 2 + 1, dtype=complex)

    ret[0] = -param.kappa*Y[0]-1j*param.deltac*Y[0]+np.sum(param.gs*Y[1::2])+ilist[idx]-1j*qlist[idx]
    ret[1::2] = -(param.gperp+1j*param.spins*2*np.pi)*Y[1::2]+param.gs*Y[2::2]*Y[0]
    ret[2::2] = -param.gpar*(1+Y[2::2])-2*param.gs*(Y[1::2]*np.conj(Y[0])+np.conj(Y[1::2])*Y[0])
    return ret


# In[11]:


def do_calculation(drive, tlistpump, tlistafter, init):
    # calc response for rectangular pulse
    # using mbes (maxwell-bloch-equations) for n spins
    # output: asolpump:     A_sol with pump on
    #         asoldecay:    pump turned off
    #
    # shape of asol: col: [0]: cavity, [1,2] spin_1 (sigma-, sigmaz), spin_2, ..
    #                row: time
    # init: initial state: [0]: cavity, [1,2] spin_1 (sigma-, sigmaz), spin_2, ..


    #this should be very fast as well because scipy uses fortran 
    asolpump, infodict = odeintz(mbes, init, tlistpump, args=(drive,), full_output=True)
    init2 = asolpump[-1,:]
    asoldecay = odeintz(mbes, init2, tlistafter, args=(0,))
    
    # this is necessary because we are clustering the spins
    # weighting the spins accrording to pdf (?)
    asolpump[:,2::2] = asolpump[:,2::2]*param.pdf
    asolpump[:,1::2] = asolpump[:,1::2]*param.pdf
    asoldecay[:,2::2] = asoldecay[:,2::2]*param.pdf
    asoldecay[:,1::2] = asoldecay[:,1::2]*param.pdf
    return asolpump, asoldecay, infodict

def do_calculation_stimulated(drivepump, driveprobe, tlistpump, tlistwait, tlistprobe, tlistafter, init):
    #this should be very fast as well because scipy uses fortran 
    asolpump, infodict = odeintz(mbes, init, tlistpump, args=(drivepump,), full_output=True)
    init2 = asolpump[-1,:]
    asolwait = odeintz(mbes, init2, tlistwait, args=(0,))
    init3 = asolwait[-1,:]
    asolprobe = odeintz(mbes, init3, tlistprobe, args=(driveprobe,))
    init4 = asolprobe[-1,:]
    asoldecay = odeintz(mbes, init4, tlistafter, args=(0,))
    
    #this is necessary because we are clustering the spins
    asolpump[:,2::2] = asolpump[:,2::2]*param.pdf
    asolpump[:,1::2] = asolpump[:,1::2]*param.pdf
    asoldecay[:,2::2] = asoldecay[:,2::2]*param.pdf
    asoldecay[:,1::2] = asoldecay[:,1::2]*param.pdf
    asolwait[:, 2::2] = asolwait[:, 2::2]*param.pdf
    asolwait[:, 1::2] = asolwait[:, 1::2]*param.pdf
    asolprobe[:, 1::2] = asolprobe[:, 1::2]*param.pdf
    asolprobe[:, 2::2] = asolprobe[:, 2::2]*param.pdf
    return asolpump, asolwait, asolprobe, asoldecay, infodict
    
def do_calculation_soc(ilist, qlist, tlistpump, tlistafter, init):
    """this method is if you want to use shaped drive, which means that the drive-lists as i and q channels
    shape should be given as first arguments
    
    factor is the number by which one has to multiply tlistpump, to get a all integer list as 0,1,2,3...numsteps-1
    """
    #this should be very fast as well because scipy uses fortran 
    asolpump, infodict = odeintz(mbes_soc, init, tlistpump, args=(ilist, qlist, tlistpump), full_output=True)
    init2 = asolpump[-1,:]
    asoldecay = odeintz(mbes, init2, tlistafter, args=(0,))

    asolpump[:,2::2] = asolpump[:,2::2]*param.pdf
    asolpump[:,1::2] = asolpump[:,1::2]*param.pdf
    asoldecay[:,2::2] = asoldecay[:,2::2]*param.pdf
    asoldecay[:,1::2] = asoldecay[:,1::2]*param.pdf
    
    return asolpump, asoldecay, infodict
    #solution is that the first entry is the cavity, and then the first spin (sigma-, sigmaz) then the second etc...
    
def do_calculation_soc_pump_only(ilist, qlist, tlist, init):
    """this method is if you want to use shaped drive, which means that the drive-lists as i and q channels
    shape should be given as first arguments
    
    it doesn't use a decay afterwards such that it is better for genetic algorithm
    
    factor is the number by which one has to multiply tlistpump, to get a all integer list as 0,1,2,3...numsteps-1
    """
    #this should be very fast as well because scipy uses fortran
    asol, infodict = odeintz(mbes_soc, init, tlist, args=(ilist, qlist, tlist), full_output=True)
    
    asol[:,2::2] = asol[:,2::2]*param.pdf
    asol[:,1::2] = asol[:,1::2]*param.pdf
    
    return asol, infodict
    #solution is that the first entry is the cavity, and then the first spin (sigma-, sigmaz) then the second etc...


# ## SOC pulses

# In[12]:


def pulse_rwa(t, args):
    wgrund = args[0]  # fourier frequencies
    fcomps = args[1]  # fourier amplitudes
    pulse = 0
    for i in range(len(fcomps)):
        pulse = pulse+(fcomps[i])*np.sin((i+1)*wgrund*t)
    return pulse

def p_dbm_to_wrabi(p_in):
    aref = 2 * np.pi * 50       # rabi freq per sqrt(dBm) amplitude
    a_in = aref*10**(p_in/20.)  # power calibration: dB -> Rabi freq    [todo: f or w units?]

    return a_in

def p_wrabi_to_dbm(w_rabi):
    aref = 2 * np.pi * 50       # rabi freq per sqrt(dBm) amplitude
    p = 20 * np.log10(w_rabi/aref)

    return p

def p_db_to_factor(p_db):
    return 10**(p_db/20.)

def calc_pulse_rect(t_pump, p_in):
    """
    Calc s_z after rectangular pulse
    :param t_pump:
    :param p_in:
    :return:
    """
    a_in = p_dbm_to_wrabi(p_in)

    init = np.ones(param.nbins*2+1)*(-1)
    init[0] = 0
    init[1::2] = 0

    tlist_decay = np.arange(0, 1, 1)  # dummy decay

    # todo: calcing for whole tlist is probably super inefficient
    tlist = np.arange(0, t_pump + param.dt, param.dt)  # discrete time for ode
    asolpump, asoldecay, infodict = do_calculation(a_in, tlist, tlist_decay, init)

    return np.real(np.sum(asolpump[:, 2::2], axis=1))[-1]       # final s_z


def plot_rabi(p_in, t_start_us, t_end_us, n_t=100):
    # to validate pumptime
    # vary pumptime and look at sig_z. should see rabis.

    t_pulse_list = np.linspace(t_start_us, t_end_us, n_t)
    sz_final = []

    for i, t in enumerate(t_pulse_list):
        sz_final.append(calc_pulse_rect(t, p_in))

    plt.figure()
    plt.xlabel('Pulse time (ns) @ P={} dB'.format(p_in))
    plt.ylabel(r'$\sigma_z$')
    plt.plot(1000*t_pulse_list, sz_final)

def plot_rabi_2d_t_q(p_in, t_start_us, t_end_us, q_start, q_end, n_t=100, n_q=15):
    # to validate pumptime
    # vary pumptime and look at sig_z. should see rabis.

    t_pulse_list = np.linspace(t_start_us, t_end_us, n_t)
    q_list = np.linspace(q_start, q_end, n_q)

    sz_final = np.zeros([n_t, n_q])

    for j, q in enumerate(q_list):
        param.quali = q
        param.init()
        for i, t in enumerate(t_pulse_list):
            sz_final[i,j] = calc_pulse_rect(t, p_in)

    plt.figure()
    plt.xlabel('Pulse time (ns) @ P={} dB'.format(p_in))
    plt.ylabel(r'$Q$')
    plt.pcolor(1000*t_pulse_list, q_list, np.transpose(sz_final))#, vmin=-1, vmax=1)
    plt.colorbar()


def plot_rabi_2d_t_a(t_start_us, t_end_us, a_start, a_end, q=250, n_t=100, n_a=15):
    # to validate pumptime
    # vary pumptime and look at sig_z. should see rabis.

    t_pulse_list = np.linspace(t_start_us, t_end_us, n_t)
    a_list = np.linspace(a_start, a_end, n_a)
    param.quali = q
    param.init()
    sz_final = np.zeros([n_t, n_a])

    for j, a in enumerate(a_list):
        for i, t in enumerate(t_pulse_list):
            p_in = p_wrabi_to_dbm(a)
            sz_final[i,j] = calc_pulse_rect(t, p_in)

    plt.figure()
    plt.xlabel('Pulse time (ns) @ Q={} dB'.format(q))
    plt.ylabel(r'$a$')
    plt.pcolor(1000*t_pulse_list, a_list, np.transpose(sz_final))#, vmin=-1, vmax=1)
    plt.colorbar()


def plot_pulse_t_g(pulse, g_coll_start, gcoll_end, n_q=15):
    # to validate pumptime
    # vary pumptime and look at sig_z. should see rabis.

    t_list = pulse.t
    g_list = np.linspace(g_coll_start, gcoll_end, n_q)


    alos = np.zeros((len(g_list), len(t_list)))
    szlos = np.zeros(np.shape(alos))

    init = np.ones(param.nbins * 2 + 1) * (-1)
    init[0] = 0
    init[1::2] = 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for j, gcoll in enumerate(g_list):
        param.gcoll = gcoll
        param.init()

        pulse_i = pulse.to_iq()[0]
        pulse_q = pulse.to_iq()[1]

        asolpump, infodict = do_calculation_soc_pump_only(pulse_i, pulse_q, pulse.t, init)
        alos[j, :] = (abs(asolpump[:, 0])) ** 2
        szlos[j, :] = np.real(np.sum(asolpump[:, 2::2], axis=1))

        leave_out_factor = np.ceil(len(g_list) / 10.0)
        if j % leave_out_factor != 0:
            continue
        ax1.plot(1000 * t_list, alos[j, :], label=r'$g_c$={:.2f}, $g_s$={:.2f} MHz'.format(gcoll, np.max(param.gs)))
        ax2.plot(1000 * t_list, szlos[j, :], label=r'$g_c$={:.2f}, $g_s$={:.2f} MHz'.format(gcoll, np.max(param.gs)))


    # cav field
    ax1.set_xlabel('Time (ns)')
    ax1.set_title(r'Cavity field $a$')
    ax1.set_ylabel(r'$a$')
    ax1.legend()

    # spins
    ax2.set_xlabel('Time (ns)')
    ax2.set_title(r'Spin population $\sigma_z$')
    ax2.set_ylabel('$\sigma_z$')
    ax2.legend()


    """
    # cavity field amplitude a
    im1 = ax1.pcolor(t_list, g_list, alos, norm=colors.LogNorm())
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Coll couping (Mhz)')
    ax1.set_title(r'Cavity field $a$')
    # cbaxes = fig.add_axes([0.0, 0.1, 0.03, 0.8])
    # fig.colorbar(im1)

    # ensemble spin population sig_z
    im2 = ax2.pcolor(t_list, g_list, szlos)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Coll couping (Mhz)')
    ax2.set_title(r'Spin population $\sigma_z$')
    """

def init_power():
    # Calibrate power

    pumptime = param.pumptime
    decaytime = param.decaytime
    dt = param.dt
    tlist = np.arange(0, pumptime+dt, dt)
    tlistdecay = np.arange(0, decaytime+dt, dt)

    # test time it takes to calculate
    init = np.ones(param.nbins*2+1)*(-1)
    init[0] = 0
    init[1::2] = 0

    pulsei = np.ones((len(tlist)))

    # input power
    pin = np.arange(-45, 0,1)

    alos = np.zeros((len(pin), len(tlist)+len(tlistdecay)))
    szlos = np.zeros(np.shape(alos))

    for ctr, p in enumerate(pin):
        dr = p_dbm_to_wrabi(p)
        asolpump, asoldecay, infodict = do_calculation(dr, tlist, tlistdecay, init)
        alos[ctr, : ] = np.hstack((abs(asolpump[:,0])**2, abs(asoldecay[:,0])**2))
        szlos[ctr, :] = np.hstack((np.real(np.sum(asolpump[:, 2::2], axis=1)), np.real(np.sum(asoldecay[:, 2::2], axis=1))))

    # some plots, probably relevant for power calibration

    tplot = np.linspace(0, 1000*(pumptime+decaytime), len(tlist)+len(tlistdecay))
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))

    # cavity field amplitude a
    im1 = ax1.pcolor(tplot, pin, alos, norm=colors.LogNorm())
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Input power (dB)')
    ax1.set_title(r'Cavity field $a$')
    #cbaxes = fig.add_axes([0.0, 0.1, 0.03, 0.8])
    #fig.colorbar(im1)

    # ensemble spin population sig_z
    im2 = ax2.pcolor(tplot, pin, szlos)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Input power (dB)')
    ax2.set_title(r'Spin population $\sigma_z$')

    #fig.colorbar(im2)

    # sig_z, cut through power at end of pumptime (?)
    fig, ax3 = plt.subplots(1, 1)
    idx = min(range(len(tplot)), key=lambda i: abs(tplot[i]-pumptime*1000))
    plt.plot(pin, szlos[:, idx])
    ax3.set_xlabel('Power (dB)')
    ax3.set_ylabel(r'$\sigma_z$')
    ax3.set_title(r'Spin population after $t_{pump}$')

    # a, cut through times
    fig, ax4 = plt.subplots(1, 1)
    for idx, p in enumerate(pin):
        if idx % 4 != 0:
            continue
        ax4.plot(tplot, alos[idx,:], label='P={}'.format(str(p)))
    ax4.legend()
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel(r'$a$')
    ax4.set_title(r'Cavity field $a$')


def init_power_arb_pulse(pulse):
    # Calibrate power for compensated pulse

    pumptime = pulse.t[-1]
    dt = pulse.dt
    tlist = pulse.t

    # test time it takes to calculate
    init = np.ones(param.nbins * 2 + 1) * (-1)
    init[0] = 0
    init[1::2] = 0

    # input power in dB amplification from rabi frequencies
    pin = np.arange(0, 40, 2)

    alos = np.zeros((len(pin), len(tlist)))
    szlos = np.zeros(np.shape(alos))

    for ctr, p in enumerate(pin):
        kappa = p_db_to_factor(p)
        pulse_i = pulse.to_iq()[0] * kappa
        pulse_q = pulse.to_iq()[1] * kappa

        asolpump, infodict = do_calculation_soc_pump_only(pulse_i, pulse_q, pulse.t, init)
        alos[ctr, :] = (abs(asolpump[:, 0])) ** 2
        szlos[ctr, :] = np.real(np.sum(asolpump[:, 2::2], axis=1))

    # some plots, probably relevant for power calibration

    tplot = np.linspace(0, 1000 * (pumptime), len(tlist))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # cavity field amplitude a
    ax1.pcolor(tplot, pin, alos, norm=colors.LogNorm())
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Input power amplification(dB)')
    ax1.set_title(r'Cavity field $a$')

    # ensemble spin population sig_z
    ax2.pcolor(tplot, pin, szlos)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Input power amplification (dB)')
    ax2.set_title(r'Spin population $\sigma_z$')

    # sig_z, cut through power at end of pumptime (?)
    fig, ax3 = plt.subplots(1, 1)
    idx = min(range(len(tplot)), key=lambda i: abs(tplot[i] - pumptime * 1000))
    plt.plot(pin, szlos[:, idx])
    ax3.set_xlabel('Power amplification(dB)')
    ax3.set_ylabel(r'$\sigma_z$')
    ax3.set_title(r'Spin population after $t_{pump}$')

    # a, cut through times
    fig, ax4 = plt.subplots(1, 1)
    leave_out_factor = np.ceil(len(pin) / 10.0)
    for idx, p in enumerate(pin):
        if idx % leave_out_factor != 0:
            continue
        ax4.plot(tplot, alos[idx, :], label='P={}'.format(str(p)))
    ax4.legend()
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel(r'$a$')
    ax4.set_title(r'Cavity field $a$')


def calculate_fitness(amp_t_i, amp_t_q, t_list=[]):

    pulse_i = amp_t_i
    pulse_q = amp_t_q
    init = np.ones(param.nbins*2+1)*(-1)
    init[0] = 0
    init[1::2] = 0
    # probably solve
    solution, infodict = do_calculation_soc_pump_only(pulse_i, pulse_q, t_list, init)
    
    #plt.plot(timelist, pulse_i)
    #plt.show()
    
    return -(np.real(np.sum(solution[:, 2::2], axis=1)[-1]))

def pulse_t_from_f(fcomps, t, wgrund):
    fcomps_i = fcomps[:,0]
    fcomps_q = fcomps[:,1]
    pulse_i = pulse_rwa(t, [wgrund, fcomps_i])
    pulse_q = pulse_rwa(t, [wgrund, fcomps_q])
    
    return (pulse_i, pulse_q)


# In[16]:


class SimulatedAnnealing:
    def __init__(self, pumptime, n_steps):
        self.tp = np.linspace(0, pumptime, n_steps)
        
        self.number_fcp = 8 # fourier components
        self.max_ampl = 2*np.pi*50/self.number_fcp    # power constraint Rabi frequency = 50 MHz?
        self.wgrund = np.pi/self.tp[-1]               # base Fourier component = Rabi frequency of assumed pi pulse
        self.average_fitness = []
        self.best_fitness = []
        
        self.stepsize = self.max_ampl*0.1
        self.temp = 5
        self.number_tries = 50 #number of tries per cycle
        self.dT = 0.99

    def plot_pulse_interactive(self, ax, t, ys):

        if ax.lines:
            for i, line in enumerate(ax.lines):
                line.set_xdata(t)
                line.set_ydata(ys[i])
        else:
            for y in ys:
                ax.plot(t, y)
        plt.pause(0.05)

    def main_old(self):
        import time
        
        populationPulse = np.zeros((param.numsteps, 2))
        populationFourier = np.zeros((self.number_fcp, 2))
        populationFitness = np.zeros((1))
        populationSolution = np.zeros((param.numsteps, 2*param.nbins+1), dtype=complex)
        
        frand = np.random.uniform(-self.max_ampl, self.max_ampl, (self.number_fcp, 2))
        populationFourier[:, :] = frand
        pulse_i, pulse_q = pulse_t_from_f(populationFourier, self.tp, self.wgrund)
        energy = calculate_fitness(populationFourier, self.tp, self.wgrund)
        #disp1 = display(Markdown(""), display_id='0')
        
        fig, ax = plt.subplots(1,1)

        # debug plot_pulse_interactive
        #for f in range(5):
        #    self.plot_pulse_interactive(ax, self.tp, np.random.random(size=len(self.tp)))
        #    time.sleep(1)
        energy_best = 1
        pulse_fcomps_best = None


        while self.temp > 0:
            for i in range(self.number_tries):
                
                newfcomps = populationFourier + self.stepsize*np.random.uniform(-1, 1, (self.number_fcp, 2))
                newfcomps = np.clip(newfcomps, -self.max_ampl, self.max_ampl)
                
                energynew = calculate_fitness(newfcomps, self.tp, self.wgrund)
                               
                if energynew < energy:
                    populationFourier = newfcomps
                    energy = energynew
                    if energy < energy_best:
                        energy_best = energy
                        pulse_fcomps_best = populationFourier
                        print('New best pulse, inversion %f. fcomps %s ' % (-energy_best, pulse_fcomps_best))
                else:
                    # todo: this is weird. use energynew even if not lower?
                    # accept pulse with certain probabilty, even if not better
                    # see scipy.optimize.basinhopping
                    prop = min(1, np.exp(-(energynew-energy)/self.temp))
                    if np.random.uniform() < prop:
                        populationFourier = newfcomps
                        energy = energynew

                accepted = (energy == energynew)
                print("Temp: %.3f, Inversion: %f, Step: %d, accepted: %s"%(self.temp, -energy, i, accepted))
                
                # construct pulse in t domain
                pulse_i, pulse_q = pulse_t_from_f(populationFourier, self.tp, self.wgrund)
    
                self.plot_pulse_interactive(ax, self.tp, [pulse_i, pulse_q])

            self.temp *= self.dT
        return populationFourier

    def main_play(self):

        # matlab optimized (compensated pulse)
        path = "input\\"
        file = '180612_compensated_pi_200ns_downsampled400.mat'
        pulse_raw = scipy.io.loadmat(path + '\\' + file)
        t = pulse_raw['t_comp_in']
        y_complex = pulse_raw['y_comp_in']

        w_rabi = np.pi / t[0][-1]
        kappa_p = 500 # factor vacuum Rabi freq -> cavity Rabi freq

        # create comp pulse
        pulse_comp = AmpPhasePulse()
        pulse_comp.set_t_axis(t[0, :])
        pulse_comp.amps_c = y_complex[0, :]
        # convert to rabi freq in w units
        pulse_comp.amps_c = 2*np.pi*  np.real(pulse_comp.amps_c) + np.imag(pulse_comp.amps_c)
        pulse_comp.amps_c *= kappa_p


        # create rect pulse
        pulse_rect = AmpPhasePulse()
        pulse_rect.set_t_axis(t[0, :])

        pulse_rect.amps_c = np.ones(len(pulse_rect.t)) * w_rabi
        pulse_rect.amps_c *= kappa_p

        # show 2d plots: power a, s_z vs (amplification, time)
        #init_power_arb_pulse(pulse_comp)
        #init_power_arb_pulse(pulse_rect)
        #sz_comp = calculate_fitness(pulse_comp.to_iq()[0], pulse_comp.to_iq()[1], pulse_comp.t)
        #sz_rect = calculate_fitness(pulse_rect.to_iq()[0], pulse_rect.to_iq()[1], pulse_rect.t)
        # for reference
        sr_ref = calc_pulse_rect(pulse_rect.t[-1], 0)




        ##########
        ## Compensated and rect pulses for different couplings
        ##########
        #self.plot_comp_pulses_gcoll()

        #####
        ## 2d for rect pulse
        #####
        #plot_rabi_2d_t_a(0, 200e-3, 10e3, 50e3, n_t=50, n_a=5)

        plt.show()
        exit()


    def plot_comp_pulses_gcoll(self):
        # matlab optimized (compensated pulse)
        path = 'C:\\Users\\timo.joas\\OneDrive\\_Promotion\\Software\\Easyspin CompensatedPulse\\output'
        file = '180612_compensated_pi_200ns_downsampled400.mat'
        pulse_raw = scipy.io.loadmat(path + '\\' + file)
        t = pulse_raw['t_comp_in']
        y_complex = pulse_raw['y_comp_in']

        w_rabi = np.pi / t[0][-1]
        kappa_p = 500  # factor vacuum Rabi freq -> cavity Rabi freq

        # create comp pulse
        pulse_comp = AmpPhasePulse()
        pulse_comp.set_t_axis(t[0, :])
        pulse_comp.amps_c = y_complex[0, :]
        # convert to rabi freq in w units
        pulse_comp.amps_c = 2*np.pi*  np.real(pulse_comp.amps_c) + np.imag(pulse_comp.amps_c)
        pulse_comp.amps_c *= kappa_p

        # create rect pulse
        pulse_rect = AmpPhasePulse()
        pulse_rect.set_t_axis(t[0, :])

        pulse_rect.amps_c = np.ones(len(pulse_rect.t)) * w_rabi
        pulse_rect.amps_c *= kappa_p

        plot_pulse_t_g(pulse_rect, 1e-2 * 10.7 * np.pi, 5e-2* 10.7 * np.pi, 5)
        plot_pulse_t_g(pulse_comp, 1e-2 * 10.7 * np.pi, 5e-2* 10.7 * np.pi, 5)



    def main(self):

        pulse = FourierPulse(self.wgrund, self.number_fcp, self.max_ampl)
        pulse_new = FourierPulse(self.wgrund, self.number_fcp, self.max_ampl)

        pulse.init_random()

        energy = calculate_fitness(pulse.get_amp_t_i(self.tp), pulse.get_amp_t_i(self.tp), self.tp)

        fig, ax = plt.subplots(1, 1)

        # debug plot_pulse_interactive
        # for f in range(5):
        #    self.plot_pulse_interactive(ax, self.tp, np.random.random(size=len(self.tp)))
        #    time.sleep(1)
        energy_best = 1
        pulse_fcomps_best = None

        while self.temp > 0:
            for i in range(self.number_tries):

                f_comps_new = pulse.f_comps + self.stepsize * np.random.uniform(-1, 1, (self.number_fcp, 2))
                f_comps_new = pulse.clip_amps(f_comps_new)

                pulse_new.f_comps = f_comps_new

                energynew = calculate_fitness(pulse_new.get_amp_t_i(self.tp), pulse_new.get_amp_t_i(self.tp), self.tp)

                if energynew < energy:
                    pulse.f_comps = f_comps_new
                    energy = energynew
                    if energy < energy_best:
                        energy_best = energy
                        pulse_fcomps_best = pulse.f_comps
                        print('New best pulse, inversion %f. fcomps %s ' % (-energy_best, pulse_fcomps_best))
                else:
                    # accept pulse with certain probabilty, even if not better
                    # see scipy.optimize.basinhopping
                    prop = min(1, np.exp(-(energynew - energy) / self.temp))
                    if np.random.uniform() < prop:
                        pulse.f_comps = f_comps_new
                        energy = energynew

                accepted = (energy == energynew)
                print("Temp: %.3f, Inversion: %f, Step: %d, accepted: %s" % (self.temp, -energy, i, accepted))

                # construct pulse in t domain
                pulse_i, pulse_q = pulse.pulse_t_from_f(self.tp)

                self.plot_pulse_interactive(ax, self.tp, [pulse_i, pulse_q])

            self.temp *= self.dT
        return pulse




path_out = init_output_dir()
param = InputParams()
init()


#plot_rabi(0, 10e-3, param.pumptime*2)
#plot_rabi_2d_t_q(20, 10e-3, param.pumptime/2, 10, 50, n_t=100, n_q=15)

metropo = SimulatedAnnealing(param.pumptime, param.numsteps)

metropo.main_play()

#res = metropo.main()

plt.show()







#!/usr/bin/env python
# coding: utf-8

import numpy as np
from _helper_functions import fsolvez
from scipy.optimize import fsolve

def funca(a, eta, deltac, g, spins, gpar, gperp, kappa):

    spins2 = 2*np.pi*(spins)
    deltas = deltac+spins2
    gamma = gperp+gpar/2
    #s1 = 2*a*g**2*gpar
    #s2 = gpar+2*gperp+2*1j*(deltac+spins2)
    #s3 = 8*(gpar+2*gperp)*np.abs(a)**2*g**2
    #s4 = ((gpar+2*gperp)**2+4*spins2*(2*deltac+spins2))
    
    
    s1 = g**2*gpar*(gamma-1j*deltas)*a
    s2 = gpar*(gamma**2+deltas**2)+4*np.abs(a)**2*g**2*gamma
    
    #s = -s1/(s2*(gpar+s3/s4))
    s = -s1/s2
    
    ret = -1j*a*deltac + eta - a*kappa + np.sum(s)
    return ret.real, ret.imag

def funca_maser(a, pump, eta, deltac, g, spins, gpar, gperp, kappa, nbar=0):
    deltas = deltac+2*np.pi*(spins)
    gamma = gperp+gpar*(0.5+nbar)+pump*0.5 
    
    s1 = g**2*(gpar-pump)*(gamma-1j*deltas)
    s2 = 4*np.abs(a)**2*g**2*gamma+(gpar+pump+2*nbar*gpar)*(gamma**2+deltas**2)
    
    #s = -s1/(s2*(gpar+s3/s4))
    s = s1/s2
    
    ret = -1j*a*deltac + eta - a*kappa - a*np.sum(s)
    return ret.real, ret.imag

def funca2_maser(a2, pump, eta, deltac, g, spins, gpar, gperp, kappa):

    deltas = deltac+2*np.pi*(spins)
    gamma = gperp+gpar/2+pump/2
    
    den = (gpar+pump)*(gamma**2+deltas**2)+4*a2*g**2*gamma
   
    
    s1 = kappa+np.sum(g**2*gamma*(gpar-pump)/den)
    s2 = deltac+np.sum(g**2*deltas*(-gpar+pump)/den)

    
    ret = eta**2*1/(s1**2+s2**2) - a2
    return ret

def funcada_maser_cumulant_with_filter(Y, pump, deltac, g, spins, gpar, gperp, kappa, nbar, kappa_f, g_f, deltaf, detuning_array):
    ret = np.empty(len(Y), dtype=np.complex128)
    c1 = gpar/2*(1+2*nbar)+gperp+pump/2
    deltas = deltac+2*np.pi*(spins)
    delta = deltac-deltaf
    deltasf = deltas-deltaf
    deltasc = deltas-deltac
    ktot = kappa+kappa_f
    
    bdb = Y[0]
    ada = Y[1]
    asp = Y[2:]
    adsm = np.conjugate(asp)
    
    sz = -(gpar-pump+2*adsm*g+2*asp*g)/(gpar+2*nbar*gpar+pump)

    s1 = np.sum((2*asp.real)*g)
    s2 = np.sum(asp*g/(c1+kappa_f-1j*deltasf))
    sd1 = np.sum(g**2*sz/(c1+kappa_f-1j*deltasf))
    
    alpha = g_f*(ada-bdb+s2)/(delta + 1j*ktot - 1j*sd1)
    
    y = (-1j*asp*g_f+alpha*g*sz)/(c1+kappa_f-1j*deltasf)
    
    s12 = g*sz*np.sum(np.dot(detuning_array, adsm*g)-adsm*g/(2*c1))
    s11 = asp*np.sum(np.dot(detuning_array, g**2*sz)-g**2*sz/(2*c1))

    ret[0] = -2*bdb*kappa_f+1j*g_f*(alpha-np.conjugate(alpha))
    ret[1] = -2*(nbar-ada)*kappa-1j*g_f*(alpha-np.conjugate(alpha))+s1
    ret[2:] = -1j*g_f*y+ada*g*sz+0.5*g*(1+sz)-asp*(c1+kappa-1j*deltasc)+s11+s12

    return ret


def funcsz(sz, eta, deltac, g, spins, gpar, gperp, kappa):
    spins2 = 2*np.pi*(spins)
    deltas = deltac+spins2
    gamma = gperp+gpar/2
    ret = np.empty(len(spins))
    
    asum = g**2*sz/(gamma+1j*deltas)
    a = eta/(1j*deltac+kappa-np.sum(asum))
    
    s1 = 4*np.abs(a)**2*g**2*gamma*sz
    s2 = gamma**2+deltas**2
    s3 = -gpar*(1+sz)
    
    ret = s1/s2+s3
    return ret.real


def a_steady(eta, deltac, g, spins, gpar, gperp, kappa, init):
    # make sure that spins are already detuned correctly from cavity
    #deltac: wc-wp
    def equations(x):
        ar, ai = x
        return funca(ar+1j*ai, eta, deltac, g, spins, gpar, gperp, kappa)
    ar, ai = fsolve(equations, init)

    return ar, ai

def a_steady_maser(pump, eta, deltac, g, spins, gpar, gperp, kappa, init):
    # make sure that spins are already detuned correctly from cavity
    #deltac: wc-wp
    def equations(x):
        ar, ai = x
        return funca_maser(ar+1j*ai, pump, eta, deltac, g, spins, gpar, gperp, kappa)
    ar, ai = fsolve(equations, init)

    return ar, ai

def a_steady_maser_temp(nbar, pump, eta, deltac, g, spins, gpar, gperp, kappa, init):
    # make sure that spins are already detuned correctly from cavity
    #deltac: wc-wp
    #nbar: number of thermal photons
    def equations(x):
        ar, ai = x
        return funca_maser(ar+1j*ai, pump, eta, deltac, g, spins, gpar, gperp, kappa, nbar)
    ar, ai = fsolve(equations, init)

    return ar, ai


def a2_steady_maser(pump, eta, deltac, g, spins, gpar, gperp, kappa, init):
    # make sure that spins are already detuned correctly from cavity
    #deltac: wc-wp

    def equations(a2):
        return funca2_maser(a2, pump, eta, deltac, g, spins, gpar, gperp, kappa)
    
    a2 = fsolve(equations, init)

    return a2

def ada_steady_maser_cumulant_with_filter_cav(nbar, pump, deltac, g, spins, gpar, gperp, kappa, kappa_f, g_f, deltaf, init):
    # make sure that spins are already detuned correctly from cavity
    #deltac: wc-wp
    #nbar: number of thermal photons
    
    # here i create already the detuning array of size nbins*nbins. it only has to be done once that's why I do it here
    
    c1 = gpar/2*(1+2*nbar)+gperp+pump/2
    deltas = deltac+2*np.pi*(spins)
    
    spin_array = np.zeros((len(spins), len(spins)), dtype=np.complex128)
    for k in range(len(spins)):
        for j in range(len(spins)):
            spin_array[k,j] = 1/(2*c1-1j*(deltas[k]-deltas[j]))
    def equations(Y):
        return funcada_maser_cumulant_with_filter(Y, pump, deltac, g, spins, gpar, gperp, kappa, nbar, kappa_f, g_f, deltaf, spin_array)
    
    ret, infodict, ier, mesg = fsolvez(equations, init)
    #return adaggera and szs
    return ret, infodict, ier, mesg
    
    
def sz_steady_maser_from_a2(a2, pump, eta, deltac, g, spins, gpar, gperp, kappa):
    deltas = deltac+2*np.pi*(spins)
    gamma = gperp+gpar/2+pump/2
    
    s1 = pump-gpar
    s2 = 4*a2*g**2*gamma
    s3 = gamma**2+deltas**2
    
    return s1/(gpar+pump+s2/s3)

def sz_steady_maser_from_a2_temp(a2, nbar, pump, eta, deltac, g, spins, gpar, gperp, kappa):
    deltas = deltac+2*np.pi*(spins)
    gamma = gperp+gpar*(0.5+nbar)+pump*0.5 

    
    s1 = pump-gpar
    s2 = 4*a2*g**2*gamma
    s3 = gamma**2+deltas**2
    
    return s1/(gpar+pump+2*nbar*gpar+s2/s3)


def a_steady_hp(eta, deltac, g, spins, gpar, gperp, kappa, init):
    #init is only there for backwards compatibility
    deltas = deltac+ 2*np.pi*(spins)
    gamma = gperp+gpar/2
    ret = eta/(kappa+1j*deltac+np.sum(g**2/(1j*deltas+gamma)))
    return ret.real, ret.imag
            

def cav_steady(omegap, omegac, eta, kappa):
    #careful here. omegac-omegap is negative for omegap>omegac
    s = eta/(kappa+1j*(omegac-omegap))
    ar, ai = (s.real, s.imag)
    return ar, ai

def cav_steady_fano(omegap, omegac, eta, kappa, phi, dt, offset, offset_phi):
    #careful here. omegac-omegap is negative for omegap>omegac
    s1 = np.exp(-1j*phi)*np.exp(-1j*dt*omegap)
    s2 = eta/(kappa+1j*(omegac-omegap))
    s3 = offset*np.exp(1j*offset_phi)
    
    s = s1*(s2+s3)
    ar, ai = (s.real, s.imag)
    return ar, ai

def cav_steady_fano2(omegap, omegac, eta, kappa, phi, dt, offset, offset_phi):
    #careful here. omegac-omegap is negative for omegap>omegac
    s1 = np.exp(-1j*phi)*np.exp(-1j*dt*omegap)
    s2 = eta/(kappa-1j*(omegac-omegap))
    s3 = offset*np.exp(1j*offset_phi)
    
    s = s1*(s2+s3)
    ar, ai = (s.real, s.imag)
    return ar, ai


def sz_steady(eta, deltac, g, spins, gpar, gperp, kappa, init, pdf):
    # spins are detuned from cavity! spins = wc-ws
    #deltac: wc-wp
    def equations(x):
        sz = x
        return funcsz(sz, eta, deltac, g, spins, gpar, gperp, kappa)
    sz = fsolve(equations, init)

    return sz*pdf


def sz_steady_from_a(eta, deltac, g, spins, gpar, gperp, kappa, init, pdf):
    #init is here for the cavity amplitude not for the sz spins
    ar, ai = a_steady(eta, deltac, g, spins, gpar, gperp, kappa, init)
    absa2 = ar**2+ai**2
    
    spins2 = 2*np.pi*(spins)
    deltas = deltac+spins2
    gamma = gperp+gpar/2
    
    sz = -gpar*(gamma**2+deltas**2)/(4*absa2*g**2*gamma+gpar*(gamma**2+deltas**2))
    return sz*pdf

def a_steady_with_sz(eta, deltac, g, spins, gpar, gperp, kappa, szspins):
    spins2 = 2*np.pi*(spins)
    deltas = deltac+spins2
    gamma = gperp+gpar/2
    s = kappa - np.sum(g**2*szspins/(gamma+ 1j*deltas)) + 1j*deltac
    ret = eta/s
    return ret.real, ret.imag

def empty_cavity(eta, deltac, kappa):
    return eta/(kappa+1j*deltac)


def purcell(kappa, gcoll, delta, nreal):
    return kappa*gcoll**2/(nreal*(kappa**2/4+delta**2))*1e6 #1e6 to transform frm MHz to Hz

#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.special import gamma
from scipy.sparse.lil import lil_matrix
from scipy import integrate
import scipy.io
import math
from _helper_functions import find_nearest_sorted
#from numba import jit

#@jit
def jacobian_mbes(Y, t, *args):
    kappa  = args[-6]
    deltac  = args[-5]
    gs = args[-4]
    gperp = args[-3]
    spins = args[-2]
    gpar = args[-1]
    gamma = (gperp+gpar/2)
    deltas = spins*2*np.pi+deltac
    leny = len(Y)
    indices0 = np.arange(2, leny, 4)
    indices1 = np.arange(3, leny, 4)
    indices2 = np.arange(4, leny, 4)
    indices3 = np.arange(5, leny, 4)
    J = np.zeros((leny, leny))
    J[0,0] = -kappa
    J[0,1] = deltac
    J[0, 2::4] = gs
    J[1,0] = -deltac
    J[1,1] = -kappa
    J[1,3::4] = gs
    J[2::4, 0] = gs*Y[4::4]
    J[3::4, 0] = gs*Y[5::4]
    J[4::4, 0] = -4*gs*Y[2::4]
    J[2::4, 1] = -gs*Y[5::4]
    J[3::4, 1] = gs*Y[4::4]
    J[4::4, 1] = -4*gs*Y[3::4]
    J[(indices0, indices0)] = -gamma
    J[(indices1, indices1)] = -gamma
    J[(indices2, indices2)] = -gpar
    J[(indices3, indices3)] = -gpar
    J[(indices0, indices1)] = deltas
    J[(indices1, indices0)] = -deltas
    J[(indices0, indices2)] = gs*Y[0]
    J[(indices0, indices3)] = -gs*Y[1]
    J[(indices1, indices2)] = gs*Y[1]
    J[(indices1, indices3)] = gs*Y[0]
    J[(indices2, indices0)] = -4*gs*Y[0]
    J[(indices2, indices1)] = -4*gs*Y[1]
    return J

def jacobian_mbes_masing(Y, t, *args):
    pump = args[-7]
    kappa  = args[-6]
    deltac  = args[-5]
    gs = args[-4]
    gperp = args[-3]
    spins = args[-2]
    gpar = args[-1]
    gamma = (gperp+gpar/2+pump/2)
    deltas = spins*2*np.pi+deltac
    leny = len(Y)
    indices0 = np.arange(2, leny, 4)
    indices1 = np.arange(3, leny, 4)
    indices2 = np.arange(4, leny, 4)
    indices3 = np.arange(5, leny, 4)
    J = np.zeros((leny, leny))
    J[0,0] = -kappa
    J[0,1] = deltac
    J[0, 2::4] = gs
    J[1,0] = -deltac
    J[1,1] = -kappa
    J[1,3::4] = gs
    J[2::4, 0] = gs*Y[4::4]
    J[3::4, 0] = gs*Y[5::4]
    J[4::4, 0] = -4*gs*Y[2::4]
    J[2::4, 1] = -gs*Y[5::4]
    J[3::4, 1] = gs*Y[4::4]
    J[4::4, 1] = -4*gs*Y[3::4]
    J[(indices0, indices0)] = -gamma
    J[(indices1, indices1)] = -gamma
    J[(indices2, indices2)] = -gpar-pump
    J[(indices3, indices3)] = -gpar-pump
    J[(indices0, indices1)] = deltas
    J[(indices1, indices0)] = -deltas
    J[(indices0, indices2)] = gs*Y[0]
    J[(indices0, indices3)] = -gs*Y[1]
    J[(indices1, indices2)] = gs*Y[1]
    J[(indices1, indices3)] = gs*Y[0]
    J[(indices2, indices0)] = -4*gs*Y[0]
    J[(indices2, indices1)] = -4*gs*Y[1]
    return J

def jacobian_mbes_hp(Y, t, *args):
    kappa  = args[-6]
    deltac  = args[-5]
    gs = args[-4]
    gperp = args[-3]
    spins = args[-2]
    gpar = args[-1]
    gamma = (gperp+gpar/2)
    deltas = spins*2*np.pi+deltac
    leny = len(Y)
    indices0 = np.arange(2, leny, 2)
    indices1 = np.arange(3, leny, 2)
    J = np.zeros((leny, leny))
    J[0,0] = -kappa
    J[0,1] = deltac
    J[0, 2::2] = gs
    J[1,0] = -deltac
    J[1,1] = -kappa
    J[1,3::2] = gs
    J[2::2, 0] = -gs
    J[3::2, 1] = -gs
    J[(indices0, indices0)] = -gamma
    J[(indices1, indices1)] = -gamma
    J[(indices0, indices1)] = deltas
    J[(indices1, indices0)] = -deltas
    return J

def jacobian_mbes_no_cavity(t, Y, *args):
    areal = args[-8]
    aimag = args[-7]
    tlist  = args[-6]
    deltac  = args[-5]
    gs = args[-4]
    gperp = args[-3]
    spins = args[-2]
    gpar = args[-1]
    gamma = (gperp+gpar/2)
    deltas = spins*2*np.pi+deltac
    idx = find_nearest_sorted(tlist, t)
    are = areal[idx]
    aim = areal[idx]
    leny = len(Y)
    indices0 = np.arange(0, leny, 4)
    indices1 = np.arange(1, leny, 4)
    indices2 = np.arange(2, leny, 4)
    indices3 = np.arange(3, leny, 4)
    J = np.zeros((leny, leny))
    J[(indices0, indices0)] = -gamma
    J[(indices1, indices1)] = -gamma
    J[(indices2, indices2)] = -gpar
    J[(indices3, indices3)] = -gpar
    J[(indices0, indices1)] = deltas
    J[(indices1, indices0)] = -deltas
    J[(indices0, indices2)] = gs*are
    J[(indices0, indices3)] = -gs*aim
    J[(indices1, indices2)] = gs*aim
    J[(indices1, indices3)] = gs*are
    J[(indices2, indices0)] = -4*gs*are
    J[(indices2, indices1)] = -4*gs*aim
    return J

def jacobian_mbes_2ndorder_real(t, Y, *args):
    eta, kappa, gperp, gpar, spins, deltac, gs, gmat, indexi, indexj, indexi2, indexj2, indexswitch, indexred, indexswitchred, gmatsmspre, gmatsmspim, indexdouble, maskre, maskim, gmatreidx, gmatimidx,gmatidx, gmatjacre, gmatjacim, gmatjacre2, gmatjacim2, gstiled, gstiled2, gstiled3, gstiled4, gstiledswitch, gparmat, gparmat2, J = args
    
    nspins = len(spins)
    gsreal = np.real(gs)
    gmatreal = np.real(gmat)
    gmatsmsprereal = np.real(gmatsmspre)
    gmatsmspimreal = np.real(gmatsmspim)
    deltas = spins*2*np.pi+deltac
    gamma = gperp+gpar/2
    stepsize = nspins*(nspins-1)
    
    are = Y[0]
    aim = Y[1]
    adagare = Y[2]
    adagaim = Y[3]
    aare = Y[4]
    aaim = Y[5]

    asmre = Y[6:(nspins*2+6):2]
    asmim = Y[7:(nspins*2+6):2]
    
    adagsmre = Y[(nspins*2+6):(nspins*4+6):2]
    adagsmim = Y[(nspins*2+6)+1:(nspins*4+6):2]
    
    aszre = Y[(nspins*4+6):(nspins*6+6):2]
    aszim = Y[(nspins*4+6)+1:(nspins*6+6):2]
    
    smre = Y[(nspins*6+6):(nspins*8+6):2]
    smim = Y[(nspins*6+6)+1:(nspins*8+6):2]

    szre = Y[(nspins*8+6):(nspins*10+6):2]
    szim = Y[(nspins*8+6)+1:(nspins*10+6):2]

    smsmre = Y[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2]
    smsmim = Y[(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2]
    
    smspre = Y[((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2]
    smspim = Y[((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2]

    szsmre = Y[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2]
    szsmim = Y[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2]
    
    szszre = Y[((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2]
    szszim = Y[((nspins*10+6)+4*nspins*(nspins-1))+1:((nspins*10+6)+5*nspins*(nspins-1)):2]
        
    szre1 = np.zeros(gmatreidx.shape)
    szre2 = np.zeros(gmatreidx.shape)
    smre1 = np.zeros(gmatreidx.shape)
    smre2 = np.zeros(gmatreidx.shape)
    smim1 = np.zeros(gmatreidx.shape)
    smim2 = np.zeros(gmatreidx.shape)
    asmre1 = np.zeros(gmatreidx.shape)
    asmre2 = np.zeros(gmatreidx.shape)
    asmim1 = np.zeros(gmatreidx.shape)
    asmim2 = np.zeros(gmatreidx.shape)
    adagsmre1 = np.zeros(gmatreidx.shape)
    adagsmre2 = np.zeros(gmatreidx.shape)
    adagsmim1 = np.zeros(gmatreidx.shape)
    adagsmim2 = np.zeros(gmatreidx.shape)
    aszre1 = np.zeros(gmatreidx.shape)
    aszre2 = np.zeros(gmatreidx.shape)
    aszim1 = np.zeros(gmatreidx.shape)
    aszim2 = np.zeros(gmatreidx.shape)
    
    for i in range(nspins):
        idx0 = np.where(gmatreidx[i] == -1)
        idx1 = np.where(gmatimidx[i] == -1)
        
        temp = szre[gmatreidx[i]]
        temp[idx0] = 0.
        szre1[i, :] = temp
        
        temp = szre[gmatimidx[i]]
        temp[idx1] = 0.
        szre2[i, :] = temp
        
        temp = smre[gmatreidx[i]]
        temp[idx0] = 0.
        smre1[i, :] = temp
        
        temp = smre[gmatimidx[i]]
        temp[idx1] = 0.
        smre2[i, :] = temp
        
        temp = smim[gmatreidx[i]]
        temp[idx0] = 0.
        smim1[i, :] = temp
        
        temp = smim[gmatimidx[i]]
        temp[idx1] = 0.
        smim2[i, :] = temp
        
        temp = asmre[gmatreidx[i]]
        temp[idx0] = 0.
        asmre1[i, :] = temp
        
        temp = asmre[gmatimidx[i]]
        temp[idx1] = 0.
        asmre2[i, :] = temp
        
        temp = adagsmre[gmatreidx[i]]
        temp[idx0] = 0.
        adagsmre1[i, :] = temp
        
        temp = adagsmre[gmatimidx[i]]
        temp[idx1] = 0.
        adagsmre2[i, :] = temp
        
        temp = adagsmim[gmatreidx[i]]
        temp[idx0] = 0.
        adagsmim1[i, :] = temp
        
        temp = adagsmim[gmatimidx[i]]
        temp[idx1] = 0.
        adagsmim2[i, :] = temp
        
        temp = asmim[gmatreidx[i]]
        temp[idx0] = 0.
        asmim1[i, :] = temp
        
        temp = asmim[gmatimidx[i]]
        temp[idx1] = 0.
        asmim2[i, :] = temp
        
        temp = aszre[gmatreidx[i]]
        temp[idx0] = 0.
        aszre1[i, :] = temp
        
        temp = aszre[gmatimidx[i]]
        temp[idx1] = 0.
        aszre2[i, :] = temp
        
        temp = aszim[gmatreidx[i]]
        temp[idx0] = 0.
        aszim1[i, :] = temp
        
        temp = aszim[gmatimidx[i]]
        temp[idx1] = 0.
        aszim2[i, :] = temp
    
    
    smref = np.zeros(gstiled2.shape)
    smimf = np.zeros(gstiled2.shape)
    szref = np.zeros(gstiled2.shape)
    smref2 = np.zeros(gstiled2.shape)
    adagsmref = np.zeros(gstiled2.shape)
    adagsmref2 = np.zeros(gstiled2.shape)
    adagsmimf2 = np.zeros(gstiled2.shape)
    asmref2 =  np.zeros(gstiled2.shape)
    asmimf2 =  np.zeros(gstiled2.shape)
    aszref = np.zeros(gstiled2.shape)
    aszref2 = np.zeros(gstiled2.shape)
    aszimf = np.zeros(gstiled2.shape)
    aszimf2 = np.zeros(gstiled2.shape)
    smimf2 = np.zeros(gstiled2.shape)
    szref2 = np.zeros(gstiled2.shape)
    szref3 = np.zeros(gstiled2.shape)
    smimf = np.zeros(gstiled2.shape)
    ctr = -1
    ctr2 = 1
    for i, row in enumerate(gstiled2):
        if i % (nspins-1) == 0:
            ctr += 1
            ctr2 = 0
        
        if ctr == ctr2:
            ctr2 += 1
        smref[i] = gstiled2[i]*smre[ctr]
        adagsmref[i] = gstiled2[i]*adagsmre[ctr]
        smimf[i] = gstiled2[i]*smim[ctr]
        smref2[i, ctr] = smre[ctr2]*gsreal[ctr]
        adagsmref2[i, ctr] = adagsmre[ctr2]*gsreal[ctr]
        adagsmimf2[i, ctr] = adagsmim[ctr2]*gsreal[ctr]
        asmref2[i, ctr] = asmre[ctr2]*gsreal[ctr]
        asmimf2[i, ctr] = asmim[ctr2]*gsreal[ctr]
        smimf2[i, ctr] = smim[ctr2]*gsreal[ctr]
        szref2[i, ctr] = szre[ctr2]*gsreal[ctr2]
        smimf[i] = gstiled2[i]*smim[ctr]
        szref3[i] = gstiled4[i]*szre[ctr]
        aszref2[i, ctr] = gsreal[ctr2]*aszre[ctr2]
        aszref[i] = gstiled4[i]*aszre[ctr]
        aszimf2[i, ctr] = gsreal[ctr2]*aszim[ctr2]
        aszimf[i] = gstiled4[i]*aszim[ctr]
        szref[i, ctr] = gsreal[ctr2]*szre[ctr2]
        ctr2 += 1
        

    smspim1 = smspim[indexdouble]
    spsmim1 = -smspim[indexdouble]
    smspim1[np.where(maskre==0)] = 0.
    spsmim1[np.where(maskim==0)] = 0.
    
    J[0,0] = -kappa #are
    J[0,1] = deltac #aim
    J[0,2:] = 0. #adagare
    J[0,(nspins*6+6):(nspins*8+6):2] = gsreal #smre



    J[1,0] = -deltac #are
    J[1,1] = -kappa #aim
    J[1,(nspins*6+6)+1:(nspins*8+6):2] = gsreal #smim


    J[2,0] = 2*eta #are
    J[2,1] = 0. #aim
    J[2,2] = -2*kappa #adagare
    J[2,3:] = 0. #adagaim
    J[2,(nspins*2+6):(nspins*4+6):2] = 2*gsreal #adagsmre


    J[3,:] = 0. #are
    J[3,3] = -2*kappa #adagaim


    J[4,0] = 2*eta #are
    J[4,1:] = 0. #aim
    J[4,4] = -2*kappa #aare
    J[4,5] = 2*deltac #aaim
    J[4,6:(nspins*2+6):2] = 2*gsreal #asmre


    J[5,:] = 0. #are
    J[5,1] = 2*eta #aim
    J[5,4] = -2*deltac #aare
    J[5,5] = -2*kappa #aaim
    J[5,7:(nspins*2+6):2] = 2*gsreal #asmim


    #asmre
    J[6:(nspins*2+6):2,0] = 2*aszre*gsreal-4*gsreal*are*szre #are
    J[6:(nspins*2+6):2,1] = -2*aszim*gsreal+4*aim*gsreal*szre #aim
    J[6:(nspins*2+6):2,2:] = 0. #adagare
    J[6:(nspins*2+6):2,4] = gsreal*szre #aare
    np.fill_diagonal(J[6:(nspins*2+6):2,6:(nspins*2+6):2], -gamma-kappa) #asmre
    np.fill_diagonal(J[6:(nspins*2+6):2,7:(nspins*2+6):2], deltac+deltas) #asmim
    np.fill_diagonal(J[6:(nspins*2+6):2,(nspins*4+6):(nspins*6+6):2], 2*are*gsreal)
    np.fill_diagonal(J[6:(nspins*2+6):2,(nspins*4+6)+1:(nspins*6+6):2], -2*aim*gsreal)
    np.fill_diagonal(J[6:(nspins*2+6):2,(nspins*6+6):(nspins*8+6):2], eta)
    np.fill_diagonal(J[6:(nspins*2+6):2,(nspins*8+6):(nspins*10+6):2], aare*gsreal+2*aim**2*gsreal-2*are**2*gsreal)
    J[6:(nspins*2+6):2,(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2] = gmatsmsprereal + gmatsmspimreal
    
    #asmim
    J[7:(nspins*2+6):2,0] = 2*aszim*gsreal-4*aim*gsreal*szre #are
    J[7:(nspins*2+6):2,1] = 2*aszre*gsreal-4*are*gsreal*szre #aim
    J[7:(nspins*2+6):2,2:] = 0. #adagare
    J[7:(nspins*2+6):2,5] = gsreal*szre #aaim
    np.fill_diagonal(J[7:(nspins*2+6):2,6:(nspins*2+6):2], -deltac-deltas)
    np.fill_diagonal(J[7:(nspins*2+6):2,7:(nspins*2+6):2], -gamma-kappa)            
    np.fill_diagonal(J[7:(nspins*2+6):2,(nspins*4+6):(nspins*6+6):2], 2*aim*gsreal)
    np.fill_diagonal(J[7:(nspins*2+6):2,(nspins*4+6)+1:(nspins*6+6):2], 2*are*gsreal)
    np.fill_diagonal(J[7:(nspins*2+6):2,(nspins*6+6)+1:(nspins*8+6):2], eta)
    np.fill_diagonal(J[7:(nspins*2+6):2,(nspins*8+6):(nspins*10+6):2], aaim*gsreal-4*aim*are*gsreal)
    J[7:(nspins*2+6):2,(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2] = gmatsmsprereal + gmatsmspimreal #smsmim

    
    #adagsmre
    J[(nspins*2+6):(nspins*4+6):2,0] = 2*aszre*gsreal-4*are*gsreal*szre #are
    J[(nspins*2+6):(nspins*4+6):2,1] = 2*aszim*gsreal-4*aim*gsreal*szre #aim
    J[(nspins*2+6):(nspins*4+6):2,2] = gsreal*szre #adagare
    J[(nspins*2+6):(nspins*4+6):2,3:] = 0. #adagaim
    np.fill_diagonal(J[(nspins*2+6):(nspins*4+6):2,(nspins*2+6):(nspins*4+6):2], -gamma-kappa)
    np.fill_diagonal(J[(nspins*2+6):(nspins*4+6):2,(nspins*2+6)+1:(nspins*4+6):2], -deltac+deltas)
    np.fill_diagonal(J[(nspins*2+6):(nspins*4+6):2,(nspins*4+6):(nspins*6+6):2], 2*are*gsreal)
    np.fill_diagonal(J[(nspins*2+6):(nspins*4+6):2,(nspins*4+6)+1:(nspins*6+6):2], 2*aim*gsreal)
    np.fill_diagonal(J[(nspins*2+6):(nspins*4+6):2,(nspins*6+6):(nspins*8+6):2], eta)
    np.fill_diagonal(J[(nspins*2+6):(nspins*4+6):2,(nspins*8+6):(nspins*10+6):2], gsreal*0.5+adagare*gsreal-2*aim**2*gsreal-2*are**2*gsreal)
    J[(nspins*2+6):(nspins*4+6):2,((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2] = gmatsmsprereal + gmatsmspimreal #smspre    


    #adagsmim
    J[(nspins*2+6)+1:(nspins*4+6):2,:] = 0. #are
    J[(nspins*2+6)+1:(nspins*4+6):2,3] = gsreal*szre #adagaim
    np.fill_diagonal(J[(nspins*2+6)+1:(nspins*4+6):2,(nspins*2+6):(nspins*4+6):2], deltac-deltas)
    np.fill_diagonal(J[(nspins*2+6)+1:(nspins*4+6):2,(nspins*2+6)+1:(nspins*4+6):2], -gamma-kappa)
    np.fill_diagonal(J[(nspins*2+6)+1:(nspins*4+6):2,(nspins*6+6)+1:(nspins*8+6):2], eta)
    np.fill_diagonal(J[(nspins*2+6)+1:(nspins*4+6):2,(nspins*8+6):(nspins*10+6):2], adagaim*gsreal)
    J[(nspins*2+6)+1:(nspins*4+6):2,((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2] = gmatsmsprereal-gmatsmspimreal #smspim

    
    #aszre
    J[(nspins*4+6):(nspins*6+6):2,0] = -gpar-2*gsreal*(3*adagsmre+asmre-4*aim*smim-8*are*smre) #are
    J[(nspins*4+6):(nspins*6+6):2,1] = -2*gsreal*(adagsmim+asmim-4*are*smim) #aim
    J[(nspins*4+6):(nspins*6+6):2,2] = -2*gsreal*smre #adagare
    J[(nspins*4+6):(nspins*6+6):2,3] = 2*gsreal*smim #adagaim
    J[(nspins*4+6):(nspins*6+6):2,4] = -2*gsreal*smre #aare
    J[(nspins*4+6):(nspins*6+6):2,5] = -2*gsreal*smim #aaim
    J[(nspins*4+6):(nspins*6+6):2,6:] = 0. #asmre
    np.fill_diagonal(J[(nspins*4+6):(nspins*6+6):2,6:(nspins*2+6):2], -2*are*gsreal)
    np.fill_diagonal(J[(nspins*4+6):(nspins*6+6):2,7:(nspins*2+6):2], -2*aim*gsreal)
    np.fill_diagonal(J[(nspins*4+6):(nspins*6+6):2,(nspins*2+6):(nspins*4+6):2], -6*are*gsreal)
    np.fill_diagonal(J[(nspins*4+6):(nspins*6+6):2,(nspins*2+6)+1:(nspins*4+6):2], -2*aim*gsreal)
    np.fill_diagonal(J[(nspins*4+6):(nspins*6+6):2,(nspins*4+6):(nspins*6+6):2], -gpar-kappa)
    np.fill_diagonal(J[(nspins*4+6):(nspins*6+6):2,(nspins*4+6)+1:(nspins*6+6):2], deltac)
    np.fill_diagonal(J[(nspins*4+6):(nspins*6+6):2,(nspins*6+6):(nspins*8+6):2], -gsreal*(1+2*aare+2*adagare-8*are**2))
    np.fill_diagonal(J[(nspins*4+6):(nspins*6+6):2,(nspins*6+6)+1:(nspins*8+6):2], gsreal*(-2*aaim+2*adagaim+8*aim*are))
    np.fill_diagonal(J[(nspins*4+6):(nspins*6+6):2,(nspins*8+6):(nspins*10+6):2], eta)
    J[(nspins*4+6):(nspins*6+6):2,((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2] = gmatreal #szsmre

    #aszim
    J[(nspins*4+6)+1:(nspins*6+6):2,0] = gsreal*(2*adagsmim - 2*asmim + 8*aim*smre) #are
    J[(nspins*4+6)+1:(nspins*6+6):2,1] = -gpar + gsreal*(-6*adagsmre+2*asmre+16*aim*smim+8*are*smre) #aim
    J[(nspins*4+6)+1:(nspins*6+6):2,2] = -2*gsreal*smim #adagare
    J[(nspins*4+6)+1:(nspins*6+6):2,3] = -2*gsreal*smre #adagaim
    J[(nspins*4+6)+1:(nspins*6+6):2,4] = 2*gsreal*smim #aare
    J[(nspins*4+6)+1:(nspins*6+6):2,5] = -2*gsreal*smre #aaim
    J[(nspins*4+6)+1:(nspins*6+6):2,6:] = 0.  
    np.fill_diagonal(J[(nspins*4+6)+1:(nspins*6+6):2,6:(nspins*2+6):2], 2*aim*gsreal)
    np.fill_diagonal(J[(nspins*4+6)+1:(nspins*6+6):2,7:(nspins*2+6):2], -2*are*gsreal)
    np.fill_diagonal(J[(nspins*4+6)+1:(nspins*6+6):2,(nspins*2+6):(nspins*4+6):2], -6*aim*gsreal)
    np.fill_diagonal(J[(nspins*4+6)+1:(nspins*6+6):2,(nspins*2+6)+1:(nspins*4+6):2], 2*are*gsreal)
    np.fill_diagonal(J[(nspins*4+6)+1:(nspins*6+6):2,(nspins*4+6):(nspins*6+6):2], -deltac)
    np.fill_diagonal(J[(nspins*4+6)+1:(nspins*6+6):2,(nspins*4+6)+1:(nspins*6+6):2], -gpar-kappa)
    np.fill_diagonal(J[(nspins*4+6)+1:(nspins*6+6):2,(nspins*6+6):(nspins*8+6):2], -2*aaim*gsreal-2*adagaim*gsreal+8*aim*are*gsreal)
    np.fill_diagonal(J[(nspins*4+6)+1:(nspins*6+6):2,(nspins*6+6)+1:(nspins*8+6):2], -gsreal*(1-2*aare+2*adagare-8*aim**2))
    J[(nspins*4+6)+1:(nspins*6+6):2,((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2] = gmatreal #szsmim        
                    
    #smre
    J[(nspins*6+6):(nspins*8+6):2,:] = 0. #are
    np.fill_diagonal(J[(nspins*6+6):(nspins*8+6):2,(nspins*4+6):(nspins*6+6):2], gsreal)
    np.fill_diagonal(J[(nspins*6+6):(nspins*8+6):2,(nspins*6+6):(nspins*8+6):2], -gamma)
    np.fill_diagonal(J[(nspins*6+6):(nspins*8+6):2,(nspins*6+6)+1:(nspins*8+6):2], deltas)

                     
    #smim
    J[(nspins*6+6)+1:(nspins*8+6):2,:] = 0. #are
    np.fill_diagonal(J[(nspins*6+6)+1:(nspins*8+6):2,(nspins*4+6)+1:(nspins*6+6):2], gsreal)
    np.fill_diagonal(J[(nspins*6+6)+1:(nspins*8+6):2,(nspins*6+6):(nspins*8+6):2], -deltas)
    np.fill_diagonal(J[(nspins*6+6)+1:(nspins*8+6):2,(nspins*6+6)+1:(nspins*8+6):2], -gamma)
   
    #szre
    J[(nspins*8+6):(nspins*10+6):2,:] = 0.
    np.fill_diagonal(J[(nspins*8+6):(nspins*10+6):2,(nspins*2+6):(nspins*4+6):2], -4*gsreal)
    np.fill_diagonal(J[(nspins*8+6):(nspins*10+6):2,(nspins*8+6):(nspins*10+6):2], -gpar)

    
    #szim
    J[(nspins*8+6)+1:(nspins*10+6):2,:] = 0. #all columns are 0 for szim


    #smsmre      
    J[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2,0] = gsreal[indexj2]*(-2*smre[indexi2]*szre[indexj2]+szsmre[indexred])+gsreal[indexi2]*(-2*smre[indexj2]*szre[indexi2]+szsmre[indexswitchred]) #are
    J[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2,1] = gsreal[indexj2]*(2*smim[indexi2]*szre[indexj2]-szsmim[indexred])+gsreal[indexi2]*(2*smim[indexj2]*szre[indexi2]-szsmim[indexswitchred]) #aim
    J[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2,2:] = 0. #adagare
    J[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2,6:(nspins*2+6):2] = np.multiply(gmatsmsprereal.transpose()+gmatsmspimreal.transpose(), szre1.transpose()+szre2.transpose()) #asmre
    J[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2,(nspins*4+6):(nspins*6+6):2] = np.multiply(gstiled, smre1.transpose()+smre2.transpose()) #aszre
    J[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2,(nspins*4+6)+1:(nspins*6+6):2] = -np.multiply(gstiled, smim1.transpose()+smim2.transpose()) #aszim
    J[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2,(nspins*6+6):(nspins*8+6):2] = np.multiply(gmatsmsprereal.transpose()+gmatsmspimreal.transpose(), (aszre1.transpose()+aszre2.transpose())-2*are*(szre1.transpose()+szre2.transpose())) #smre
    J[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2,(nspins*6+6)+1:(nspins*8+6):2] = np.multiply(gmatsmsprereal.transpose()+gmatsmspimreal.transpose(), -(aszim1.transpose()+aszim2.transpose())+2*aim*(szre1.transpose()+szre2.transpose())) #smim
    J[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2,(nspins*8+6):(nspins*10+6):2] = np.multiply(gstiled, 2*aim*(smim1.transpose()+smim2.transpose())-2*are*(smre1.transpose()+smre2.transpose())+(asmre1.transpose()+asmre2.transpose())) #szre
    J[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2,(nspins*8+6)+1:(nspins*10+6):2] = 0. #szim
    np.fill_diagonal(J[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2,(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2], -2*gamma) #smsmre
    np.fill_diagonal(J[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2,(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2], deltas[indexj2]+deltas[indexi2]) #smsmim
    J[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2,((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2] = are*(gmatjacre+gmatjacim) #szsmre
    J[(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2,((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2] = -aim*(gmatjacre+gmatjacim) #szsmim


    #smsmim
    J[(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2,0] = gsreal[indexj2]*(-2*smim[indexi2]*szre[indexj2]+szsmim[indexred])+gsreal[indexi2]*(-2*smim[indexj2]*szre[indexi2]+szsmim[indexswitchred]) #are
    J[(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2,1] = gsreal[indexj2]*(-2*smre[indexi2]*szre[indexj2]+szsmre[indexred])+gsreal[indexi2]*(-2*smre[indexj2]*szre[indexi2]+szsmre[indexswitchred]) #aim
    J[(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2,2:] = 0. #adagare
    J[(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2,7:(nspins*2+6):2] = np.multiply(gmatsmsprereal.transpose()+gmatsmspimreal.transpose(), szre1.transpose()+szre2.transpose()) #asmim
    J[(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2,(nspins*4+6):(nspins*6+6):2] = np.multiply(gstiled, smim1.transpose()+smim2.transpose()) #aszre
    J[(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2,(nspins*4+6)+1:(nspins*6+6):2] = np.multiply(gstiled, smre1.transpose()+smre2.transpose()) #aszim
    J[(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2,(nspins*6+6):(nspins*8+6):2] = np.multiply(gmatsmsprereal.transpose()+gmatsmspimreal.transpose(), (aszim1.transpose()+aszim2.transpose())-2*aim*(szre1.transpose()+szre2.transpose())) #smre
    J[(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2,(nspins*6+6)+1:(nspins*8+6):2] = np.multiply(gmatsmsprereal.transpose()+gmatsmspimreal.transpose(), (aszre1.transpose()+aszre2.transpose())-2*are*(szre1.transpose()+szre2.transpose())) #smim
    J[(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2,(nspins*8+6):(nspins*10+6):2] = np.multiply(gstiled, -2*are*(smim1.transpose()+smim2.transpose())-2*aim*(smre1.transpose()+smre2.transpose())+(asmim1.transpose()+asmim2.transpose()))  #szre
    np.fill_diagonal(J[(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2,(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2], -deltas[indexj2]-deltas[indexi2]) #smsmre
    np.fill_diagonal(J[(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2,(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2], -2*gamma) #smsmim
    J[(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2,((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2] = aim*(gmatjacre+gmatjacim) #szsmre
    J[(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2,((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2] = are*(gmatjacre+gmatjacim) #szsmim
    
    #smspre
    J[((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2, 0] = gsreal[indexj2]*(-2*smre[indexi2]*szre[indexj2]+szsmre[indexred])+gsreal[indexi2]*(-2*smre[indexj2]*szre[indexi2]+szsmre[indexswitchred])  #are
    J[((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2,1] = gsreal[indexj2]*(-2*smim[indexi2]*szre[indexj2]+szsmim[indexred])+gsreal[indexi2]*(-2*smim[indexj2]*szre[indexi2]+szsmim[indexswitchred])  #aim
    J[((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2,2:] = 0. #adagare
    J[((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2,(nspins*2+6):(nspins*4+6):2] = np.multiply(gmatsmsprereal.transpose()+gmatsmspimreal.transpose(), szre1.transpose()+szre2.transpose()) #adagsmre
    J[((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2,(nspins*4+6):(nspins*6+6):2] = np.multiply(gstiled, smre1.transpose()+smre2.transpose()) #aszre
    J[((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2,(nspins*4+6)+1:(nspins*6+6):2] = np.multiply(gstiled, smim1.transpose()+smim2.transpose()) #aszim
    J[((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2,(nspins*6+6):(nspins*8+6):2] = np.multiply(gmatsmsprereal.transpose()+gmatsmspimreal.transpose(), (aszre1.transpose()+aszre2.transpose())-2*are*(szre1.transpose()+szre2.transpose())) #smre
    J[((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2,(nspins*6+6)+1:(nspins*8+6):2] = np.multiply(gmatsmsprereal.transpose()+gmatsmspimreal.transpose(), (aszim1.transpose()+aszim2.transpose())-2*aim*(szre1.transpose()+szre2.transpose())) #smim
    J[((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2,(nspins*8+6):(nspins*10+6):2] = np.multiply(gstiled, -2*are*(smre1.transpose()+smre2.transpose())-2*aim*(smim1.transpose()+smim2.transpose())+(adagsmre1.transpose()+adagsmre2.transpose())) #szre
    np.fill_diagonal(J[((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2,((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2], -2*gamma)#smspre
    np.fill_diagonal(J[((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2,((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2], deltas[indexj2]-deltas[indexi2]) #smspim
    J[((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2,((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2] = are*(gmatjacre+gmatjacim) #szsmre
    J[((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2,((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2] = aim*(gmatjacre+gmatjacim)#szsmim

    #smspim
    J[((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2, 0] = gsreal[indexj2]*(2*smim[indexi2]*szre[indexj2]-szsmim[indexred])+gsreal[indexi2]*(-2*smim[indexj2]*szre[indexi2]+szsmim[indexswitchred])  #are
    J[((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2, 1] = gsreal[indexj2]*(-2*smre[indexi2]*szre[indexj2]+szsmre[indexred])+gsreal[indexi2]*(2*smre[indexj2]*szre[indexi2]-szsmre[indexswitchred])  #aim
    J[((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2, 2:] = 0. #adagare
    J[((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2, (nspins*2+6)+1:(nspins*4+6):2] = np.multiply(gmatsmsprereal.transpose()-gmatsmspimreal.transpose(), szre1.transpose()+szre2.transpose()) #adagsmim
    J[((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2, (nspins*4+6):(nspins*6+6):2] = np.multiply(gstiled, -smim1.transpose()+smim2.transpose()) #aszre
    J[((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2, (nspins*4+6)+1:(nspins*6+6):2] = np.multiply(gstiled, smre1.transpose()-smre2.transpose()) #aszim
    J[((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2, (nspins*6+6):(nspins*8+6):2] = np.multiply(gmatsmsprereal.transpose()+gmatsmspimreal.transpose(), (-aszim1.transpose()+aszim2.transpose())+2*aim*(szre1.transpose()-szre2.transpose())) #smre
    J[((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2, (nspins*6+6)+1:(nspins*8+6):2] = np.multiply(gmatsmsprereal.transpose()+gmatsmspimreal.transpose(), (aszre1.transpose()-aszre2.transpose())-2*are*(szre1.transpose()-szre2.transpose())) #smim
    J[((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2, (nspins*8+6):(nspins*10+6):2] = np.multiply(gstiled, 2*are*(smim1.transpose()-smim2.transpose())-2*aim*(smre1.transpose()-smre2.transpose())-(adagsmim1.transpose()-adagsmim2.transpose())) #szre
    np.fill_diagonal(J[((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2, ((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2], -deltas[indexj2]+deltas[indexi2]) #smspre
    np.fill_diagonal(J[((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2, ((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2], -2*gamma) #smspim
    J[((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2, ((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2] = aim*(gmatjacre-gmatjacim) #szsmre
    J[((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2, ((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2] = -are*(gmatjacre-gmatjacim) #szsmim 

    
    #szsmre
    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,0] = gsreal[indexj]*(8*smre[indexi]*smre[indexj]-2*smsmre[indexdouble]-2*smspre[indexdouble])+gsreal[indexi]*(-2*szre[indexj]*szre[indexi]+szszre[indexdouble])  #are
    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,1] = gsreal[indexj]*(8*smim[indexj]*smre[indexi]-2*smsmim[indexdouble]-2*(smspim1+spsmim1)) #aim
    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,2:] = 0. #adagare

    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,6:(nspins*2+6):2] = -2*smref #asmre
    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,7:(nspins*2+6):2] = -2*smimf #asmim
    
    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*2+6):(nspins*4+6):2] = -4*smref2-2*smref #adagsmre
    
    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*2+6)+1:(nspins*4+6):2] = 2*smimf #adagsmim
    
    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*4+6):(nspins*6+6):2] = szref2 +szref3#aszre
    
    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*6+6):(nspins*8+6):2] = -2*adagsmref2-2*asmref2+8*are*smref2 -4*adagsmref+8*aim*smimf+8*are*smref-gparmat2*gpar#smre
    
    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*6+6)+1:(nspins*8+6):2] = 2*adagsmimf2-2*asmimf2+8*aim*smref2 #smim

    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*8+6):(nspins*10+6):2] = aszref2-2*are*szref+aszref-2*are*szref3 #szre
    
    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2] = -2*are*gstiled3 #smsmre
    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2] = -2*aim*gstiled3 #smsmim
    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2] = -2*are*gstiled3 #smspre
    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2] = -2*aim*gstiledswitch #smspim
    
    np.fill_diagonal(J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2], -gpar - gamma) #szsmre
    np.fill_diagonal(J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2], deltas[indexi]) #szsmim
    J[((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2,((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2] = are*(gmatjacre2.transpose()+gmatjacim2.transpose()) #szszre

    
    #szsmim
    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,0] = gsreal[indexj]*(8*smre[indexj]*smim[indexi]-2*smsmim[indexdouble]+2*(smspim1+spsmim1)) #are
    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,1] = gsreal[indexj]*(8*smim[indexi]*smim[indexj]+2*smsmre[indexdouble]-2*smspre[indexdouble])+gsreal[indexi]*(-2*szre[indexj]*szre[indexi]+szszre[indexdouble]) #aim
    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,2:] = 0. #adagare
    
    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,6:(nspins*2+6):2] = 2*smimf #asmre
    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,7:(nspins*2+6):2] = -2*smref #asmim
    
    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*2+6):(nspins*4+6):2] = -4*smimf2-2*smimf #adagsmre
    
    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*2+6)+1:(nspins*4+6):2] = -2*smref #adagsmim
    
    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*4+6)+1:(nspins*6+6):2] = szref2 +szref3 #aszim
    
    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*6+6):(nspins*8+6):2] = -2*adagsmimf2-2*asmimf2+8*are*smimf2 #smre
    
    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*6+6)+1:(nspins*8+6):2] = -2*adagsmref2+2*asmref2+8*aim*smimf2 -4*adagsmref+8*aim*smimf+8*are*smref-gparmat2*gpar #smim

    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*8+6):(nspins*10+6):2] = aszimf2-2*aim*szref+aszimf-2*aim*szref3 #szre

    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*10+6):((nspins*10+6)+nspins*(nspins-1)):2] = 2*aim*gstiled3 #smsmre
    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,(nspins*10+6)+1:((nspins*10+6)+nspins*(nspins-1)):2] = -2*are*gstiled3 #smsmim
    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,((nspins*10+6)+nspins*(nspins-1)):((nspins*10+6)+2*nspins*(nspins-1)):2] = -2*aim*gstiled3 #smspre
    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,((nspins*10+6)+nspins*(nspins-1))+1:((nspins*10+6)+2*nspins*(nspins-1)):2] = 2*are*gstiledswitch #smspim
    
    np.fill_diagonal(J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2], -deltas[indexi]) #szsmre
    np.fill_diagonal(J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2], -gpar-gamma) #szsmim
    J[((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2,((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2] = aim*(gmatjacre2.transpose()+gmatjacim2.transpose()) #szszre
    #szszre
    J[((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2,0] = gsreal[indexj2]*(8*smre[indexj2]*szre[indexi2]-4*szsmre[indexswitchred])+gsreal[indexi2]*(8*smre[indexi2]*szre[indexj2]-4*szsmre[indexred])  #are
    J[((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2,1] = gsreal[indexj2]*(8*smim[indexj2]*szre[indexi2]-4*szsmim[indexswitchred])+gsreal[indexi2]*(8*smim[indexi2]*szre[indexj2]-4*szsmim[indexred]) #aim
    J[((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2,2:] = 0. #adagare
    J[((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2,(nspins*2+6):(nspins*4+6):2] = np.multiply(gstiled, -4*(szre1.transpose()+szre2.transpose())) #adagsmre
    J[((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2,(nspins*4+6):(nspins*6+6):2] = np.multiply(gmatsmsprereal.transpose()+gmatsmspimreal.transpose(), -4*(smre1.transpose()+smre2.transpose()))  #aszre
    J[((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2,(nspins*4+6)+1:(nspins*6+6):2] = np.multiply(gmatsmsprereal.transpose()+gmatsmspimreal.transpose(), -4*(smim1.transpose()+smim2.transpose()))  #aszim
    J[((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2,(nspins*6+6):(nspins*8+6):2] = np.multiply(gstiled, -4*(aszre1.transpose()+aszre2.transpose())+8*are*(szre1.transpose()+szre2.transpose()))  #smre
    J[((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2,(nspins*6+6)+1:(nspins*8+6):2] = np.multiply(gstiled, -4*(aszim1.transpose()+aszim2.transpose())+8*aim*(szre1.transpose()+szre2.transpose())) #smim
    np.fill_diagonal(J[((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2,(nspins*8+6):(nspins*10+6):2], -gpar)
    J[((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2,(nspins*8+6):(nspins*10+6):2] = np.multiply(gmatsmsprereal.transpose()+gmatsmspimreal.transpose(), -4*(adagsmre1.transpose()+adagsmre2.transpose())+8*aim*(smim1.transpose()+smim2.transpose())+8*are*(smre1.transpose()+smre2.transpose()))-gparmat*gpar
    #szre
    J[((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2,((nspins*10+6)+2*nspins*(nspins-1)):((nspins*10+6)+4*nspins*(nspins-1)):2] = -4*are*(gmatjacre2 + gmatjacim2) #szsmre
    J[((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2,((nspins*10+6)+2*nspins*(nspins-1))+1:((nspins*10+6)+4*nspins*(nspins-1)):2] = -4*aim*(gmatjacre2 + gmatjacim2) #szsmim
    np.fill_diagonal(J[((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2,((nspins*10+6)+4*nspins*(nspins-1)):((nspins*10+6)+5*nspins*(nspins-1)):2], -2*gpar) #szszre

    #szszim 
    
    J[((nspins*10+6)+4*nspins*(nspins-1))+1:((nspins*10+6)+5*nspins*(nspins-1)):2,:] = 0.

             
    return J
#!/usr/bin/env python
# coding: utf-8


import numpy as np
from _jacobian import *
from numba import jit


# this is as optimized as possible using numpy slicing so calling this function SHOULD be very fast

def mbes(t, Y, *p):
    eta, kappa, deltac, gs, gperp, spins, gpar, ret = p
    sm = Y[1:len(spins)+1] 
    sz = Y[len(spins)+1:]
    a = Y[0]
    ret[0] = -(kappa+1j*deltac)*a+gs.dot(sm)+eta
    ret[1:len(spins)+1] = -(gperp+gpar/2 + 1j*(spins*2*np.pi+deltac))*sm+gs*sz*a
    ret[len(spins)+1:] = -gpar*(1+sz)-2*gs*(sm*np.conj(a)+np.conj(sm)*a)
    return ret


def mbes_hp(t, Y, *p):
    eta, kappa, deltac, gs, gperp, spins, gpar, ret = p
    # the same as above, but with Hollstein-Primakoff approximation applied
    a = Y[0]
    sm = Y[1:]
    ret[0] = -kappa*a-1j*deltac*a+gs.dot(sm)+eta
    ret[1:] = -(gperp+gpar/2 + 1j*(spins*2*np.pi+deltac))*sm-gs*a
    return ret

def mbes_no_cavity(t, Y, *p):
    deltac, gs, gperp, spins, gpar, ret = p
    sm = Y[:len(spins)]
    sz = Y[len(spins):]
    ret[:len(spins)] = -(gperp+gpar/2 + 1j*(spins*2*np.pi+deltac))*sm
    ret[len(spins):] = -gpar*(1+sz)
    return ret

def mbes_soc_no_cavity(t, Y, *p):
    areal, aimag, tlist, deltac, gs, gperp, spins, gpar, ret = p
    idx = (np.abs(tlist-t)).argmin()
    sm = Y[:len(spins)]
    sz = Y[len(spins):]
    ret[:len(spins)] = -(gperp+gpar/2 + 1j*(spins*2*np.pi+deltac)) * \
        sm+gs*sz*(areal[idx]+1j*aimag[idx])
    ret[len(spins):] = -gpar*(1+sz)-2*gs*(sm*(areal[idx]-1j*aimag[idx]
                                       )+np.conj(sm)*(areal[idx]+1j*aimag[idx]))
    return ret

# %%timeit gives
# 10000 loops, best of 3: 45 µs per loop
# for 800 spins it becomes ~70µs. This is probably not the bottleneck


# this function is a lot slower, so be careful when using it

def mbes_soc(t, Y, *p):
    ilist, qlist, tlist, kappa, deltac, gs, gperp, spins, gpar, ret = p
    idx = (np.abs(tlist-t)).argmin()
    sm = Y[1:len(spins)+1] 
    sz = Y[len(spins)+1:]
    a = Y[0]
    ret[0] = -kappa*a-1j*deltac*a+gs.dot(sm)+ilist[idx]-1j*qlist[idx]
    ret[1:len(spins)+1] = -(gperp+gpar/2 + 1j*(spins*2*np.pi+deltac))*sm+gs*sz*a
    ret[len(spins)+1:] = -gpar*(1+sz)-2*gs*(sm*np.conj(a)+np.conj(sm)*a)
    return ret


def mbes_2ndorder(t, Y, *p):
    eta, kappa, gperp, gpar, spins, deltac, gs, gmat, gmat2, indexi, indexj, indexi2, indexj2, indexswitch, indexred, indexswitchred, gmatsmspre, gmatsmspim, indexdouble, maskre, maskim, gmatreidx, gmatimidx,gmatidx, gmatjacre, gmatjacim, gmatjacre2, gmatjacim2, gstiled, gstiled2, gstiled3, gstiled4, gstiledswitch, gparmat, gparmat2, ret = p
    nspins = len(spins)
    deltas = spins*2*np.pi+deltac
    gamma = gperp+gpar/2
    a = Y[0]
    adag = np.conjugate(a)
    adaga = Y[1]
    aa = Y[2]
    adagadag = np.conjugate(aa)
    asm = Y[3:(nspins+3)]
    adagsp = np.conjugate(asm)
    adagsm = Y[(nspins+3):(nspins*2+3)]
    asp = np.conjugate(adagsm)
    asz = Y[(nspins*2+3):(nspins*3+3)]
    adagsz = np.conjugate(asz)
    sm = Y[(nspins*3+3):(nspins*4+3)]
    sp = np.conjugate(sm)
    sz = Y[(nspins*4+3):(nspins*5+3)]
    
    smsm = Y[(nspins*5+3):(nspins*5+3)+nspins*(nspins-1)//2]
    smsp = Y[(nspins*5+3)+nspins*(nspins-1)//2:(nspins*5+3)+nspins*(nspins-1)]
    spsm = np.conjugate(smsp)
    szsm = Y[(nspins*5+3)+nspins*(nspins-1):(nspins*5+3)+2*nspins*(nspins-1)]
    szsp = np.conjugate(szsm)
    szsz = Y[(nspins*5+3)+2*nspins*(nspins-1):(nspins*5+3)+2*nspins*(nspins-1)+nspins*(nspins-1)//2]
    
    smsp1 = smsp[indexdouble]
    spsm1 = spsm[indexdouble]
    
    smsp1[np.where(maskim==0)] = 0.+0j
    spsm1[np.where(maskre==0)] = 0.+0j


    ret[0] = -kappa*a - 1j*deltac*a + gs.dot(sm) + eta
    
    ret[1] = -2*kappa*adaga + eta * (adag+a) + gs.dot(adagsm + asp)
    
    ret[2] = -2*(kappa + 1j*deltac)*aa + 2 * eta*a + 2*gs.dot(asm)

    ret[3:(nspins+3)] = -(kappa+gamma+1j*(deltas+deltac))*asm + eta*sm + gs*(sz*aa+2*asz*a-2*sz*a*a) + (gmatsmspre + gmatsmspim).dot(smsm)

    ret[(nspins+3):(nspins*2+3)] = -(kappa+gamma+1j*2*np.pi*spins)*adagsm + eta*sm  + gs*(asz*adag+adagsz*a+sz*adaga-2*sz*a*adag) + gs*0.5*(sz+1) + gmatsmspre.dot(smsp)+gmatsmspim.dot(spsm)

    ret[(nspins*2+3):(nspins*3+3)] = -(gpar+1j*deltac+kappa)*asz - gpar*a  + eta*sz - \
        gs*sm - 2*gs*(adagsm*a+asm*adag+sp*aa+sm*adaga) - 4*gs*(asp*a-sm*adag*a-sp*a*a) + gmat.dot(szsm)

    ret[(nspins*3+3):(nspins*4+3)] = -(gamma + 1j*deltas)*sm + gs*asz
    
    ret[(nspins*4+3):(nspins*5+3)] = -gpar*(1+sz)-4*gs*(np.real(adagsm))

    ret[(nspins*5+3):(nspins*5+3)+nspins*(nspins-1)//2] = -(2*gamma+1j*(deltas[indexj2]+deltas[indexi2]))*smsm + gs[indexj2]*(asz[indexj2]*sm[indexi2] +
szsm[indexred]*a-2*sz[indexj2]*sm[indexi2]*a+asm[indexi2]*sz[indexj2]) + gs[indexi2]*(asm[indexj2]*sz[indexi2]+asz[indexi2]*sm[indexj2]+szsm[indexswitchred]*a-2*sm[indexj2]*sz[indexi2]*a)

    ret[(nspins*5+3)+nspins*(nspins-1)//2:(nspins*5+3)+nspins*(nspins-1)] = -(2*gamma+1j*(deltas[indexj2]-deltas[indexi2]))*smsp + gs[indexj2]*(szsp[indexred]*a+asz[indexj2]*sp[indexi2]+asp[indexi2]* sz[indexj2]-2*sz[indexj2]*sp[indexi2]*a)+gs[indexi2]*(adagsm[indexj2]*sz[indexi2]+adagsz[indexi2]*sm[indexj2]+szsm[indexswitchred]*adag-2*sm[indexj2]*sz[indexi2]*adag)
#
    ret[(nspins*5+3)+nspins*(nspins-1):(nspins*5+3)+2*nspins*(nspins-1)] = -(1j*deltas[indexi]+gamma+gpar)*szsm - gpar*sm[indexi] + gs[indexi]*(asz[indexi]*sz[indexj]+asz[indexj]*sz[indexi]+szsz[indexdouble]*a-2*sz[indexj]*sz[indexi]*a) - 2*gs[indexj] *(adagsm[indexi]*sm[indexj]+adagsm[indexj]*sm[indexi]+asp[indexj]*sm[indexi]+asm[indexi]*sp[indexj]+smsm[indexdouble]*adag-2*sm[indexj]*sm[indexi]*adag-2*sp[indexj]*sm[indexi]*a+(smsp1+spsm1)*a)
    
    
    ret[(nspins*5+3)+2*nspins*(nspins-1):(nspins*5+3)+2*nspins*(nspins-1)+nspins*(nspins-1)//2] = -gpar*(sz[indexj2]+sz[indexi2])-2*gpar*szsz - \
    2*gs[indexj2]*(adagsz[indexi2]*sm[indexj2]+asz[indexi2]*sp[indexj2]+adagsm[indexj2]*sz[indexi2]+asp[indexj2]*sz[indexi2]+szsp[indexswitchred]*a+szsm[indexswitchred]*adag-2*sp[indexj2]*sz[indexi2]*a-2*sm[indexj2]*sz[indexi2]*adag) - \
    2*gs[indexi2]*(adagsm[indexi2]*sz[indexj2]+asp[indexi2]*sz[indexj2]+adagsz[indexj2]*sm[indexi2]+asz[indexj2]*sp[indexi2]+szsm[indexred]*adag+szsp[indexred]*a-2*sz[indexj2]*sp[indexi2]*a-2*sz[indexj2]*sm[indexi2]*adag)
    return ret


def mbes_soc_2ndorder(t, Y, *p):
    ilist, qlist, tlist, kappa, gperp, gpar, spins, deltac, gs, gmat, gmat2, indexi, indexj, indexi2, indexj2, indexswitch, indexred, indexswitchred, gmatsmspre, gmatsmspim, indexdouble, maskre, maskim, ret = p
    nspins = len(gs)

    deltas = spins*2*np.pi+deltac
    idx = (np.abs(tlist-t)).argmin()
    gamma = gperp+gpar/2
    a = Y[0]
    adag = np.conjugate(a)
    adaga = Y[1]
    aa = Y[2]
    adagadag = np.conjugate(aa)
    asm = Y[3:(nspins+3)]
    adagsp = np.conjugate(asm)
    adagsm = Y[(nspins+3):(nspins*2+3)]
    asp = np.conjugate(adagsm)
    asz = Y[(nspins*2+3):(nspins*3+3)]
    adagsz = np.conjugate(asz)
    sm = Y[(nspins*3+3):(nspins*4+3)]
    sp = np.conjugate(sm)
    sz = Y[(nspins*4+3):(nspins*5+3)]
    
    smsm = Y[(nspins*5+3):(nspins*5+3)+nspins*(nspins-1)//2]
    smsp = Y[(nspins*5+3)+nspins*(nspins-1)//2:(nspins*5+3)+nspins*(nspins-1)]
    spsm = np.conjugate(smsp)
    szsm = Y[(nspins*5+3)+nspins*(nspins-1):(nspins*5+3)+2*nspins*(nspins-1)]
    szsp = np.conjugate(szsm)
    szsz = Y[(nspins*5+3)+2*nspins*(nspins-1):(nspins*5+3)+2*nspins*(nspins-1)+nspins*(nspins-1)//2]
    
    smsp1 = smsp[indexdouble]
    spsm1 = spsm[indexdouble]
    smsp1[np.where(maskim==0)] = 0.+0j
    spsm1[np.where(maskre==0)] = 0.+0j

    ret[0] = -kappa*a - 1j*deltac*a + gs.dot(sm) + ilist[idx] - 1j*qlist[idx]
    
    ret[1] = -2*kappa*adaga + ilist[idx] * \
        (adag+a) - 1j*qlist[idx]*(adag-a) + 2*gs.dot(adagsm.real)
    ret[2] = -2*(kappa + 1j*deltac)*aa + 2 * \
        (ilist[idx]-1j*qlist[idx])*a + 2*gs.dot(asm)

    ret[3:(nspins+3)] = -(kappa+gamma+1j*(deltas+deltac))*asm + (ilist[idx] - 1j*qlist[idx])*sm + \
        gmatsmspre.dot(smsp)+gmatsmspim.dot(spsm) + gs*(asz*adag+adagsz*a+sz*adaga-2*sz*a*adag) + gs*0.5*(sz+1)

    ret[(nspins+3):(nspins*2+3)] = -(kappa+gamma+1j*2*np.pi*spins)*adagsm + (ilist[idx] + 1j*qlist[idx])*sm + \
        gmat.dot(smsp) + gs*(asz*adag+adagsz*a+sz*adaga-2*sz*a*adag) + gs*0.5*(sz+1)

    ret[(nspins*2+3):(nspins*3+3)] = -(gpar+1j*deltac+kappa)*asz - gpar*a + gmat.dot(szsm) + (ilist[idx] - 1j*qlist[idx])*sz - \
        gs*sm - 2*gs*(adagsm*a+asm*adag+sp*aa+sm*adaga) - 4*gs*(asp*a-sm*adag*a-sp*a*a)

    ret[(nspins*3+3):(nspins*4+3)] = -(gamma + 1j*deltas)*sm + gs*asz
    
    ret[(nspins*4+3):(nspins*5+3)] = -gpar*(1+sz)-4*gs*(np.real(adagsm))

    ret[(nspins*5+3):(nspins*5+3)+nspins*(nspins-1)//2] = -(2*gamma+1j*(deltas[indexj2]+deltas[indexi2]))*smsm + gs[indexj2]*(asz[indexj2]*sm[indexi2] +
szsm[indexred]*a-2*sz[indexj2]*sm[indexi2]*a+asm[indexi2]*sz[indexj2]) + gs[indexi2]*(asm[indexj2]*sz[indexi2]+asz[indexi2]*sm[indexj2]+szsm[indexswitchred]*a-2*sm[indexj2]*sz[indexi2]*a)

    ret[(nspins*5+3)+nspins*(nspins-1)//2:(nspins*5+3)+nspins*(nspins-1)] = -(2*gamma+1j*(deltas[indexj2]-deltas[indexi2]))*smsp + gs[indexj2]*(szsp[indexred]*a+asz[indexj2]*sp[indexi2]+asp[indexi2]* sz[indexj2]-2*sz[indexj2]*sp[indexi2]*a)+gs[indexi2]*(adagsm[indexj2]*sz[indexi2]+adagsz[indexi2]*sm[indexj2]+szsm[indexswitchred]*adag-2*sm[indexj2]*sz[indexi2]*adag)
#
    ret[(nspins*5+3)+nspins*(nspins-1):(nspins*5+3)+2*nspins*(nspins-1)] = -(1j*deltas[indexi]+gamma+gpar)*szsm - gpar*sm[indexi] + gs[indexi]*(asz[indexi]*sz[indexj]+asz[indexj]*sz[indexi]+szsz[indexdouble]*a-2*sz[indexj]*sz[indexi]*a) - 2*gs[indexj] *(adagsm[indexi]*sm[indexj]+adagsm[indexj]*sm[indexi]+asp[indexj]*sm[indexi]+asm[indexi]*sp[indexj]+smsm[indexdouble]*adag-2*sm[indexj]*sm[indexi]*adag-2*sp[indexj]*sm[indexi]*a+(smsp1+spsm1)*a)
    
    ret[(nspins*5+3)+2*nspins*(nspins-1):(nspins*5+3)+2*nspins*(nspins-1)+nspins*(nspins-1)//2] = -gpar*(sz[indexj2]+sz[indexi2])-2*gpar*szsz - \
    2*gs[indexj2]*(adagsz[indexi2]*sm[indexj2]+asz[indexi2]*sp[indexj2]+adagsm[indexj2]*sz[indexi2]+asp[indexj2]*sz[indexi2]+szsp[indexswitchred]*a+szsm[indexswitchred]*adag-2*sp[indexj2]*sz[indexi2]*a-2*sm[indexj2]*sz[indexi2]*adag) - \
    2*gs[indexi2]*(adagsm[indexi2]*sz[indexj2]+asp[indexi2]*sz[indexj2]+adagsz[indexj2]*sm[indexi2]+asz[indexj2]*sp[indexi2]+szsm[indexred]*adag+szsp[indexred]*a-2*sz[indexj2]*sp[indexi2]*a-2*sz[indexj2]*sm[indexi2]*adag)
    return ret

def mbes_soc_with_spin_drive(t, Y, *p):
    ilist, qlist, ispins, qspins, tlist, kappa, deltac, gs, gperp, spins, gpar, ret = p
    idx = (np.abs(tlist-t)).argmin()
    sm = Y[1:len(spins)+1] 
    sz = Y[len(spins)+1:]
    a = Y[0]
    ret[0] = -kappa*a-1j*deltac*a+gs.dot(sm)+ilist[idx]-1j*qlist[idx]
    ret[1:len(spins)+1] = -(gperp+gpar/2 + 1j*(spins*2*np.pi+deltac))*sm + \
        gs*sz*a + 1/2*(1j*ispins[idx]-qspins[idx])*sz
    ret[len(spins)+1:] = -gpar*(1+sz)-2*gs*(sm*np.conj(a)+np.conj(sm)*a) - \
        1j*ispins[idx]*(np.conj(sm)-sm)+qspins[idx]*(np.conj(sm)+sm)
    return ret


def mbes_soc_det(t, Y, *p):
    ilist, qlist, delta_list, tlist, kappa, deltac, gs, gperp, spins, gpar, ret = p
    idx = (np.abs(tlist-t)).argmin()
    sm = Y[1:len(spins)+1] 
    sz = Y[len(spins)+1:]
    a = Y[0]
    ret[0] = -kappa*a-1j*deltac*a + \
        gs.dot(sm)+ilist[idx]-1j*qlist[idx]
    ret[1:len(spins)+1] = -(gperp+gpar/2 + 1j*(spins*2*np.pi+deltac +
                  delta_list[idx]))*sm+gs*sz*a
    ret[len(spins)+1:] = -gpar*(1+sz)-2*gs * \
        (sm*np.conj(a)+np.conj(sm)*a)
    return ret


def mbes_det(t, Y, *p):
    delta_list, tlist, kappa, deltac, gs, gperp, spins, gpar, ret = p
    idx = (np.abs(tlist-t)).argmin()
    sm = Y[1:len(spins)+1] 
    sz = Y[len(spins)+1:]
    a = Y[0]
    ret[0] = -kappa*a-1j*deltac*a+gs.dot(sm)
    ret[1:len(spins)+1] = -(gperp+gpar/2 + 1j*(spins*2*np.pi+deltac +
                  delta_list[idx]))*sm+gs*sz*a
    ret[len(spins)+1:] = -gpar*(1+sz)-2*gs * \
        (sm*np.conj(a)+np.conj(sm)*a)
    return ret


def mbes_soc_grape(t, Y, *p):
    ilist, qlist, tlist, kappa, deltac, gs, gperp, spins, gpar, ret = p
    idx = (np.abs(tlist-t)).argmin()
    sm = Y[1:len(spins)+1] 
    sz = Y[len(spins)+1:]
    a = Y[0]
    ret[0] = -kappa*a-1j*deltac*a + \
        gs.dot(sm)+ilist[idx]-1j*qlist[idx]
    ret[1:len(spins)+1] = -(gperp+gpar/2 + 1j*(spins*2*np.pi+deltac)) * \
        sm+gs*sz*a
    ret[len(spins)+1:] = -gpar*(1+sz)-2*gs * \
        (sm*np.conj(a)+np.conj(sm)*a)
    return ret


def mbes_soc_func(t, Y, *p):
    func, func_args, kappa, deltac, gs, gperp, spins, gpar, ret = p
    pi, pq = func(t, *func_args)
    sm = Y[1:len(spins)+1] 
    sz = Y[len(spins)+1:]
    a = Y[0]
    ret[0] = -kappa*a-1j*deltac*a+gs.dot(sm)+pi-1j*pq
    ret[1:len(spins)+1] = -(gperp+gpar/2 + 1j*(spins*2*np.pi+deltac)) * \
        sm+gs*sz*a
    ret[len(spins)+1:] = -gpar*(1+sz)-2*gs * \
        (sm*np.conj(a)+np.conj(sm)*a)
    return ret


def mbes_simulate_losses(t, Y, *p):
    eta, kappain, kappa, deltac, gs, gperp, spins, gpar, ret = p
    sm = Y[1:len(spins)+1] 
    sz = Y[len(spins)+1:]
    a = Y[0]
    ret[0] = -kappa*a-1j*deltac*a + \
        gs.dot(sm)+np.sqrt(2*kappain)*eta
    ret[1:len(spins)+1] = -(gperp+gpar/2 + 1j*(spins*2*np.pi+deltac)) * \
        sm+gs*sz*a
    ret[len(spins)+1:] = -gpar*(1+sz)-2*gs * \
        (sm*np.conj(a)+np.conj(sm)*a)
    return ret


def obes(t, Y, *p):
    eta, fielddist, gperp, spins, gpar, ret = p
    # optical bloch equations
    sm = Y[:len(spins)] 
    sz = Y[len(spins):]
    ret[:len(spins)] = -(gperp+gpar/2 + 1j*spins*2*np.pi) * sm+0.5*(1j*eta)*fielddist*sz
    ret[len(spins):] = -gpar*(1+sz)+1j*eta*fielddist*(sm-np.conj(sm))
    return ret


def obes_soc(t, Y, *p):
    ilist, qlist, tlist, fielddist, gperp, spins, gpar, ret = p
    idx = (np.abs(tlist-t)).argmin()
    sm = Y[:len(spins)] 
    sz = Y[len(spins):]
    ret[:len(spins)] = -(gperp+gpar/2 + 1j*spins*2*np.pi)*sm+0.5 * fielddist*(1j*ilist[idx]-qlist[idx])*sz
    ret[len(spins):] = -gpar*(1+sz)+1j*ilist[idx]*fielddist*(sm - np.conj(sm))+fielddist*qlist[idx]*(sm+np.conj(sm))
    return ret


def mbes_floquet(t, Y, *p):
    delta_t, tlist, eta, kappa, deltac, gs, gperp, spins, gpar, ret = p
    idx = (np.abs(tlist-t)).argmin()
    delt_t = delta_t[idx]
    sm = Y[1:len(spins)+1] 
    sz = Y[len(spins)+1:]
    a = Y[0]
    ret[0] = -kappa*a-1j*deltac*a+gs.dot(sm)+eta
    ret[1:len(spins)+1] = -(gperp+gpar/2 + 1j*(spins*2*np.pi+deltac+delt_t)
                  )*sm+gs*sz*a
    ret[len(spins)+1:] = -gpar*(1+sz)-2*gs * \
        (sm*np.conj(a)+np.conj(sm)*a)
    return ret


def mbes_floquet_hp(t, Y, *p):
    delta_t, tlist, eta, kappa, deltac, gs, gperp, spins, gpar, ret = p
    # the same as above, but with Hollstein-Primakoff approximation applied
    idx = (np.abs(tlist-t)).argmin()
    a = Y[0]
    sm = Y[1:]
    delt_t = delta_t[idx]
    ret[0] = -kappa*a-1j*deltac*a+gs.dot(sm)+eta
    ret[1:] = -(gperp+gpar/2 + 1j*(spins*2*np.pi+deltac+delt_t))*sm-gs*a
    return ret


def mbes_floquet_func(t, Y, *p):
    func, func_args, eta, kappa, deltac, gs, gperp, spins, gpar, ret = p
    delta_t = func(t, *func_args)
    sm = Y[1:len(spins)+1] 
    sz = Y[len(spins)+1:]
    a = Y[0]
    ret[0] = -kappa*a-1j*deltac*a+gs.dot(sm)+eta
    ret[1:len(spins)+1] = -(gperp+gpar/2 + 1j*(spins*2*np.pi +
                  deltac+delta_t))*sm+gs*sz*a
    ret[len(spins)+1:] = -gpar*(1+sz)-2*gs * \
        (sm*np.conj(a)+np.conj(sm)*a)
    return ret


def mbes_masing(t, Y, *p):
    pump, kappa, deltac, gs, gperp, spins, gpar, ret = p
    sm = Y[1:len(spins)+1] 
    sz = Y[len(spins)+1:]
    a = Y[0]
    ret[0] = -(kappa+1j*deltac)*a+gs.dot(sm)
    ret[1:len(spins)+1] = -(gperp+gpar/2 + pump/2 + 1j*(spins*2*np.pi+deltac))*sm+gs*sz*a
    ret[len(spins)+1:] = -gpar*(1+sz)+pump*(1-sz)-2*gs*(sm*np.conj(a)+np.conj(sm)*a)
    return ret


def mbes_masing_temp(t, Y, *p):
    nbar, pump, kappa, deltac, gs, gperp, spins, gpar, ret = p
    sm = Y[1:len(spins)+1] 
    sz = Y[len(spins)+1:]
    a = Y[0]
    ret[0] = -(kappa+1j*deltac)*a+gs.dot(sm)
    ret[1:len(spins)+1] = -(gperp+gpar*(0.5+nbar) + pump/2 +
                  1j*(spins*2*np.pi+deltac))*sm+gs*sz*a
    ret[len(spins)+1:] = -gpar*(1+(2*nbar+1)*sz)+pump*(1-sz)-2 * \
        gs*(sm*np.conj(a)+np.conj(sm)*a)
    return ret
    



def get_jacobian(mbes_func):
    jacobian_dict = {mbes_2ndorder: jacobian_mbes_2ndorder_real}  # {mbes: jacobian_mbes, mbes_soc: jacobian_mbes, mbes_soc_grape: jacobian_mbes, mbes_hp: jacobian_mbes_hp, mbes_soc_no_cavity:jacobian_mbes_no_cavity, mbes_masing: jacobian_mbes_masing}
    #jacobian_dict = {}
    try:
        return jacobian_dict[mbes_func]
    except KeyError:
        return None

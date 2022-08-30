#!/usr/bin/env python
# coding: utf-8

import numpy as np

from _helper_functions import *
from _mbes import *
from numba import njit


def solve_mbes(mbes_func, init, pdf, tlistpump, tlistdecay, *args):
    #mbes_func: MBE functions to solve (defined in _mbes)
    #init: inital state
    #pdf: pdf for spins, necessary for normalizing the solutions
    #tlistpump: timelist for which the pump tone is turned on
    #tlistdecay: timelist for which the decay is computed (keep None if you don't want any decay)
    #*args: function arguments to pass to the MBE functions
    ret = np.empty(len(init), dtype=np.complex128)
    solpump, infodict = odeintz(mbes_func, init, tlistpump, args=(*args, ret), jac=get_jacobian(mbes_func))
    solp_ret = prepare_solution(solpump, init, pdf, tlistpump)
    if tlistdecay is not None:
        initdecay = solpump[-1,:].copy()
        #make sure that the last six arguments correspond to the one defined in the mbes function in _mbes
        soldecay = odeintz(mbes, initdecay, tlistdecay, args=(0, *args[-6:], ret), jac=get_jacobian(mbes), full_output=False)
        sold_ret = prepare_solution(soldecay, initdecay, pdf, tlistdecay)
        return solp_ret, sold_ret, infodict
    else:
        return solp_ret, infodict


def test_jac_2ndorder(jac_func, init, pdf, tlistpump, tlistdecay, gs, *args):
    J = np.zeros((len(init.view(np.float64)), len(init.view(np.float64))))
    args=(*args, gs, *create_sorting_indices(gs), J)
    newf = njit(jac_func)
    return newf(init.view(np.float64), 0, *args), newf, args


def solve_mbes_2ndorder(mbes_func, init, pdf, tlistpump, tlistdecay, gs, *args):
    #mbes_func: MBE functions to solve (defined in _mbes)
    #init: inital state
    #pdf: pdf for spins, necessary for normalizing the solutions
    #tlistpump: timelist for which the pump tone is turned on
    #tlistdecay: timelist for which the decay is computed (keep None if you don't want any decay)
    #gs: coupling list
    #spins: detuning spins from the cavity
    #*args: additional function arguments to pass to the MBE functions
    ret = np.empty(len(init), dtype=np.complex128)
    
    jacf = get_jacobian(mbes_func)
    if jacf:
        jacf = njit(jacf)
    
    sortidx = create_sorting_indices(gs)
    solpump, infodict = odeintz(mbes_func, init, tlistpump, args=(*args, gs, *sortidx, ret), jac=jacf)
    solp_ret = prepare_solution_2ndorder(solpump, init, pdf, tlistpump)
    if tlistdecay is not None:
        initdecay = solpump[-1,:].copy()
        #make sure that the last six arguments correspond to the one defined in the mbes function in _mbes
        soldecay, infodict2 = odeintz(mbes_2ndorder, initdecay, tlistdecay, args=(0.,  *args[-5:], gs, *sortidx, ret), jac=jacf, full_output=True)
        sold_ret = prepare_solution_2ndorder(soldecay, initdecay, pdf, tlistdecay)
        return solp_ret, sold_ret, infodict
    else:
        return solp_ret, infodict   


def setup_mbes(mbes_func, init, pdf, tlistpump, tlistdecay, *args):
    #mbes_func: MBE functions to solve (defined in _mbes)
    #init: inital state
    #pdf: pdf for spins, necessary for normalizing the solutions
    #tlistpump: timelist for which the pump tone is turned on
    #tlistdecay: timelist for which the decay is computed (keep None if you don't want any decay)
    #*args: function arguments to pass to the MBE functions
    ret = np.empty(len(init), dtype=np.complex128)
    jac = get_jacobian(mbes_func)
    jac = None
    problem_julia_func(mbes_func, np.asarray(init, dtype=np.complex128), tlistpump, tlistdecay, args=(*args, ret), jac=jac)

def setup_mbes_2ndorder(mbes_func, init, pdf, tlistpump, tlistdecay, gs, *args):
    #mbes_func: MBE functions to solve (defined in _mbes)
    #init: inital state
    #pdf: pdf for spins, necessary for normalizing the solutions
    #tlistpump: timelist for which the pump tone is turned on
    #tlistdecay: timelist for which the decay is computed (keep None if you don't want any decay)
    #gs: coupling list
    #spins: detuning spins from the cavity
    #*args: additional function arguments to pass to the MBE functions
    ret = np.empty(len(init), dtype=np.complex128)
    jac = get_jacobian(mbes_func)
    jac = None
    if not jac:
        sorts = create_sorting_indices_small(gs)
    else:
        sorts = create_sorting_indices(gs)
    problem_julia_func(mbes_func, init, tlistpump, tlistdecay, args=(*args, gs, *sorts, ret), jac=jac)

def test_mbes_2ndorder2(mbes_func, init, pdf, tlistpump, tlistdecay, gs, *args):
    #mbes_func: MBE functions to solve (defined in _mbes)
    #init: inital state
    #pdf: pdf for spins, necessary for normalizing the solutions
    #tlistpump: timelist for which the pump tone is turned on
    #tlistdecay: timelist for which the decay is computed (keep None if you don't want any decay)
    #gs: coupling list
    #spins: detuning spins from the cavity
    #*args: additional function arguments to pass to the MBE functions
    ret = np.empty(len(init), dtype=np.complex128)
    return problem_julia_func(mbes_func, init, tlistpump, tlistdecay, args=(*args, gs, *create_sorting_indices(gs), ret), jac=None)

def test_mbes_2ndorder(mbes_func, init, pdf, tlistpump, tlistdecay, gs, *args):
    #mbes_func: MBE functions to solve (defined in _mbes)
    #init: inital state
    #pdf: pdf for spins, necessary for normalizing the solutions
    #tlistpump: timelist for which the pump tone is turned on
    #tlistdecay: timelist for which the decay is computed (keep None if you don't want any decay)
    #gs: coupling list
    #spins: detuning spins from the cavity
    #*args: additional function arguments to pass to the MBE functions
    ret = np.empty(len(init), dtype=np.complex128)
    return mbes_func(0, init.view(np.complex128), *(*args, gs, *create_sorting_indices(gs), ret))
           
def test_mbes_2ndorder_julia(mbes_func, init, pdf, tlistpump, tlistdecay, gs, *args):
    ret = np.empty(len(init), dtype=np.complex128).view(np.float64)
    func = test_julia(mbes_func, init, tlistpump, tlistdecay, args=(*args, gs, *create_sorting_indices(gs), ret), jac=None)
    
    Main.ret = ret
    Main.init = init.view(np.float64)
    Main.pars = (np.array([args[0]]), )
    
    res = Main.eval("f(ret, init, pars, 0)")
    return res
    
def solve_mbes_general(mbes_func, init, pdf, tlist, *args):
    #most general version of solving mbes. Use for example if you have time dependent pump and detuning, where you can define the driving amplitude and the detuning of the spins accordingly
    ret = np.empty(len(init), dtype=np.complex128)
    solution, infodict = odeintz(mbes_func, init, tlist, args=(*args,ret), jac=get_jacobian(mbes_func))
    solution_ret = prepare_solution(solution, init, pdf, tlist)
    return solution_ret, infodict


def solve_mbes_only_decay(init, pdf, tlistdecay, *args):
    #init: inital state
    #pdf: pdf for spins, necessary for normalizing the solutions
    #tlistdecay: timelist for which the decay is computed
    #*args: function arguments to pass to the MBE functions 
    ret = np.empty(len(init), dtype=np.complex128)
    soldecay, infodict = odeintz(mbes, init, tlistdecay, args=(0, *args, ret), jac=get_jacobian(mbes))
    sold_ret = prepare_solution(soldecay, init, pdf, tlistdecay)
    return sold_ret, infodict
    
    
def solve_mbes_without_cavity(mbes_func, init, pdf, tlistpump, tlistdecay, *args):
    #mbes_func: MBE functions to solve (defined in _mbes)
    #init: inital state
    #pdf: pdf for spins, necessary for normalizing the solutions
    #tlistpump: timelist for which the pump tone is turned on
    #tlistdecay: timelist for which the decay is computed (keep None if you don't want any decay)
    #*args: function arguments to pass to the MBE functions 
    ret = np.empty(len(init), dtype=np.complex128)
    solpump, infodict = odeintz(mbes_func, init, tlistpump, args=(*args,ret), jac=get_jacobian(mbes_func))
    solp_ret = prepare_solution_no_cav(solpump, init, pdf, tlistpump)

    if tlistdecay is not None:
        initdecay = solpump[-1,:].copy()
        #make sure that the last six arguments correspond to the one defined in the mbes function in _mbes
        soldecay = odeintz(mbes_no_cavity, initdecay, tlistdecay, args=(*args[-5:], ret), jac=get_jacobian(mbes_no_cavity))
        sold_ret = prepare_solution_no_cav(soldecay, initdecay, pdf, tlistdecay)
        return solp_ret, sold_ret, infodict
    else:
        return solp_ret, infodict
    

def solve_mbes_wait(mbes_func, init, pdf, tlistinit, tlistprobe, tlistdecay, *args):
    #mbes_func: MBE functions to solve (defined in _mbes)
    #init: inital state
    #pdf: pdf for spins, necessary for normalizing the solutions
    #tlistpump: timelist for which the pump tone is turned on
    #tlistinit: timelist before the pulse is played
    #tlistdecay: timelist for which the decay is computed (keep None if you don't want any decay)
    #*args: function arguments to pass to the MBE functions 
    ret = np.empty(len(init), dtype=np.complex128)
    initinter = init.copy()
    solinter = odeintz(mbes, initinter, tlistinit, args=(0, *args[-6:], ret), jac=get_jacobian(mbes))
    soli_ret = prepare_solution(solinter, initinter, pdf, tlistinit)
    
    initprobe = solinter[-1,:].copy()
    solprobe, infodict = odeintz(mbes_func, initprobe, tlistprobe, args=(*args,ret), jac=get_jacobian(mbes_func))
    solpr_ret = prepare_solution(solprobe, initprobe, pdf, tlistprobe)
        
    if tlistdecay is not None:
        initdecay = solprobe[-1,:].copy()
        #make sure that the last six arguments correspond to the one defined in the mbes function in _mbes
        soldecay = odeintz(mbes, initdecay, tlistdecay, args=(0, *args[-6:], ret), jac=get_jacobian(mbes))
        sold_ret = prepare_solution(soldecay, initdecay, pdf, tlistdecay)
        return soli_ret, solpr_ret, sold_ret, infodict
    else:
        return soli_ret, solpr_ret, infodict
    
def solve_mbes_cpmg(mbes_func, init, pdf, tlistpump, tlistinter, tlistdecay, *args):
    ret = np.empty(len(init), dtype=np.complex128)
    solpump, infodict = odeintz(mbes_func, init, tlistpump, args=(*args, ret), jac=get_jacobian(mbes_func))
    solp_ret = prepare_solution(solpump, init, pdf, tlistpump)
        
    initinter = solpump[-1,:].copy()
    solinter = odeintz(mbes, initinter, tlistinter, args=(0, *args[-6:], ret), jac=get_jacobian(mbes))
    soli_ret = prepare_solution(solinter, initinter, pdf, tlistinter)
    
    initpump2 = solinter[-1,:].copy()
    solpump2 = odeintz(mbes, initpump2, tlistpump, args=(-1j*args[0]*2, *args[-6:], ret), jac=get_jacobian(mbes))
    solp2_ret = prepare_solution(solpump2, initpump2, pdf, tlistpump)
    
    initdecay = solpump2[-1,:].copy()
    soldecay = odeintz(mbes, initdecay, tlistdecay, args=(0, *args[-6:], ret), jac=get_jacobian(mbes))
    sold_ret = prepare_solution(soldecay, initdecay, pdf, tlistdecay)
    
    return solp_ret, soli_ret, solp2_ret, sold_ret, infodict

def solve_mbes_cpmg_time(mbes_func, init, pdf, tlistpump, tlistinter, tlistdecay, *args):
    ret = np.empty(len(init), dtype=np.complex128)
    solpump, infodict = odeintz(mbes_func, init, tlistpump, args=(*args, ret), jac=get_jacobian(mbes_func))
    solp_ret = prepare_solution(solpump, init, pdf, tlistpump)
        
    initinter = solpump[-1,:].copy()
    solinter = odeintz(mbes, initinter, tlistinter, args=(0, *args[-6:], ret), jac=get_jacobian(mbes))
    soli_ret = prepare_solution(solinter, initinter, pdf, tlistinter)
    
    initpump2 = solinter[-1,:].copy()
    dt = tlistpump[1]-tlistpump[0]
    tlistpump2 = np.arange(0, 2*tlistpump[-1]+dt, dt)
    solpump2 = odeintz(mbes, initpump2, tlistpump2, args=(-1j*args[0], *args[-6:], ret), jac=get_jacobian(mbes))
    solp2_ret = prepare_solution(solpump2, initpump2, pdf, tlistpump2)
    
    initdecay = solpump2[-1,:].copy()
    soldecay = odeintz(mbes, initdecay, tlistdecay, args=(0, *args[-6:], ret), jac=get_jacobian(mbes))
    sold_ret = prepare_solution(soldecay, initdecay, pdf, tlistdecay)
        
    return solp_ret, soli_ret, solp2_ret, sold_ret, infodict

def solve_hahn_cpmg(mbes_func, init, pdf, tlistpump, tlistinter, tlistdecay, *args):
    ret = np.empty(len(init), dtype=np.complex128)
    solpump, infodict = odeintz(mbes_func, init, tlistpump, args=(*args,ret ), jac=get_jacobian(mbes_func))
    solp_ret = prepare_solution(solpump, init, pdf, tlistpump)
        
    initinter = solpump[-1,:].copy()
    solinter = odeintz(mbes, initinter, tlistinter, args=(0, *args[-6:], ret), jac=get_jacobian(mbes))
    soli_ret = prepare_solution(solinter, initinter, pdf, tlistinter)
    
    initpump2 = solinter[-1,:].copy()
    solpump2 = odeintz(mbes, initpump2, tlistpump, args=(args[0]*2, *args[-6:], ret), jac=get_jacobian(mbes))
    solp2_ret = prepare_solution(solpump2, initpump2, pdf, tlistpump)
    
    initdecay = solpump2[-1,:].copy()
    soldecay = odeintz(mbes, initdecay, tlistdecay, args=(0, *args[-6:], ret), jac=get_jacobian(mbes))
    sold_ret = prepare_solution(soldecay, initdecay, pdf, tlistdecay)
        
    return solp_ret, soli_ret, solp2_ret, sold_ret, infodict

def solve_mbes_hahn_time(mbes_func, init, pdf, tlistpump, tlistinter, tlistdecay, *args):
    ret = np.empty(len(init), dtype=np.complex128)
    solpump, infodict = odeintz(mbes_func, init, tlistpump, args=(*args, ret), jac=get_jacobian(mbes_func))
    solp_ret = prepare_solution(solpump, init, pdf, tlistpump)
        
    initinter = solpump[-1,:].copy()
    solinter = odeintz(mbes, initinter, tlistinter, args=(0, *args[-6:], ret), jac=get_jacobian(mbes))
    soli_ret = prepare_solution(solinter, initinter, pdf, tlistinter)
    
    initpump2 = solinter[-1,:].copy()
    dt = tlistpump[1]-tlistpump[0]
    tlistpump2 = np.arange(0, 2*tlistpump[-1]+dt, dt)
    solpump2 = odeintz(mbes, initpump2, tlistpump2, args=(args[0], *args[-6:], ret), jac=get_jacobian(mbes))
    solp2_ret = prepare_solution(solpump2, initpump2, pdf, tlistpump2)
    
    initdecay = solpump2[-1,:].copy()
    soldecay = odeintz(mbes, initdecay, tlistdecay, args=(0, *args[-6:], ret), jac=get_jacobian(mbes))
    sold_ret = prepare_solution(soldecay, initdecay, pdf, tlistdecay)
        
    return solp_ret, soli_ret, solp2_ret, sold_ret, infodict


def solve_mbes_pump_probe(mbes_func, init, pdf, tlistpump, tlistinter, tlistprobe, tlistdecay, ampl_probe, *args):
    #mbes_func: MBE functions to solve (defined in _mbes)
    #init: inital state
    #pdf: pdf for spins, necessary for normalizing the solutions
    #tlistpump: timelist for which the pump tone is turned on
    #tlistinter: timelist between the two pulses
    #tlistprobe: timelist for which the probe pulse is turned on
    #tlistdecay: timelist for which the decay is computed (keep None if you don't want any decay)
    #*args: function arguments to pass to the MBE functions 
    ret = np.empty(len(init), dtype=np.complex128)
    solpump, infodict = odeintz(mbes_func, init, tlistpump, args=(*args, ret), jac=get_jacobian(mbes_func))
    solp_ret = prepare_solution(solpump, init, pdf, tlistpump)
        
    initinter = solpump[-1,:].copy()
    solinter = odeintz(mbes, initinter, tlistinter, args=(0, *args[-6:], ret), jac=get_jacobian(mbes))
    soli_ret = prepare_solution(solinter, initinter, pdf, tlistinter)
    
    initprobe = solinter[-1,:].copy()
    solprobe = odeintz(mbes, initprobe, tlistprobe, args=(ampl_probe, *args[-6:], ret), jac=get_jacobian(mbes))
    solpr_ret = prepare_solution(solprobe, initprobe, pdf, tlistprobe)
        
    if tlistdecay is not None:
        initdecay = solprobe[-1,:].copy()
        #make sure that the last six arguments correspond to the one defined in the mbes function in _mbes
        soldecay = odeintz(mbes, initdecay, tlistdecay, args=(0, *args[-6:], ret), jac=get_jacobian(mbes))
        sold_ret = prepare_solution(soldecay, initdecay, pdf, tlistdecay)
        return solp_ret, soli_ret, solpr_ret, sold_ret, infodict
    else:
        return solp_ret, soli_ret, solpr_ret, infodict


def get_pulses_from_cavity_field(a, af, sm, tlist, tlistf, kappa, deltac, gs):
    dtf = tlistf[1]-tlistf[0]
    dt = tlist[1]-tlist[0]
    dadtf = np.gradient(af, dtf)
    dadt = dadtf[::int(dt/dtf)]
    sumspins = np.array([sm[i,:]*gs for i in range(len(sm))])
    imiq = dadt+kappa*a+1j*deltac*a-np.sum(sumspins, axis=1)
    ipulse = imiq.real
    qpulse = imiq.imag
    return ipulse, qpulse


def solve_obes(obes_func, init, pdf, tlistpump, tlistdecay, *args):
    #mbes_func: MBE functions to solve (defined in _mbes)
    #init: inital state
    #pdf: pdf for spins, necessary for normalizing the solutions
    #tlistpump: timelist for which the pump tone is turned on
    #tlistdecay: timelist for which the decay is computed (keep None if you don't want any decay)
    #*args: function arguments to pass to the MBE functions 
    ret = np.empty(len(init), dtype=np.complex128)
    solpump, infodict = odeintz(obes_func, init, tlistpump, args=(*args,ret ), jac=get_jacobian(obes_func)) 
    solp_ret = prepare_solution_no_cav(solpump, init, pdf, tlistpump)
    if tlistdecay is not None:
        initdecay = solpump[-1,:].copy()
        #make sure that the last six arguments correspond to the one defined in the mbes function in _mbes
        soldecay = odeintz(obes, initdecay, tlistdecay, args=(0, *args[-4:], ret), jac=get_jacobian(obes))
        sold_ret = prepare_solution_no_cav(soldecay, initdecay, pdf, tlistdecay)
        return solp_ret, sold_ret, infodict
    else:
        return solp_ret, infodict
    
    
def solve_mbes_floquet(mbes_func, init, pdf, tlistpump, tlistdecay, *args):
    #init: inital state
    #pdf: pdf for spins, necessary for normalizing the solutions
    #tlistpump: timelist for which the pump tone is turned on
    #*args: function arguments to pass to the MBE functions 
    #reshape idx is for if we have a a g_list
    ret = np.empty(len(init), dtype=np.complex128)
    solpump, infodict = odeintz(mbes_func, init, tlistpump, args=(*args, ret), jac=get_jacobian(mbes_func))
    solp_ret = prepare_solution(solpump, init, pdf, tlistpump)
    if tlistdecay is not None:
        initdecay = solpump[-1,:].copy()
        #make sure that the last six arguments correspond to the one defined in the mbes function in _mbes
        soldecay = odeintz(mbes, initdecay, tlistdecay, args=(0, *args[-6:], ret), jac=get_jacobian(mbes))
        sold_ret = prepare_solution(soldecay, initdecay, pdf, tlistdecay)
        return solp_ret, sold_ret, infodict
    else:
        return solp_ret, infodict
    return solp_ret, infodict




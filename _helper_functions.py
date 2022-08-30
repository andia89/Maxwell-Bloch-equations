#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.special import gamma
from scipy import integrate
from scipy.optimize import fsolve
import scipy.io
import math
import nbkode
import weakref
import gc
import re
import fileinput
from numba import jit, njit
import numba
import inspect
from jitcode import y, jitcode

from julia.api import Julia
jl = Julia(sysimage="/home/andreas/Physics/julia-1.8.0-rc1/pythonsys/sys.so", compiled_modules=False, debug=False)
from julia import Main
from diffeqpy import de


def tsallis(f, q, width):
    delta = (width)/2*np.sqrt(2*q-2)/np.sqrt(2**q-2)/np.sqrt(2)
    norm = np.sqrt(q-1)*gamma(1/(q-1))/(np.sqrt(2*np.pi)*delta*gamma((3-q)/(2*(q-1))))
    val = (1+(q-1)*f**2/delta**2/2)**(1/(1-q))
    return norm*val

def cauchy(x, pos, fwhm, ampl):
    return 1/((x-pos)**2+0.25*fwhm**2)*0.25*fwhm**2*ampl

def get_thermal_photons(fr, temp):
    #fr in MHz frequency 
    #temp in K
    hbokb = 7.63823511e-6  # for MHz of resonance frequency
    if temp == 0:
        return 0
    else:
        return np.exp(-hbokb*2*np.pi*fr/temp)/(1-np.exp(-hbokb*2*np.pi*fr/temp))

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx



def find_nearest_sorted(array,value):
    """much faster for sorted arrays"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


    
def find_nearest_sorted_left(array,value):
    """much faster for sorted arrays"""
    idx = np.searchsorted(array, value, side="left")
    if idx < len(array):
        diff = math.fabs(value - array[idx])
    else:
        diff = None
    if idx == 0 or diff == 0:
        return idx
    else:
        return idx - 1


    
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def smooth_data(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def fsolvez(func, z0, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)
    def realfunc(x, *args):
        z = x.view(np.complex128)
        dzdt = func(z, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)
    result, infodict, ier, message = fsolve(realfunc, z0.view(np.float64), **kwargs, full_output=True)

    z = result.view(np.complex128)
    return z, infodict, ier, message


def odeintz(func, z0, t, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    ## Disallow Jacobian-related arguments.
    #_unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    #bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    #if len(bad_args) > 0:
    #    raise ValueError("The odeint argument %r is not supported by "
    #                     "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    full_output = True
    if "full_output" in kwargs:
        full_output = kwargs.pop("full_output")

    if not "method" in kwargs or kwargs.get("method") in ("RK45", "RK23"):
        kwargs.pop("jac")
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    #def realfunc(t, x, *args):
    #    z = x.view(np.complex128)
    #    dzdt = func(t, z, *args)
    #    # func might return a python list, so convert its return
    #    # value to an array with type np.complex128, and then return
    #    # a np.float64 view of that array.
    #    return np.asarray(dzdt, dtype=np.complex128).view(np.float64)
    result = integrate.solve_ivp(njit(func), (t[0],t[-1]), z0, t_eval=t, **kwargs)
    z = np.asarray(result.y).transpose()
    #z = res[:, ::2]+1j*res[:,1::2]
    shape = z.shape
    if shape[0] < len(t):
        z = np.vstack((z, np.zeros((len(t)-shape[0], shape[1]))))
    if full_output:
        return z, result
    return z

def odeintz_julia(func, z0, t, **kwargs):
    func2 = lambda u,p,t: func(t, u, *p)
    prob = de.ODEProblem(func2, np.asarray(z0, dtype=np.complex128), (t[0], t[-1]), kwargs['args'])
    sol = de.solve(prob)
    z = np.asarray(sol(t)).transpose()
    return z, sol


def remake_julia_problem(params=None, init=None, tspan=None):

    if params is not None:
        Main.eval("problem.p[1] = %f"%params[0])
    if init is not None:
        Main.init = init
        Main.eval("problem.z0 = init")
        Main.init = None
    if init is not None:
        Main.tspan = tspan
        Main.eval("problem.z0 = tspan")
        Main.tspan = None

def format_julia_sol_2ndorder(sol, t, tlistdecay, init, pdf):
    sold = np.asarray(sol)
    k1 = np.ascontiguousarray(sold[:len(t), :]).view(np.complex128)
    los = prepare_solution_2ndorder(k1, init, pdf, t)
    if tlistdecay is None:
        return los
    else:
        k1 = np.ascontiguousarray(sold[-len(tlistdecay):, :]).view(np.complex128)
        losd = prepare_solution_2ndorder(k1, init, pdf, tlistdecay)
        return los, losd

    
def clean_julia():
    Main.eval("ccall(:malloc_trim, Cvoid, (Cint,), 0)")
    Main.eval("ccall(:malloc_trim, Int32, (Int32,), 0)")
    Main.GC.gc()
    gc.collect()    
    
def solve_julia_problem(tpump=None, tlistdecay=None):
    df = tpump[-1]-tpump[-2]
    solver = de.RK4()
    #solver = de.KenCarp47(linsolve=de.KrylovJL_GMRES())
    Main.solver = solver
    if tlistdecay is None or tpump is None:
                Main.eval("""
            solve(problem, solver,saveat=%f)
        """%(df))
    else:
        s = "using DifferentialEquations; cbfunc = PresetTimeCallback(%f,affect!);"%(tpump[-1])
        Main.eval("""
        function affect!(integrator)
            nothing
            integrator.p[1] = 0.;
        end
        %s
        """%s)
        Main.eval("""
            solution = DifferentialEquations.solve(problem, solver, callback=cbfunc, saveat=%f)
            sol = solution.u
        """%(df))
        solution = Main.sol
    sol = np.asarray(solution)
    Main.solution = None
    Main.sol = None
    clean_julia()
    return sol


def problem_julia(func, z0, t, tlistdecay, **kwargs):
    #make sure that the first argument is the one that is changing over time
    args = kwargs["args"][:-1]
    jac = kwargs["jac"]
    jac = None
    argsdt = (numba.typeof(args[0]),)
    lines = inspect.getsource(func).strip().split("\n")
    argnames = lines[1].strip().split(",")
    argnames1 = [i.strip() for i in argnames][:len(args)]
    argnames0 = argnames1.pop(0)

    new_args = ["    %s = p_0"%argnames[0]]
    paramstr = "p_0, "


    newlines = """@numba.cfunc(c_sig)\ndef f(n_states, du_, u_, %st):\n    u = numba.carray(u_, n_states)\n    du = numba.carray(du_, n_states)\n"""%(paramstr)
    newlines = newlines.split("\n")
    
    if jac:
        newlinesjac = """@numba.cfunc(c_sig_jac)\ndef fjac(n_states, J_, u_, %st):\n    u = numba.carray(u_, n_states)\n    J = numba.carray(J_, (n_states, n_states))\n"""%(paramstr)
        newlinesjac = newlinesjac.split("\n")
        linesjac = inspect.getsource(jac).strip().split("\n")
        changed_lines = []
        for line in linesjac[2:-1]:
            line2 = line.replace("ret", "du")
            line2 = line2.replace("Y", "u")
            changed_lines.append(line2)
        newfuncjac = newlinesjac+new_args+changed_lines
        
    
    changed_lines = []
    for line in lines[2:-1]:
        line2 = line.replace("ret", "du")
        line2 = line2.replace("Y", "u")
        changed_lines.append(line2)
    newfunc = newlines+new_args+changed_lines

    
    def wrapper(args):
        context = {}
        for i, argname in enumerate(argnames1):
            context[argname]=args[i+1]
        c_sig = numba.types.void(
                numba.types.int64,
                numba.types.CPointer(numba.types.complex128),
                numba.types.CPointer(numba.types.complex128),
                *argsdt,
                numba.types.float64
        )
        context["numba"] = numba
        context["np"] = np
        context["c_sig"] = c_sig

        if jac:
            c_sig_jac = numba.types.void(
                numba.types.int64,
                numba.types.CPointer(numba.types.float64),
                numba.types.CPointer(numba.types.float64),
                *argsdt,
                numba.types.float64
            )
            context["c_sig_jac"] = c_sig_jac
            exec("\n".join(newfuncjac), context, context)
            fjac = context["fjac"]
            
        exec("\n".join(newfunc), context, context)
        f = context["f"]
        context.clear()
        
        dtype_dict = {"int64":"Int64", "float64":"Float64", "complex128":"ComplexF64"}
        funcstr = """
        
        using PyCall

        function wrap_python_callback(callback)
            py_func_ptr = callback.address
            func_ptr = convert(Csize_t, py_func_ptr)
            func_ptr = convert(Ptr{Nothing}, func_ptr)

            function f(du,u,p,t)
                callback
                ucomp = reinterpret(ComplexF64, u)
                n_states = length(ucomp)
                ccall(
                      convert(Ptr{Nothing}, func_ptr),
                      Nothing,
                      (Int64, 
                      Ptr{ComplexF64}, 
                      Ptr{ComplexF64}, 
                      Float64,
                      Float64),
                      n_states,
                      du,
                      ucomp,
                      p[1],
                      t
                    )
                return reinterpret(Float64, du)
            end
        end

        """
        if jac:
            funcstr_jac = """

            using PyCall

            function wrap_python_callback_jac(callback)
                py_func_ptr = callback.address
                func_ptr = convert(Csize_t, py_func_ptr)
                func_ptr = convert(Ptr{Nothing}, func_ptr)

                function f_jac(J,u,p,t)
                    callback
                    n_states = length(ucomp)   
                     ccall(
                          convert(Ptr{Nothing}, func_ptr),
                          Nothing,
                          (Int64, 
                          Ptr{Float64}, 
                          Ptr{Float64}, 
                          Float64,
                          Float64),
                          n_states,
                          du,
                          ucomp,
                          p[1],
                          t
                        )
                end
            end

            """
            jac_f =  Main.eval(funcstr_jac)
            out_jac = jac_f(fjac)

        jul_f = Main.eval(funcstr)
    
        out = jul_f(f)
        Main.GC.gc()
        gc.collect()
        if jac:
            return out, out_jac
        else:
            return out
    
    if jac:
        func, jacfunc = wrapper(args)
        odefunc = de.ODEFunction(func, jac=jacfunc);
    else:
        odefunc = de.ODEFunction(wrapper(args));
    if tlistdecay is None:
        prob = de.ODEProblem(odefunc, z0.view(np.float64), (t[0], t[-1]), (args[0], ))
    else:
        prob = de.ODEProblem(odefunc, z0.view(np.float64), (t[0], 2*t[-1]+tlistdecay[-1]-t[-2]), (args[0], ))
    return prob



def problem_julia_func(func, z0, t, tlistdecay, **kwargs):
    #make sure that the first argument is the one that is changing over time
    args = kwargs["args"][:-1]
    jac = kwargs["jac"]
    Main.eval("using LinearAlgebra")
    Main.eval('include("_mbes.jl")')
    funcname = func.__name__
    funcjulia = Main.eval(funcname)
    

    argsn = []
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            if str(arg.dtype).startswith("int"):
                argsn.append(arg+1)
            else:
                argsn.append(arg)
        else:
            argsn.append(arg)
    #argsn = tuple(argsn)

    Main.eval("""function f(du::Vector{Float64},u::Vector{Float64},p,t::Float64)
        ucomp = reinterpret(ComplexF64, u)
        ducomp = reinterpret(ComplexF64, du)
        %s(ducomp, ucomp, p, t)
        return reinterpret(Float64, ducomp)
        end
    """%funcname)
    
    if jac:
        Main.eval('include("_jacobian.jl")')
        funcname_jac = jac.__name__
        jacf = Main.eval(funcname_jac)
        odefunc = de.ODEFunction(Main.f, jac= jacf)
    else:
        odefunc = de.ODEFunction(Main.f)
    problem = de.ODEProblem(odefunc, z0.view(np.float64), (t[0], 2*t[-1]+tlistdecay[-1]-t[-2]), argsn)
    Main.problem = problem
    del problem
    clean_julia()


def create_sorting_indices(gs):
    gs = np.real(gs)
    nspins = len(gs)
    gmat = np.zeros((nspins, nspins*(nspins-1)), dtype=np.complex128)
    gmatjacre = np.zeros((nspins*(nspins-1), nspins*(nspins-1)), dtype=np.float64)
    gmatjacim = np.zeros(gmatjacre.shape, dtype=np.float64)
    gmatjacre2 = np.zeros((nspins*(nspins-1), nspins*(nspins-1)), dtype=np.float64)
    gmatjacim2 = np.zeros(gmatjacre2.shape, dtype=np.float64)
    for i in range(nspins):
        np.fill_diagonal(gmatjacre[i*(nspins-1):(i+1)*(nspins-1), i*(nspins-1):(i+1)*(nspins-1)] ,np.real(gs[i]))
    for i in range(nspins):
        gstemp = gs[np.arange(nspins)!=i]
        np.fill_diagonal(gmatjacre2[i*(nspins-1):(i+1)*(nspins-1), i*(nspins-1):(i+1)*(nspins-1)] ,np.real(gstemp))
    
    gmatidx = np.ones((nspins, nspins*(nspins-1)), dtype=int)*(-1)
    for i in range(len(gmat)):
        gmat[i, i*(nspins-1):(i+1)*(nspins-1)] = gs[np.arange(len(gs)) != i]
        gmatidx[i, i*(nspins-1):(i+1)*(nspins-1)] = np.arange(len(gs))[np.arange(len(gs)) != i]
    
    #we need a bit of gymnastics here because we need to switch indices around in the  
    #equations
    indexj = [[i]*(nspins-1) for i in range(len(gs))]
    indexj = np.asarray(indexj).reshape(nspins*(nspins-1))
    
    indexi = []
    ctr = 0
    oldj = indexj[0]
    for i, j in enumerate(indexj):
        if oldj != j:
            ctr = 0
        if ctr==j:
            ctr += 1
        indexi.append(ctr)
        ctr += 1
        oldj = j
    indexi = np.asarray(indexi)
    
    
    l1 = []
    for i in range(nspins):
        startidx = i-1
        if startidx < 0:
            startidx = nspins-1
        l = [startidx]
        toadd = startidx
        for j in range(1, nspins-1):
            if j==i:
                toadd += (nspins-1)+nspins
            else:
                toadd += (nspins-1)
            l.append(toadd)
        l1.append(l)
    indexswitch = np.asarray(l1).reshape(nspins*(nspins-1))
    
    ctr = 0
    for i, temparr in enumerate(l1):
        ctr2 = 0
        for k, idx in enumerate(temparr):
            if i == ctr2:
                ctr2 += 1
            gmatjacim[ctr, idx] = np.real(gs[ctr2])
            gmatjacim2[ctr, idx] = np.real(gs[i])
            ctr += 1
            ctr2 += 1
    
    delcols = []
    for i in range(nspins-1):
        for j in range(i+1):
            delcols.append((nspins-1)*(i+1)+j)

    gmatsmspre = np.delete(gmat, delcols, axis=1)
    gmatreidx = np.delete(gmatidx, delcols, axis=1)
    
    
    gmatsmspim = np.zeros((gmatsmspre.shape), dtype=gmat.dtype)
    gmatimidx = np.ones((gmatreidx.shape), dtype=gmatreidx.dtype)*(-1)
    colidx0 = 0
    for i in range(nspins-1):
        colidx1 = (i+1)*nspins-(i+1)*(i+2)//2
        np.fill_diagonal(gmatsmspim[i+1:, colidx0:colidx1], gs[i])
        np.fill_diagonal(gmatimidx[i+1:, colidx0:colidx1], i)
        colidx0 = colidx1
    indexj2 = np.delete(indexj, delcols)
    indexi2 = np.delete(indexi, delcols)
    indexred = np.delete(np.arange(nspins*(nspins-1)), delcols)
    nonzeros = np.nonzero(gmatsmspim+gmatsmspre)
    indexdouble = nonzeros[1]
    maskre = np.zeros((indexdouble.shape), dtype=indexdouble.dtype)
    maskim = np.zeros((indexdouble.shape), dtype=indexdouble.dtype)
    gmatjacre = np.delete(gmatjacre, delcols, axis=0)
    gmatjacim = np.delete(gmatjacim, delcols, axis=0)
    gmatjacre2 = np.delete(gmatjacre2, delcols, axis=0)
    gmatjacim2 = np.delete(gmatjacim2, delcols, axis=0)
    for i, val in enumerate(indexdouble):
        if (gmatsmspre[nonzeros[0][i], nonzeros[1][i]] != 0):
            maskre[i] = 1
        if (gmatsmspim[nonzeros[0][i], nonzeros[1][i]] != 0):
            maskim[i] = 1
    indexswitchred = np.delete(indexswitch, delcols)
    
    idxredre = np.where(gmatreidx == -1)
    idxredim = np.where(gmatimidx == -1)
    gstiledre = np.tile(gs, (nspins*(nspins-1)//2,1))
    gstiledre.transpose()[idxredre] = 0
    gstiledim = np.tile(gs, (nspins*(nspins-1)//2,1))
    gstiledim.transpose()[idxredim] = 0
    gstiled = gstiledre+gstiledim
    gstiled4 = np.zeros((nspins*(nspins-1), nspins))
    gstiled2 = np.zeros((nspins*(nspins-1), nspins))
    for i in range(nspins):
        idxs = np.arange(nspins)!=i
        tempar = gstiled2[i*(nspins-1):(i+1)*(nspins-1), idxs]
        tempar2 = gstiled4[i*(nspins-1):(i+1)*(nspins-1), idxs]
        np.fill_diagonal(tempar, gs[i])
        np.fill_diagonal(tempar2, gs[np.arange(nspins)!=i])
        gstiled2[i*(nspins-1):(i+1)*(nspins-1), idxs] = tempar
        gstiled4[i*(nspins-1):(i+1)*(nspins-1), idxs] = tempar2
    gparmat = gstiled.astype(bool).astype(int).astype(np.float64)
    gparmat2 = gstiled2.astype(bool).astype(int).astype(np.float64)
    gstiled3 = np.zeros((nspins*(nspins-1), nspins*(nspins-1)//2))
    ctr = -1
    for i, row in enumerate(gstiled3):
        if i % (nspins-1) == 0:
            ctr += 1
        gstiled3[i, indexdouble[i]] = np.real(gs[ctr])
    
    gstiledswitch = np.zeros(gstiled3.shape)
    tempmask = maskre-maskim
    for i, row in enumerate(gstiledswitch):
        gstiledswitch[i, :] = tempmask[i]*gstiled3[i]
    
    
    return np.ascontiguousarray(gmat), indexi, indexj, indexi2, indexj2, indexswitch, indexred, indexswitchred, np.ascontiguousarray(gmatsmspre), np.ascontiguousarray(gmatsmspim), indexdouble, maskre, maskim, np.ascontiguousarray(gmatreidx), np.ascontiguousarray(gmatimidx),np.ascontiguousarray(gmatidx), np.ascontiguousarray(gmatjacre), np.ascontiguousarray(gmatjacim), np.ascontiguousarray(gmatjacre2), np.ascontiguousarray(gmatjacim2), np.ascontiguousarray(gstiled),np.ascontiguousarray(gstiled2), np.ascontiguousarray(gstiled3), np.ascontiguousarray(gstiled4),np.ascontiguousarray(gstiledswitch), np.ascontiguousarray(gparmat) ,np.ascontiguousarray(gparmat2)



def create_sorting_indices_small(gs):
    gs = np.real(gs)
    nspins = len(gs)
    gmat = np.zeros((nspins, nspins*(nspins-1)), dtype=np.complex128)

    gmatidx = np.ones((nspins, nspins*(nspins-1)), dtype=int)*(-1)
    for i in range(len(gmat)):
        gmat[i, i*(nspins-1):(i+1)*(nspins-1)] = gs[np.arange(len(gs)) != i]
        gmatidx[i, i*(nspins-1):(i+1)*(nspins-1)] = np.arange(len(gs))[np.arange(len(gs)) != i]
    
    #we need a bit of gymnastics here because we need to switch indices around in the  
    #equations
    indexj = [[i]*(nspins-1) for i in range(len(gs))]
    indexj = np.asarray(indexj).reshape(nspins*(nspins-1))
    
    indexi = []
    ctr = 0
    oldj = indexj[0]
    for i, j in enumerate(indexj):
        if oldj != j:
            ctr = 0
        if ctr==j:
            ctr += 1
        indexi.append(ctr)
        ctr += 1
        oldj = j
    indexi = np.asarray(indexi)
    
    
    l1 = []
    for i in range(nspins):
        startidx = i-1
        if startidx < 0:
            startidx = nspins-1
        l = [startidx]
        toadd = startidx
        for j in range(1, nspins-1):
            if j==i:
                toadd += (nspins-1)+nspins
            else:
                toadd += (nspins-1)
            l.append(toadd)
        l1.append(l)
    indexswitch = np.asarray(l1).reshape(nspins*(nspins-1))
    
    
    delcols = []
    for i in range(nspins-1):
        for j in range(i+1):
            delcols.append((nspins-1)*(i+1)+j)
    gmatsmspre = np.delete(gmat, delcols, axis=1)

    gmatsmspim = np.zeros((gmatsmspre.shape), dtype=gmat.dtype)
    colidx0 = 0
    for i in range(nspins-1):
        colidx1 = (i+1)*nspins-(i+1)*(i+2)//2
        np.fill_diagonal(gmatsmspim[i+1:, colidx0:colidx1], gs[i])
        colidx0 = colidx1
    indexj2 = np.delete(indexj, delcols)
    indexi2 = np.delete(indexi, delcols)
    indexred = np.delete(np.arange(nspins*(nspins-1)), delcols)
    nonzeros = np.nonzero(gmatsmspim+gmatsmspre)
    indexdouble = nonzeros[1]
    maskre = np.zeros((indexdouble.shape), dtype=indexdouble.dtype)
    maskim = np.zeros((indexdouble.shape), dtype=indexdouble.dtype)

    for i, val in enumerate(indexdouble):
        if (gmatsmspre[nonzeros[0][i], nonzeros[1][i]] != 0):
            maskre[i] = 1
        if (gmatsmspim[nonzeros[0][i], nonzeros[1][i]] != 0):
            maskim[i] = 1
    indexswitchred = np.delete(indexswitch, delcols)

    return np.ascontiguousarray(gmat), indexi, indexj, indexi2, indexj2, indexswitch, indexred, indexswitchred, np.ascontiguousarray(gmatsmspre), np.ascontiguousarray(gmatsmspim), indexdouble, maskre, maskim

def correct_for_fano(omegap, realpart, imagpart, phi, dt, offset, offset_phi):
    ampl = realpart+1j*imagpart
    s1 = np.exp(1j*phi)*np.exp(1j*dt*omegap)
    s3 = offset*np.exp(1j*offset_phi)
    retampl = ampl*s1-s3
    return retampl.real, retampl.imag


def prepare_solution(solution, init, pdf, tlist):
    #check if HP approximation is used
    hp = False
    if len(init) == len(pdf)+1:
        hp = True
    #reshape_idx for if g_list is bigger than 1
    if not hp:
        reshape_idx = int(np.round(len(init)-1)/2/len(pdf))
        solution_dummy = 0
        solution_return = np.empty((len(tlist), len(init)), dtype=complex)
        for k in range(reshape_idx):
            solution_dummy += solution[:, (1+2*k*len(pdf)):(1+2*(k+1)*len(pdf))]

        solution_return[:,1::2] = solution_dummy[:,:len(pdf)]*pdf/reshape_idx
        solution_return[:,2::2] = solution_dummy[:,len(pdf):]*pdf/reshape_idx
        solution_return[:, 0] = solution[:,0]
    else:
        reshape_idx = int(np.round(len(init)-1)/len(pdf))
        print(reshape_idx)
        solution_dummy = 0
        solution_return = np.empty((len(tlist), len(pdf)+1), dtype=complex)
        for k in range(reshape_idx):
            solution_dummy += solution[:, (1+k*len(pdf)):(1+(k+1)*len(pdf))]
        solution_return[:,1:] = solution_dummy[:,:]*pdf/reshape_idx
        solution_return[:, 0] = solution[:,0]
    return solution_return


def prepare_solution_2ndorder(solution, init, pdf, tlist):
    n = len(init)
    nspins = len(pdf)
    # only keep sm, sz and a in order for the solution to work the same way as for first order
    solution_return = np.empty((len(tlist), len(pdf)*2+1), dtype=np.complex128)
    solution_return[:, 1::2] = solution[:, (nspins*3+3):(nspins*4+3)]*pdf
    solution_return[:, 2::2] = solution[:, (nspins*4+3):(nspins*5+3)]*pdf
    solution_return[:, 0] = solution[:,0]
    return solution_return


def prepare_solution_no_cav(solution, init, pdf, tlist):
    solution_return = np.empty((len(tlist), len(pdf)*2), dtype=complex)
    solution_return[:,1::2] = solution[:,len(pdf):]*pdf
    solution_return[:,0::2] = solution[:,:len(pdf)]*pdf
    return solution_return
    
    

def replace_in_file(file_path, variable_name, new_value):
    with fileinput.input(file_path, inplace=True) as f:
        for line in f:
            new_line = re.sub('^(%s\W)(.*)'%variable_name, '%s = %s'%(variable_name, new_value), line);
            print(new_line, end='')

def normalize(values, actual_bounds, desired_bounds):
    return np.array([desired_bounds[0] + (x - actual_bounds[0]) * (desired_bounds[1] - desired_bounds[0]) / (actual_bounds[1] - actual_bounds[0]) for x in values])
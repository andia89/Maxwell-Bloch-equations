import numba
c_sig = numba.types.void(
                numba.types.int64,
                numba.types.CPointer(numba.types.complex128),
                numba.types.CPointer(numba.types.complex128),
                numba.types.float64,
                numba.types.float64
        )
import numpy as np
@numba.cfunc(c_sig)
def f(n_states, du_, u_, p_0, t):
    u = numba.carray(u_, n_states)
    du = numba.carray(du_, n_states)

    eta = p_0
    nspins = len(spins)

    deltas = spins*2*np.pi+deltac
    gamma = gperp+gpar/2
    a = u[0]
    adag = np.conjugate(a)
    adaga = u[1]
    aa = u[2]
    adagadag = np.conjugate(aa)
    asm = u[3:(nspins+3)]
    adagsp = np.conjugate(asm)
    adagsm = u[(nspins+3):(nspins*2+3)]
    asp = np.conjugate(adagsm)
    asz = u[(nspins*2+3):(nspins*3+3)]
    adagsz = np.conjugate(asz)
    sm = u[(nspins*3+3):(nspins*4+3)]
    sp = np.conjugate(sm)
    sz = u[(nspins*4+3):(nspins*5+3)]
    
    smsm = u[(nspins*5+3):(nspins*5+3)+nspins*(nspins-1)//2]
    smsp = u[(nspins*5+3)+nspins*(nspins-1)//2:(nspins*5+3)+nspins*(nspins-1)]
    spsm = np.conjugate(smsp)
    szsm = u[(nspins*5+3)+nspins*(nspins-1):(nspins*5+3)+2*nspins*(nspins-1)]
    szsp = np.conjugate(szsm)
    szsz = u[(nspins*5+3)+2*nspins*(nspins-1):(nspins*5+3)+2*nspins*(nspins-1)+nspins*(nspins-1)//2]
    
    smsp1 = smsp[indexdouble]
    spsm1 = spsm[indexdouble]
    smsp1[np.where(maskim==0)] = 0.+0j
    spsm1[np.where(maskre==0)] = 0.+0j

    du[0] = -kappa*a - 1j*deltac*a + gs.dot(sm) + eta
    
    du[1] = -2*kappa*adaga + eta * (adag+a) + gs.dot(adagsm + asp)
    
    du[2] = -2*(kappa + 1j*deltac)*aa + 2 * eta*a + 2*gs.dot(asm)

    du[3:(nspins+3)] = -(kappa+gamma+1j*(deltas+deltac))*asm + eta*sm + gs*(sz*aa+2*asz*a-2*sz*a*a) + gmat2.dot(smsm)

    du[(nspins+3):(nspins*2+3)] = -(kappa+gamma+1j*2*np.pi*spins)*adagsm + eta*sm  + gs*(asz*adag+adagsz*a+sz*adaga-2*sz*a*adag) + gs*0.5*(sz+1) + gmatsmspre.dot(smsp)+gmatsmspim.dot(spsm)

    du[(nspins*2+3):(nspins*3+3)] = -(gpar+1j*deltac+kappa)*asz - gpar*a  + eta*sz - \
        gs*sm - 2*gs*(adagsm*a+asm*adag+sp*aa+sm*adaga) - 4*gs*(asp*a-sm*adag*a-sp*a*a) + gmat.dot(szsm)

    du[(nspins*3+3):(nspins*4+3)] = -(gamma + 1j*deltas)*sm + gs*asz
    
    du[(nspins*4+3):(nspins*5+3)] = -gpar*(1+sz)-4*gs*(np.real(adagsm))

    du[(nspins*5+3):(nspins*5+3)+nspins*(nspins-1)//2] = -(2*gamma+1j*(deltas[indexj2]+deltas[indexi2]))*smsm + gs[indexj2]*(asz[indexj2]*sm[indexi2] +
szsm[indexred]*a-2*sz[indexj2]*sm[indexi2]*a+asm[indexi2]*sz[indexj2]) + gs[indexi2]*(asm[indexj2]*sz[indexi2]+asz[indexi2]*sm[indexj2]+szsm[indexswitchred]*a-2*sm[indexj2]*sz[indexi2]*a)

    du[(nspins*5+3)+nspins*(nspins-1)//2:(nspins*5+3)+nspins*(nspins-1)] = -(2*gamma+1j*(deltas[indexj2]-deltas[indexi2]))*smsp + gs[indexj2]*(szsp[indexred]*a+asz[indexj2]*sp[indexi2]+asp[indexi2]* sz[indexj2]-2*sz[indexj2]*sp[indexi2]*a)+gs[indexi2]*(adagsm[indexj2]*sz[indexi2]+adagsz[indexi2]*sm[indexj2]+szsm[indexswitchred]*adag-2*sm[indexj2]*sz[indexi2]*adag)
#
    du[(nspins*5+3)+nspins*(nspins-1):(nspins*5+3)+2*nspins*(nspins-1)] = -(1j*deltas[indexi]+gamma+gpar)*szsm - gpar*sm[indexi] + gs[indexi]*(asz[indexi]*sz[indexj]+asz[indexj]*sz[indexi]+szsz[indexdouble]*a-2*sz[indexj]*sz[indexi]*a) - 2*gs[indexj] *(adagsm[indexi]*sm[indexj]+adagsm[indexj]*sm[indexi]+asp[indexj]*sm[indexi]+asm[indexi]*sp[indexj]+smsm[indexdouble]*adag-2*sm[indexj]*sm[indexi]*adag-2*sp[indexj]*sm[indexi]*a+(smsp1+spsm1)*a)
    
    du[(nspins*5+3)+2*nspins*(nspins-1):(nspins*5+3)+2*nspins*(nspins-1)+nspins*(nspins-1)//2] = -gpar*(sz[indexj2]+sz[indexi2])-2*gpar*szsz - \
    2*gs[indexj2]*(adagsz[indexi2]*sm[indexj2]+asz[indexi2]*sp[indexj2]+adagsm[indexj2]*sz[indexi2]+asp[indexj2]*sz[indexi2]+szsp[indexswitchred]*a+szsm[indexswitchred]*adag-2*sp[indexj2]*sz[indexi2]*a-2*sm[indexj2]*sz[indexi2]*adag) - \
    2*gs[indexi2]*(adagsm[indexi2]*sz[indexj2]+asp[indexi2]*sz[indexj2]+adagsz[indexj2]*sm[indexi2]+asz[indexj2]*sp[indexi2]+szsm[indexred]*adag+szsp[indexred]*a-2*sz[indexj2]*sp[indexi2]*a-2*sz[indexj2]*sm[indexi2]*adag)

function mbes_2ndorder(du, u, p, t)
    if (length(p) < 25)
        eta, kappa, gperp, gpar, spins, deltac, gs, gmat, indexi, indexj, indexi2, indexj2, indexswitch, indexred, indexswitchred, gmatsmspre, gmatsmspim, indexdouble, maskre, maskim = p
    else
        eta, kappa, gperp, gpar, spins, deltac, gs, gmat, indexi, indexj, indexi2, indexj2, indexswitch, indexred, indexswitchred, gmatsmspre, gmatsmspim, indexdouble, maskre, maskim, gmatreidx, gmatimidx, gmatidx, gmatjacre, gmatjacim, gmatjacre2, gmatjacim2, gstiled, gstiled2, gstiled3, gstiled4, gstiledswitch, gparmat, gparmat2 = p
    end
    nspins = length(spins)
    deltas = (((spins*2)*pi) .+ deltac)
    gamma = (gperp + (gpar/2))
    
    a = u[1]
    adag = conj(a)
    adaga = u[2]
    aa = u[3]
    adagadag = conj(aa)
    asm = @view u[4:(nspins + 3)]
    
    adagsp = conj(asm)
    adagsm = @view u[(nspins + 4):((nspins*2) + 3)]
    asp = conj(adagsm)
    asz = @view u[((nspins*2) + 4):((nspins*3) + 3)]
    adagsz = conj(asz)
    sm = @view u[((nspins*3) + 4):((nspins*4) + 3)]
    sp = conj(sm)
    sz = @view u[((nspins*4) + 4):((nspins*5) + 3)]
    
    
    smsm = @view u[((nspins*5) + 4):(((nspins*5) + 3) + ((nspins*(nspins - 1)) ÷ 2))]
    smsp = @view u[(((nspins*5) + 4) + ((nspins*(nspins - 1)) ÷ 2)):(((nspins*5) + 3) + (nspins*(nspins - 1)))]
    spsm = conj(smsp)
    szsm = @view u[(((nspins*5) + 4) + (nspins*(nspins - 1))):(((nspins*5) + 3) + ((2*nspins)*(nspins - 1)))]
    szsp = conj(szsm)
    szsz = @view u[(((nspins*5) + 4) + ((2*nspins)*(nspins - 1))):((((nspins*5) + 3) + ((2*nspins)*(nspins - 1))) + ((nspins*(nspins - 1)) ÷ 2))]
    smsp1 = smsp[indexdouble]
    spsm1 = spsm[indexdouble]
    
    smsp1[maskim .== 1] .= (0.0 + 0im)
    spsm1[maskre .== 1] .= (0.0 + 0im)
    
    
    du[1] = ((((-(kappa)*a) - ((1im*deltac)*a)) + dot(gs, sm)) + eta)
    du[2] = ((((-2*kappa)*adaga) + (eta*(adag + a))) + dot(gs, (adagsm + asp)))

    du[3] = ((((-2*(kappa + (1im*deltac)))*aa) + ((2*eta)*a)) + 2*dot(gs, asm))

    
    du[4:(nspins + 3)] = -(kappa.+gamma.+1im*(deltas.+deltac)).*asm + eta*sm + gs.*(sz*aa+2*asz*a-2*sz*a*a) + (gmatsmspre .+ gmatsmspim)*smsm



    du[(nspins + 4):((nspins*2) + 3)] = ((((((-(((kappa + gamma) .+ (((1im*2)*pi).*spins))).*adagsm) + (eta.*sm)) + (gs.*((((asz.*adag) + (adagsz.*a)) + (sz.*adaga)) - (((2*sz).*a).*adag)))) + ((gs.*0.5).*(sz .+ 1))) + gmatsmspre*smsp) + gmatsmspim*spsm)

    du[((nspins*2) + 4):((nspins*3) + 3)] = (((((((-(((gpar .+ (1im*deltac)) .+ kappa)).*asz) .- (gpar*a)) + (eta*sz)) .- (gs.*sm)) .- ((2*gs).*((((adagsm*a) + (asm*adag)) + (sp*aa)) + (sm*adaga)))) .- ((4*gs).*(((asp*a) .- ((sm*adag)*a)) .- ((sp*a)*a)))) + gmat* szsm)

    du[((nspins*3) + 4):((nspins*4) + 3)] = ((-((gamma .+ (1im*deltas))).*sm) + (gs.*asz))
    du[((nspins*4) + 4):((nspins*5) + 3)] = ((-(gpar)*(1 .+ sz)) - ((4*gs).*real(adagsm)))

    du[((nspins*5) + 4):(((nspins*5) + 3) + ((nspins*(nspins - 1)) ÷ 2))] = (((-(((2*gamma) .+ (1im*(deltas[indexj2] + deltas[indexi2])))).*smsm) + (gs[indexj2].*((((asz[indexj2].*sm[indexi2]) + (szsm[indexred]*a)) - (((2*sz[indexj2]).*sm[indexi2])*a)) + (asm[indexi2].*sz[indexj2])))) + (gs[indexi2].*((((asm[indexj2].*sz[indexi2]) + (asz[indexi2].*sm[indexj2])) + (szsm[indexswitchred]*a)) - ((2*sm[indexj2].*sz[indexi2])*a))))
    
    du[(((nspins*5) + 4) + ((nspins*(nspins - 1)) ÷ 2)):(((nspins*5) + 3) + (nspins*(nspins - 1)))] = (((-(((2*gamma) .+ (1im*(deltas[indexj2] - deltas[indexi2])))).*smsp) + (gs[indexj2].*((((szsp[indexred]*a) + (asz[indexj2].*sp[indexi2])) + (asp[indexi2].*sz[indexj2])) - (((2*sz[indexj2]).*sp[indexi2])*a)))) + (gs[indexi2].*((((adagsm[indexj2].*sz[indexi2]) + (adagsz[indexi2].*sm[indexj2])) + (szsm[indexswitchred]*adag)) - (((2*sm[indexj2]).*sz[indexi2])*adag))))
    
    du[(((nspins*5) + 4) + (nspins*(nspins - 1))):(((nspins*5) + 3) + ((2*nspins)*(nspins - 1)))] = ((((-((((1im*deltas[indexi]) .+ gamma) .+ gpar)).*szsm) - (gpar*sm[indexi])) + (gs[indexi].*((((asz[indexi].*sz[indexj]) + (asz[indexj].*sz[indexi])) + (szsz[indexdouble]*a)) - (((2*sz[indexj]).*sz[indexi])*a)))) - ((2*gs[indexj]).*((((((((adagsm[indexi].*sm[indexj]) + (adagsm[indexj].*sm[indexi])) + (asp[indexj].*sm[indexi])) + (asm[indexi].*sp[indexj])) + (smsm[indexdouble]*adag)) - (((2*sm[indexj]).*sm[indexi])*adag)) - (((2*sp[indexj]).*sm[indexi])*a)) + ((smsp1 + spsm1)*a))))


    du[(((nspins*5) + 4) + ((2*nspins)*(nspins - 1))):((((nspins*5) + 3) + ((2*nspins)*(nspins - 1))) + ((nspins*(nspins - 1)) ÷ 2))] = ((((-(gpar)*(sz[indexj2] + sz[indexi2])) - ((2*gpar)*szsz)) - ((2*gs[indexj2]).*((((((((adagsz[indexi2].*sm[indexj2]) + (asz[indexi2].*sp[indexj2])) + (adagsm[indexj2].*sz[indexi2])) + (asp[indexj2].*sz[indexi2])) + (szsp[indexswitchred]*a)) + (szsm[indexswitchred]*adag)) - (((2*sp[indexj2]).*sz[indexi2])*a)) - (((2*sm[indexj2]).*sz[indexi2])*adag)))) - ((2*gs[indexi2]).*((((((((adagsm[indexi2].*sz[indexj2]) + (asp[indexi2].*sz[indexj2])) + (adagsz[indexj2].*sm[indexi2])) + (asz[indexj2].*sp[indexi2])) + (szsm[indexred]*adag)) + (szsp[indexred]*a)) - (((2*sz[indexj2]).*sp[indexi2])*a)) - (((2*sz[indexj2]).*sm[indexi2])*adag))))
end



function mbes(du, u, p, t)
    eta, kappa, deltac, gs, gperp, spins, gpar = p
    a = u[1]
    sm = @view u[2:length(spins)+1] 
    sz = @view u[length(spins)+2:end]

    du[1] = -(kappa + 1im*deltac)*a + dot(gs, sm) + eta
    du[2:length(spins)+1] = -(gperp+gpar/2 .+ 1im*(spins*2*pi.+deltac)).*sm+gs.*sz*a
    du[length(spins)+2:end] = -gpar*(1. .+ sz)-2*gs.*(sm*conj(a).+conj(sm)*a)
end
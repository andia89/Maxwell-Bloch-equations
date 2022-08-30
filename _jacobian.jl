
function jacobian_mbes{T0, T1, RT}(Y::T0, t::T1)::RT
kappa = args[-6]
deltac = args[-5]
gs = args[-4]
gperp = args[-3]
spins = args[-2]
gpar = args[-1]
gamma = (gperp + (gpar/2))
deltas = (((spins*2)*np.pi) + deltac)
leny = length(Y)
indices0 = arange(np, 2, leny, 4)
indices1 = arange(np, 3, leny, 4)
indices2 = arange(np, 4, leny, 4)
indices3 = arange(np, 5, leny, 4)
J = zeros(np, (leny, leny))
J[(0, 0)] = -(kappa)
J[(0, 1)] = deltac
J[(0, 2..)] = gs
J[(1, 0)] = -(deltac)
J[(1, 1)] = -(kappa)
J[(1, 3..)] = gs
J[(2.., 0)] = (gs*Y[4..])
J[(3.., 0)] = (gs*Y[5..])
J[(4.., 0)] = ((-4*gs)*Y[2..])
J[(2.., 1)] = (-(gs)*Y[5..])
J[(3.., 1)] = (gs*Y[4..])
J[(4.., 1)] = ((-4*gs)*Y[3..])
J[(indices0, indices0)] = -(gamma)
J[(indices1, indices1)] = -(gamma)
J[(indices2, indices2)] = -(gpar)
J[(indices3, indices3)] = -(gpar)
J[(indices0, indices1)] = deltas
J[(indices1, indices0)] = -(deltas)
J[(indices0, indices2)] = (gs*Y[0])
J[(indices0, indices3)] = (-(gs)*Y[1])
J[(indices1, indices2)] = (gs*Y[1])
J[(indices1, indices3)] = (gs*Y[0])
J[(indices2, indices0)] = ((-4*gs)*Y[0])
J[(indices2, indices1)] = ((-4*gs)*Y[1])
return J
end

function jacobian_mbes_masing{T0, T1, RT}(Y::T0, t::T1)::RT
pump = args[-7]
kappa = args[-6]
deltac = args[-5]
gs = args[-4]
gperp = args[-3]
spins = args[-2]
gpar = args[-1]
gamma = ((gperp + (gpar/2)) + (pump/2))
deltas = (((spins*2)*np.pi) + deltac)
leny = length(Y)
indices0 = arange(np, 2, leny, 4)
indices1 = arange(np, 3, leny, 4)
indices2 = arange(np, 4, leny, 4)
indices3 = arange(np, 5, leny, 4)
J = zeros(np, (leny, leny))
J[(0, 0)] = -(kappa)
J[(0, 1)] = deltac
J[(0, 2..)] = gs
J[(1, 0)] = -(deltac)
J[(1, 1)] = -(kappa)
J[(1, 3..)] = gs
J[(2.., 0)] = (gs*Y[4..])
J[(3.., 0)] = (gs*Y[5..])
J[(4.., 0)] = ((-4*gs)*Y[2..])
J[(2.., 1)] = (-(gs)*Y[5..])
J[(3.., 1)] = (gs*Y[4..])
J[(4.., 1)] = ((-4*gs)*Y[3..])
J[(indices0, indices0)] = -(gamma)
J[(indices1, indices1)] = -(gamma)
J[(indices2, indices2)] = (-(gpar) - pump)
J[(indices3, indices3)] = (-(gpar) - pump)
J[(indices0, indices1)] = deltas
J[(indices1, indices0)] = -(deltas)
J[(indices0, indices2)] = (gs*Y[0])
J[(indices0, indices3)] = (-(gs)*Y[1])
J[(indices1, indices2)] = (gs*Y[1])
J[(indices1, indices3)] = (gs*Y[0])
J[(indices2, indices0)] = ((-4*gs)*Y[0])
J[(indices2, indices1)] = ((-4*gs)*Y[1])
return J
end

function jacobian_mbes_hp{T0, T1, RT}(Y::T0, t::T1)::RT
kappa = args[-6]
deltac = args[-5]
gs = args[-4]
gperp = args[-3]
spins = args[-2]
gpar = args[-1]
gamma = (gperp + (gpar/2))
deltas = (((spins*2)*np.pi) + deltac)
leny = length(Y)
indices0 = arange(np, 2, leny, 2)
indices1 = arange(np, 3, leny, 2)
J = zeros(np, (leny, leny))
J[(0, 0)] = -(kappa)
J[(0, 1)] = deltac
J[(0, 2..)] = gs
J[(1, 0)] = -(deltac)
J[(1, 1)] = -(kappa)
J[(1, 3..)] = gs
J[(2.., 0)] = -(gs)
J[(3.., 1)] = -(gs)
J[(indices0, indices0)] = -(gamma)
J[(indices1, indices1)] = -(gamma)
J[(indices0, indices1)] = deltas
J[(indices1, indices0)] = -(deltas)
return J
end

function jacobian_mbes_no_cavity{T0, T1, RT}(t::T0, Y::T1)::RT
areal = args[-8]
aimag = args[-7]
tlist = args[-6]
deltac = args[-5]
gs = args[-4]
gperp = args[-3]
spins = args[-2]
gpar = args[-1]
gamma = (gperp + (gpar/2))
deltas = (((spins*2)*np.pi) + deltac)
idx = find_nearest_sorted(tlist, t)
are = areal[idx]
aim = areal[idx]
leny = length(Y)
indices0 = arange(np, 0, leny, 4)
indices1 = arange(np, 1, leny, 4)
indices2 = arange(np, 2, leny, 4)
indices3 = arange(np, 3, leny, 4)
J = zeros(np, (leny, leny))
J[(indices0, indices0)] = -(gamma)
J[(indices1, indices1)] = -(gamma)
J[(indices2, indices2)] = -(gpar)
J[(indices3, indices3)] = -(gpar)
J[(indices0, indices1)] = deltas
J[(indices1, indices0)] = -(deltas)
J[(indices0, indices2)] = (gs*are)
J[(indices0, indices3)] = (-(gs)*aim)
J[(indices1, indices2)] = (gs*aim)
J[(indices1, indices3)] = (gs*are)
J[(indices2, indices0)] = ((-4*gs)*are)
J[(indices2, indices1)] = ((-4*gs)*aim)
return J
end

function jacobian_mbes_2ndorder_real{T0, T1, RT}(t::T0, Y::T1)::RT
eta, kappa, gperp, gpar, spins, deltac, gs, gmat, gmat2, indexi, indexj, indexi2, indexj2, indexswitch, indexred, indexswitchred, gmatsmspre, gmatsmspim, indexdouble, maskre, maskim, gmatreidx, gmatimidx, gmatidx, gmatjacre, gmatjacim, gmatjacre2, gmatjacim2, gstiled, gstiled2, gstiled3, gstiled4, gstiledswitch, gparmat, gparmat2, J = args
nspins = length(spins)
gsreal = real(np, gs)
gmatreal = real(np, gmat)
gmatsmsprereal = real(np, gmatsmspre)
gmatsmspimreal = real(np, gmatsmspim)
deltas = (((spins*2)*np.pi) + deltac)
gamma = (gperp + (gpar/2))
stepsize = (nspins*(nspins - 1))
are = Y[0]
aim = Y[1]
adagare = Y[2]
adagaim = Y[3]
aare = Y[4]
aaim = Y[5]
asmre = Y[6..((nspins*2) + 6)]
asmim = Y[7..((nspins*2) + 6)]
adagsmre = Y[((nspins*2) + 6)..((nspins*4) + 6)]
adagsmim = Y[(((nspins*2) + 6) + 1)..((nspins*4) + 6)]
aszre = Y[((nspins*4) + 6)..((nspins*6) + 6)]
aszim = Y[(((nspins*4) + 6) + 1)..((nspins*6) + 6)]
smre = Y[((nspins*6) + 6)..((nspins*8) + 6)]
smim = Y[(((nspins*6) + 6) + 1)..((nspins*8) + 6)]
szre = Y[((nspins*8) + 6)..((nspins*10) + 6)]
szim = Y[(((nspins*8) + 6) + 1)..((nspins*10) + 6)]
smsmre = Y[((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1)))]
smsmim = Y[(((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1)))]
smspre = Y[(((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))]
smspim = Y[((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))]
szsmre = Y[(((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))]
szsmim = Y[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))]
szszre = Y[(((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1)))]
szszim = Y[((((nspins*10) + 6) + ((4*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1)))]
szre1 = zeros(np, gmatreidx.shape)
szre2 = zeros(np, gmatreidx.shape)
smre1 = zeros(np, gmatreidx.shape)
smre2 = zeros(np, gmatreidx.shape)
smim1 = zeros(np, gmatreidx.shape)
smim2 = zeros(np, gmatreidx.shape)
asmre1 = zeros(np, gmatreidx.shape)
asmre2 = zeros(np, gmatreidx.shape)
asmim1 = zeros(np, gmatreidx.shape)
asmim2 = zeros(np, gmatreidx.shape)
adagsmre1 = zeros(np, gmatreidx.shape)
adagsmre2 = zeros(np, gmatreidx.shape)
adagsmim1 = zeros(np, gmatreidx.shape)
adagsmim2 = zeros(np, gmatreidx.shape)
aszre1 = zeros(np, gmatreidx.shape)
aszre2 = zeros(np, gmatreidx.shape)
aszim1 = zeros(np, gmatreidx.shape)
aszim2 = zeros(np, gmatreidx.shape)
for i in (0:nspins - 1)
idx0 = where(np, gmatreidx[i] == -1)
idx1 = where(np, gmatimidx[i] == -1)
temp = szre[gmatreidx[i]]
temp[idx0] = 0.0
szre1[(i, ..)] = temp
temp = szre[gmatimidx[i]];
temp[idx1] = 0.0
szre2[(i, ..)] = temp
temp = smre[gmatreidx[i]];
temp[idx0] = 0.0
smre1[(i, ..)] = temp
temp = smre[gmatimidx[i]];
temp[idx1] = 0.0
smre2[(i, ..)] = temp
temp = smim[gmatreidx[i]];
temp[idx0] = 0.0
smim1[(i, ..)] = temp
temp = smim[gmatimidx[i]];
temp[idx1] = 0.0
smim2[(i, ..)] = temp
temp = asmre[gmatreidx[i]];
temp[idx0] = 0.0
asmre1[(i, ..)] = temp
temp = asmre[gmatimidx[i]];
temp[idx1] = 0.0
asmre2[(i, ..)] = temp
temp = adagsmre[gmatreidx[i]];
temp[idx0] = 0.0
adagsmre1[(i, ..)] = temp
temp = adagsmre[gmatimidx[i]];
temp[idx1] = 0.0
adagsmre2[(i, ..)] = temp
temp = adagsmim[gmatreidx[i]];
temp[idx0] = 0.0
adagsmim1[(i, ..)] = temp
temp = adagsmim[gmatimidx[i]];
temp[idx1] = 0.0
adagsmim2[(i, ..)] = temp
temp = asmim[gmatreidx[i]];
temp[idx0] = 0.0
asmim1[(i, ..)] = temp
temp = asmim[gmatimidx[i]];
temp[idx1] = 0.0
asmim2[(i, ..)] = temp
temp = aszre[gmatreidx[i]];
temp[idx0] = 0.0
aszre1[(i, ..)] = temp
temp = aszre[gmatimidx[i]];
temp[idx1] = 0.0
aszre2[(i, ..)] = temp
temp = aszim[gmatreidx[i]];
temp[idx0] = 0.0
aszim1[(i, ..)] = temp
temp = aszim[gmatimidx[i]];
temp[idx1] = 0.0
aszim2[(i, ..)] = temp
end
smref = zeros(np, gstiled2.shape)
smimf = zeros(np, gstiled2.shape)
szref = zeros(np, gstiled2.shape)
smref2 = zeros(np, gstiled2.shape)
adagsmref = zeros(np, gstiled2.shape)
adagsmref2 = zeros(np, gstiled2.shape)
adagsmimf2 = zeros(np, gstiled2.shape)
asmref2 = zeros(np, gstiled2.shape)
asmimf2 = zeros(np, gstiled2.shape)
aszref = zeros(np, gstiled2.shape)
aszref2 = zeros(np, gstiled2.shape)
aszimf = zeros(np, gstiled2.shape)
aszimf2 = zeros(np, gstiled2.shape)
smimf2 = zeros(np, gstiled2.shape)
szref2 = zeros(np, gstiled2.shape)
szref3 = zeros(np, gstiled2.shape)
smimf = zeros(np, gstiled2.shape);
ctr = -1
ctr2 = 1
for (i, row) in gstiled2.iter().enumerate()
if (i % (nspins - 1)) == 0
ctr += 1
ctr2 = 0
end
if ctr == ctr2
ctr2 += 1
end
smref[i] = (gstiled2[i]*smre[ctr])
adagsmref[i] = (gstiled2[i]*adagsmre[ctr])
smimf[i] = (gstiled2[i]*smim[ctr])
smref2[(i, ctr)] = (smre[ctr2]*gsreal[ctr])
adagsmref2[(i, ctr)] = (adagsmre[ctr2]*gsreal[ctr])
adagsmimf2[(i, ctr)] = (adagsmim[ctr2]*gsreal[ctr])
asmref2[(i, ctr)] = (asmre[ctr2]*gsreal[ctr])
asmimf2[(i, ctr)] = (asmim[ctr2]*gsreal[ctr])
smimf2[(i, ctr)] = (smim[ctr2]*gsreal[ctr])
szref2[(i, ctr)] = (szre[ctr2]*gsreal[ctr2])
smimf[i] = (gstiled2[i]*smim[ctr])
szref3[i] = (gstiled4[i]*szre[ctr])
aszref2[(i, ctr)] = (gsreal[ctr2]*aszre[ctr2])
aszref[i] = (gstiled4[i]*aszre[ctr])
aszimf2[(i, ctr)] = (gsreal[ctr2]*aszim[ctr2])
aszimf[i] = (gstiled4[i]*aszim[ctr])
szref[(i, ctr)] = (gsreal[ctr2]*szre[ctr2])
ctr2 += 1
end
smspim1 = smspim[indexdouble]
spsmim1 = -(smspim[indexdouble])
smspim1[where(np, maskre == 0)] = 0.0
spsmim1[where(np, maskim == 0)] = 0.0
J[(0, 0)] = -(kappa)
J[(0, 1)] = deltac
J[(0, 2..)] = 0.0
J[(0, ((nspins*6) + 6)..((nspins*8) + 6))] = gsreal
J[(1, 0)] = -(deltac)
J[(1, 1)] = -(kappa)
J[(1, (((nspins*6) + 6) + 1)..((nspins*8) + 6))] = gsreal
J[(2, 0)] = (2*eta)
J[(2, 1)] = 0.0
J[(2, 2)] = (-2*kappa)
J[(2, 3..)] = 0.0
J[(2, ((nspins*2) + 6)..((nspins*4) + 6))] = (2*gsreal)
J[(3, ..)] = 0.0
J[(3, 3)] = (-2*kappa)
J[(4, 0)] = (2*eta)
J[(4, 1..)] = 0.0
J[(4, 4)] = (-2*kappa)
J[(4, 5)] = (2*deltac)
J[(4, 6..((nspins*2) + 6))] = (2*gsreal)
J[(5, ..)] = 0.0
J[(5, 1)] = (2*eta)
J[(5, 4)] = (-2*deltac)
J[(5, 5)] = (-2*kappa)
J[(5, 7..((nspins*2) + 6))] = (2*gsreal)
J[(6..((nspins*2) + 6), 0)] = (((2*aszre)*gsreal) - (((4*gsreal)*are)*szre))
J[(6..((nspins*2) + 6), 1)] = (((-2*aszim)*gsreal) + (((4*aim)*gsreal)*szre))
J[(6..((nspins*2) + 6), 2..)] = 0.0
J[(6..((nspins*2) + 6), 4)] = (gsreal*szre)
fill_diagonal(np, J[(6..((nspins*2) + 6), 6..((nspins*2) + 6))], (-(gamma) - kappa));
fill_diagonal(np, J[(6..((nspins*2) + 6), 7..((nspins*2) + 6))], (deltac + deltas));
fill_diagonal(np, J[(6..((nspins*2) + 6), ((nspins*4) + 6)..((nspins*6) + 6))], ((2*are)*gsreal));
fill_diagonal(np, J[(6..((nspins*2) + 6), (((nspins*4) + 6) + 1)..((nspins*6) + 6))], ((-2*aim)*gsreal));
fill_diagonal(np, J[(6..((nspins*2) + 6), ((nspins*6) + 6)..((nspins*8) + 6))], eta);
fill_diagonal(np, J[(6..((nspins*2) + 6), ((nspins*8) + 6)..((nspins*10) + 6))], (((aare*gsreal) + ((2*pow(aim, 2))*gsreal)) - ((2*pow(are, 2))*gsreal)));
J[(6..((nspins*2) + 6), ((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))))] = (gmatsmsprereal + gmatsmspimreal)
J[(7..((nspins*2) + 6), 0)] = (((2*aszim)*gsreal) - (((4*aim)*gsreal)*szre))
J[(7..((nspins*2) + 6), 1)] = (((2*aszre)*gsreal) - (((4*are)*gsreal)*szre))
J[(7..((nspins*2) + 6), 2..)] = 0.0
J[(7..((nspins*2) + 6), 5)] = (gsreal*szre)
fill_diagonal(np, J[(7..((nspins*2) + 6), 6..((nspins*2) + 6))], (-(deltac) - deltas));
fill_diagonal(np, J[(7..((nspins*2) + 6), 7..((nspins*2) + 6))], (-(gamma) - kappa));
fill_diagonal(np, J[(7..((nspins*2) + 6), ((nspins*4) + 6)..((nspins*6) + 6))], ((2*aim)*gsreal));
fill_diagonal(np, J[(7..((nspins*2) + 6), (((nspins*4) + 6) + 1)..((nspins*6) + 6))], ((2*are)*gsreal));
fill_diagonal(np, J[(7..((nspins*2) + 6), (((nspins*6) + 6) + 1)..((nspins*8) + 6))], eta);
fill_diagonal(np, J[(7..((nspins*2) + 6), ((nspins*8) + 6)..((nspins*10) + 6))], ((aaim*gsreal) - (((4*aim)*are)*gsreal)));
J[(7..((nspins*2) + 6), (((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))))] = (gmatsmsprereal + gmatsmspimreal)
J[(((nspins*2) + 6)..((nspins*4) + 6), 0)] = (((2*aszre)*gsreal) - (((4*are)*gsreal)*szre))
J[(((nspins*2) + 6)..((nspins*4) + 6), 1)] = (((2*aszim)*gsreal) - (((4*aim)*gsreal)*szre))
J[(((nspins*2) + 6)..((nspins*4) + 6), 2)] = (gsreal*szre)
J[(((nspins*2) + 6)..((nspins*4) + 6), 3..)] = 0.0
fill_diagonal(np, J[(((nspins*2) + 6)..((nspins*4) + 6), ((nspins*2) + 6)..((nspins*4) + 6))], (-(gamma) - kappa));
fill_diagonal(np, J[(((nspins*2) + 6)..((nspins*4) + 6), (((nspins*2) + 6) + 1)..((nspins*4) + 6))], (-(deltac) + deltas));
fill_diagonal(np, J[(((nspins*2) + 6)..((nspins*4) + 6), ((nspins*4) + 6)..((nspins*6) + 6))], ((2*are)*gsreal));
fill_diagonal(np, J[(((nspins*2) + 6)..((nspins*4) + 6), (((nspins*4) + 6) + 1)..((nspins*6) + 6))], ((2*aim)*gsreal));
fill_diagonal(np, J[(((nspins*2) + 6)..((nspins*4) + 6), ((nspins*6) + 6)..((nspins*8) + 6))], eta);
fill_diagonal(np, J[(((nspins*2) + 6)..((nspins*4) + 6), ((nspins*8) + 6)..((nspins*10) + 6))], ((((gsreal*0.5) + (adagare*gsreal)) - ((2*pow(aim, 2))*gsreal)) - ((2*pow(are, 2))*gsreal)));
J[(((nspins*2) + 6)..((nspins*4) + 6), (((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))))] = (gmatsmsprereal + gmatsmspimreal)
J[((((nspins*2) + 6) + 1)..((nspins*4) + 6), ..)] = 0.0
J[((((nspins*2) + 6) + 1)..((nspins*4) + 6), 3)] = (gsreal*szre)
fill_diagonal(np, J[((((nspins*2) + 6) + 1)..((nspins*4) + 6), ((nspins*2) + 6)..((nspins*4) + 6))], (deltac - deltas));
fill_diagonal(np, J[((((nspins*2) + 6) + 1)..((nspins*4) + 6), (((nspins*2) + 6) + 1)..((nspins*4) + 6))], (-(gamma) - kappa));
fill_diagonal(np, J[((((nspins*2) + 6) + 1)..((nspins*4) + 6), (((nspins*6) + 6) + 1)..((nspins*8) + 6))], eta);
fill_diagonal(np, J[((((nspins*2) + 6) + 1)..((nspins*4) + 6), ((nspins*8) + 6)..((nspins*10) + 6))], (adagaim*gsreal));
J[((((nspins*2) + 6) + 1)..((nspins*4) + 6), ((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))))] = (gmatsmsprereal - gmatsmspimreal)
J[(((nspins*4) + 6)..((nspins*6) + 6), 0)] = (-(gpar) - ((2*gsreal)*((((3*adagsmre) + asmre) - ((4*aim)*smim)) - ((8*are)*smre))))
J[(((nspins*4) + 6)..((nspins*6) + 6), 1)] = ((-2*gsreal)*((adagsmim + asmim) - ((4*are)*smim)))
J[(((nspins*4) + 6)..((nspins*6) + 6), 2)] = ((-2*gsreal)*smre)
J[(((nspins*4) + 6)..((nspins*6) + 6), 3)] = ((2*gsreal)*smim)
J[(((nspins*4) + 6)..((nspins*6) + 6), 4)] = ((-2*gsreal)*smre)
J[(((nspins*4) + 6)..((nspins*6) + 6), 5)] = ((-2*gsreal)*smim)
J[(((nspins*4) + 6)..((nspins*6) + 6), 6..)] = 0.0
fill_diagonal(np, J[(((nspins*4) + 6)..((nspins*6) + 6), 6..((nspins*2) + 6))], ((-2*are)*gsreal));
fill_diagonal(np, J[(((nspins*4) + 6)..((nspins*6) + 6), 7..((nspins*2) + 6))], ((-2*aim)*gsreal));
fill_diagonal(np, J[(((nspins*4) + 6)..((nspins*6) + 6), ((nspins*2) + 6)..((nspins*4) + 6))], ((-6*are)*gsreal));
fill_diagonal(np, J[(((nspins*4) + 6)..((nspins*6) + 6), (((nspins*2) + 6) + 1)..((nspins*4) + 6))], ((-2*aim)*gsreal));
fill_diagonal(np, J[(((nspins*4) + 6)..((nspins*6) + 6), ((nspins*4) + 6)..((nspins*6) + 6))], (-(gpar) - kappa));
fill_diagonal(np, J[(((nspins*4) + 6)..((nspins*6) + 6), (((nspins*4) + 6) + 1)..((nspins*6) + 6))], deltac);
fill_diagonal(np, J[(((nspins*4) + 6)..((nspins*6) + 6), ((nspins*6) + 6)..((nspins*8) + 6))], (-(gsreal)*(((1 + (2*aare)) + (2*adagare)) - (8*pow(are, 2)))));
fill_diagonal(np, J[(((nspins*4) + 6)..((nspins*6) + 6), (((nspins*6) + 6) + 1)..((nspins*8) + 6))], (gsreal*(((-2*aaim) + (2*adagaim)) + ((8*aim)*are))));
fill_diagonal(np, J[(((nspins*4) + 6)..((nspins*6) + 6), ((nspins*8) + 6)..((nspins*10) + 6))], eta);
J[(((nspins*4) + 6)..((nspins*6) + 6), (((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))] = gmatreal
J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), 0)] = (gsreal*(((2*adagsmim) - (2*asmim)) + ((8*aim)*smre)))
J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), 1)] = (-(gpar) + (gsreal*((((-6*adagsmre) + (2*asmre)) + ((16*aim)*smim)) + ((8*are)*smre))))
J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), 2)] = ((-2*gsreal)*smim)
J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), 3)] = ((-2*gsreal)*smre)
J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), 4)] = ((2*gsreal)*smim)
J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), 5)] = ((-2*gsreal)*smre)
J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), 6..)] = 0.0
fill_diagonal(np, J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), 6..((nspins*2) + 6))], ((2*aim)*gsreal));
fill_diagonal(np, J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), 7..((nspins*2) + 6))], ((-2*are)*gsreal));
fill_diagonal(np, J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), ((nspins*2) + 6)..((nspins*4) + 6))], ((-6*aim)*gsreal));
fill_diagonal(np, J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), (((nspins*2) + 6) + 1)..((nspins*4) + 6))], ((2*are)*gsreal));
fill_diagonal(np, J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), ((nspins*4) + 6)..((nspins*6) + 6))], -(deltac));
fill_diagonal(np, J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), (((nspins*4) + 6) + 1)..((nspins*6) + 6))], (-(gpar) - kappa));
fill_diagonal(np, J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), ((nspins*6) + 6)..((nspins*8) + 6))], ((((-2*aaim)*gsreal) - ((2*adagaim)*gsreal)) + (((8*aim)*are)*gsreal)));
fill_diagonal(np, J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), (((nspins*6) + 6) + 1)..((nspins*8) + 6))], (-(gsreal)*(((1 - (2*aare)) + (2*adagare)) - (8*pow(aim, 2)))));
J[((((nspins*4) + 6) + 1)..((nspins*6) + 6), ((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))] = gmatreal
J[(((nspins*6) + 6)..((nspins*8) + 6), ..)] = 0.0
fill_diagonal(np, J[(((nspins*6) + 6)..((nspins*8) + 6), ((nspins*4) + 6)..((nspins*6) + 6))], gsreal);
fill_diagonal(np, J[(((nspins*6) + 6)..((nspins*8) + 6), ((nspins*6) + 6)..((nspins*8) + 6))], -(gamma));
fill_diagonal(np, J[(((nspins*6) + 6)..((nspins*8) + 6), (((nspins*6) + 6) + 1)..((nspins*8) + 6))], deltas);
J[((((nspins*6) + 6) + 1)..((nspins*8) + 6), ..)] = 0.0
fill_diagonal(np, J[((((nspins*6) + 6) + 1)..((nspins*8) + 6), (((nspins*4) + 6) + 1)..((nspins*6) + 6))], gsreal);
fill_diagonal(np, J[((((nspins*6) + 6) + 1)..((nspins*8) + 6), ((nspins*6) + 6)..((nspins*8) + 6))], -(deltas));
fill_diagonal(np, J[((((nspins*6) + 6) + 1)..((nspins*8) + 6), (((nspins*6) + 6) + 1)..((nspins*8) + 6))], -(gamma));
J[(((nspins*8) + 6)..((nspins*10) + 6), ..)] = 0.0
fill_diagonal(np, J[(((nspins*8) + 6)..((nspins*10) + 6), ((nspins*2) + 6)..((nspins*4) + 6))], (-4*gsreal));
fill_diagonal(np, J[(((nspins*8) + 6)..((nspins*10) + 6), ((nspins*8) + 6)..((nspins*10) + 6))], -(gpar));
J[((((nspins*8) + 6) + 1)..((nspins*10) + 6), ..)] = 0.0
J[(((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))), 0)] = ((gsreal[indexj2]*(((-2*smre[indexi2])*szre[indexj2]) + szsmre[indexred])) + (gsreal[indexi2]*(((-2*smre[indexj2])*szre[indexi2]) + szsmre[indexswitchred])))
J[(((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))), 1)] = ((gsreal[indexj2]*(((2*smim[indexi2])*szre[indexj2]) - szsmim[indexred])) + (gsreal[indexi2]*(((2*smim[indexj2])*szre[indexi2]) - szsmim[indexswitchred])))
J[(((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))), 2..)] = 0.0
J[(((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))), 6..((nspins*2) + 6))] = multiply(np, (gmatsmsprereal.transpose() + gmatsmspimreal.transpose()), (szre1.transpose() + szre2.transpose()))
J[(((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))), ((nspins*4) + 6)..((nspins*6) + 6))] = multiply(np, gstiled, (smre1.transpose() + smre2.transpose()))
J[(((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))), (((nspins*4) + 6) + 1)..((nspins*6) + 6))] = -multiply(np, gstiled, (smim1.transpose() + smim2.transpose()))
J[(((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))), ((nspins*6) + 6)..((nspins*8) + 6))] = multiply(np, (gmatsmsprereal.transpose() + gmatsmspimreal.transpose()), ((aszre1.transpose() + aszre2.transpose()) - ((2*are)*(szre1.transpose() + szre2.transpose()))))
J[(((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))), (((nspins*6) + 6) + 1)..((nspins*8) + 6))] = multiply(np, (gmatsmsprereal.transpose() + gmatsmspimreal.transpose()), (-((aszim1.transpose() + aszim2.transpose())) + ((2*aim)*(szre1.transpose() + szre2.transpose()))))
J[(((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))), ((nspins*8) + 6)..((nspins*10) + 6))] = multiply(np, gstiled, ((((2*aim)*(smim1.transpose() + smim2.transpose())) - ((2*are)*(smre1.transpose() + smre2.transpose()))) + (asmre1.transpose() + asmre2.transpose())))
J[(((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))), (((nspins*8) + 6) + 1)..((nspins*10) + 6))] = 0.0
fill_diagonal(np, J[(((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))), ((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))))], (-2*gamma));
fill_diagonal(np, J[(((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))), (((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))))], (deltas[indexj2] + deltas[indexi2]));
J[(((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))), (((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))] = (are*(gmatjacre + gmatjacim))
J[(((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))), ((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))] = (-(aim)*(gmatjacre + gmatjacim))
J[((((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))), 0)] = ((gsreal[indexj2]*(((-2*smim[indexi2])*szre[indexj2]) + szsmim[indexred])) + (gsreal[indexi2]*(((-2*smim[indexj2])*szre[indexi2]) + szsmim[indexswitchred])))
J[((((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))), 1)] = ((gsreal[indexj2]*(((-2*smre[indexi2])*szre[indexj2]) + szsmre[indexred])) + (gsreal[indexi2]*(((-2*smre[indexj2])*szre[indexi2]) + szsmre[indexswitchred])))
J[((((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))), 2..)] = 0.0
J[((((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))), 7..((nspins*2) + 6))] = multiply(np, (gmatsmsprereal.transpose() + gmatsmspimreal.transpose()), (szre1.transpose() + szre2.transpose()))
J[((((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))), ((nspins*4) + 6)..((nspins*6) + 6))] = multiply(np, gstiled, (smim1.transpose() + smim2.transpose()))
J[((((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))), (((nspins*4) + 6) + 1)..((nspins*6) + 6))] = multiply(np, gstiled, (smre1.transpose() + smre2.transpose()))
J[((((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))), ((nspins*6) + 6)..((nspins*8) + 6))] = multiply(np, (gmatsmsprereal.transpose() + gmatsmspimreal.transpose()), ((aszim1.transpose() + aszim2.transpose()) - ((2*aim)*(szre1.transpose() + szre2.transpose()))))
J[((((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))), (((nspins*6) + 6) + 1)..((nspins*8) + 6))] = multiply(np, (gmatsmsprereal.transpose() + gmatsmspimreal.transpose()), ((aszre1.transpose() + aszre2.transpose()) - ((2*are)*(szre1.transpose() + szre2.transpose()))))
J[((((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))), ((nspins*8) + 6)..((nspins*10) + 6))] = multiply(np, gstiled, ((((-2*are)*(smim1.transpose() + smim2.transpose())) - ((2*aim)*(smre1.transpose() + smre2.transpose()))) + (asmim1.transpose() + asmim2.transpose())))
fill_diagonal(np, J[((((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))), ((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))))], (-(deltas[indexj2]) - deltas[indexi2]));
fill_diagonal(np, J[((((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))), (((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))))], (-2*gamma));
J[((((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))), (((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))] = (aim*(gmatjacre + gmatjacim))
J[((((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))), ((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))] = (are*(gmatjacre + gmatjacim))
J[((((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), 0)] = ((gsreal[indexj2]*(((-2*smre[indexi2])*szre[indexj2]) + szsmre[indexred])) + (gsreal[indexi2]*(((-2*smre[indexj2])*szre[indexi2]) + szsmre[indexswitchred])))
J[((((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), 1)] = ((gsreal[indexj2]*(((-2*smim[indexi2])*szre[indexj2]) + szsmim[indexred])) + (gsreal[indexi2]*(((-2*smim[indexj2])*szre[indexi2]) + szsmim[indexswitchred])))
J[((((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), 2..)] = 0.0
J[((((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), ((nspins*2) + 6)..((nspins*4) + 6))] = multiply(np, (gmatsmsprereal.transpose() + gmatsmspimreal.transpose()), (szre1.transpose() + szre2.transpose()))
J[((((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), ((nspins*4) + 6)..((nspins*6) + 6))] = multiply(np, gstiled, (smre1.transpose() + smre2.transpose()))
J[((((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), (((nspins*4) + 6) + 1)..((nspins*6) + 6))] = multiply(np, gstiled, (smim1.transpose() + smim2.transpose()))
J[((((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), ((nspins*6) + 6)..((nspins*8) + 6))] = multiply(np, (gmatsmsprereal.transpose() + gmatsmspimreal.transpose()), ((aszre1.transpose() + aszre2.transpose()) - ((2*are)*(szre1.transpose() + szre2.transpose()))))
J[((((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), (((nspins*6) + 6) + 1)..((nspins*8) + 6))] = multiply(np, (gmatsmsprereal.transpose() + gmatsmspimreal.transpose()), ((aszim1.transpose() + aszim2.transpose()) - ((2*aim)*(szre1.transpose() + szre2.transpose()))))
J[((((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), ((nspins*8) + 6)..((nspins*10) + 6))] = multiply(np, gstiled, ((((-2*are)*(smre1.transpose() + smre2.transpose())) - ((2*aim)*(smim1.transpose() + smim2.transpose()))) + (adagsmre1.transpose() + adagsmre2.transpose())))
fill_diagonal(np, J[((((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), (((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))))], (-2*gamma));
fill_diagonal(np, J[((((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), ((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))))], (deltas[indexj2] - deltas[indexi2]));
J[((((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), (((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))] = (are*(gmatjacre + gmatjacim))
J[((((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), ((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))] = (aim*(gmatjacre + gmatjacim))
J[(((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), 0)] = ((gsreal[indexj2]*(((2*smim[indexi2])*szre[indexj2]) - szsmim[indexred])) + (gsreal[indexi2]*(((-2*smim[indexj2])*szre[indexi2]) + szsmim[indexswitchred])))
J[(((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), 1)] = ((gsreal[indexj2]*(((-2*smre[indexi2])*szre[indexj2]) + szsmre[indexred])) + (gsreal[indexi2]*(((2*smre[indexj2])*szre[indexi2]) - szsmre[indexswitchred])))
J[(((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), 2..)] = 0.0
J[(((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), (((nspins*2) + 6) + 1)..((nspins*4) + 6))] = multiply(np, (gmatsmsprereal.transpose() - gmatsmspimreal.transpose()), (szre1.transpose() + szre2.transpose()))
J[(((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), ((nspins*4) + 6)..((nspins*6) + 6))] = multiply(np, gstiled, (-smim1.transpose() + smim2.transpose()))
J[(((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), (((nspins*4) + 6) + 1)..((nspins*6) + 6))] = multiply(np, gstiled, (smre1.transpose() - smre2.transpose()))
J[(((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), ((nspins*6) + 6)..((nspins*8) + 6))] = multiply(np, (gmatsmsprereal.transpose() + gmatsmspimreal.transpose()), ((-aszim1.transpose() + aszim2.transpose()) + ((2*aim)*(szre1.transpose() - szre2.transpose()))))
J[(((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), (((nspins*6) + 6) + 1)..((nspins*8) + 6))] = multiply(np, (gmatsmsprereal.transpose() + gmatsmspimreal.transpose()), ((aszre1.transpose() - aszre2.transpose()) - ((2*are)*(szre1.transpose() - szre2.transpose()))))
J[(((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), ((nspins*8) + 6)..((nspins*10) + 6))] = multiply(np, gstiled, ((((2*are)*(smim1.transpose() - smim2.transpose())) - ((2*aim)*(smre1.transpose() - smre2.transpose()))) - (adagsmim1.transpose() - adagsmim2.transpose())))
fill_diagonal(np, J[(((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), (((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))))], (-(deltas[indexj2]) + deltas[indexi2]));
fill_diagonal(np, J[(((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), ((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))))], (-2*gamma));
J[(((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), (((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))] = (aim*(gmatjacre - gmatjacim))
J[(((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))), ((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))] = (-(are)*(gmatjacre - gmatjacim))
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), 0)] = ((gsreal[indexj]*((((8*smre[indexi])*smre[indexj]) - (2*smsmre[indexdouble])) - (2*smspre[indexdouble]))) + (gsreal[indexi]*(((-2*szre[indexj])*szre[indexi]) + szszre[indexdouble])))
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), 1)] = (gsreal[indexj]*((((8*smim[indexj])*smre[indexi]) - (2*smsmim[indexdouble])) - (2*(smspim1 + spsmim1))))
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), 2..)] = 0.0
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), 6..((nspins*2) + 6))] = (-2*smref)
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), 7..((nspins*2) + 6))] = (-2*smimf)
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), ((nspins*2) + 6)..((nspins*4) + 6))] = ((-4*smref2) - (2*smref))
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), (((nspins*2) + 6) + 1)..((nspins*4) + 6))] = (2*smimf)
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), ((nspins*4) + 6)..((nspins*6) + 6))] = (szref2 + szref3)
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), ((nspins*6) + 6)..((nspins*8) + 6))] = (((((((-2*adagsmref2) - (2*asmref2)) + ((8*are)*smref2)) - (4*adagsmref)) + ((8*aim)*smimf)) + ((8*are)*smref)) - (gparmat2*gpar))
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), (((nspins*6) + 6) + 1)..((nspins*8) + 6))] = (((2*adagsmimf2) - (2*asmimf2)) + ((8*aim)*smref2))
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), ((nspins*8) + 6)..((nspins*10) + 6))] = (((aszref2 - ((2*are)*szref)) + aszref) - ((2*are)*szref3))
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), ((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))))] = ((-2*are)*gstiled3)
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), (((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))))] = ((-2*aim)*gstiled3)
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), (((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))))] = ((-2*are)*gstiled3)
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), ((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))))] = ((-2*aim)*gstiledswitch)
fill_diagonal(np, J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), (((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))], (-(gpar) - gamma));
fill_diagonal(np, J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), ((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))], deltas[indexi]);
J[((((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), (((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))))] = (are*(transpose(gmatjacre2) + transpose(gmatjacim2)))
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), 0)] = (gsreal[indexj]*((((8*smre[indexj])*smim[indexi]) - (2*smsmim[indexdouble])) + (2*(smspim1 + spsmim1))))
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), 1)] = ((gsreal[indexj]*((((8*smim[indexi])*smim[indexj]) + (2*smsmre[indexdouble])) - (2*smspre[indexdouble]))) + (gsreal[indexi]*(((-2*szre[indexj])*szre[indexi]) + szszre[indexdouble])))
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), 2..)] = 0.0
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), 6..((nspins*2) + 6))] = (2*smimf)
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), 7..((nspins*2) + 6))] = (-2*smref)
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), ((nspins*2) + 6)..((nspins*4) + 6))] = ((-4*smimf2) - (2*smimf))
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), (((nspins*2) + 6) + 1)..((nspins*4) + 6))] = (-2*smref)
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), (((nspins*4) + 6) + 1)..((nspins*6) + 6))] = (szref2 + szref3)
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), ((nspins*6) + 6)..((nspins*8) + 6))] = (((-2*adagsmimf2) - (2*asmimf2)) + ((8*are)*smimf2))
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), (((nspins*6) + 6) + 1)..((nspins*8) + 6))] = (((((((-2*adagsmref2) + (2*asmref2)) + ((8*aim)*smimf2)) - (4*adagsmref)) + ((8*aim)*smimf)) + ((8*are)*smref)) - (gparmat2*gpar))
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), ((nspins*8) + 6)..((nspins*10) + 6))] = (((aszimf2 - ((2*aim)*szref)) + aszimf) - ((2*aim)*szref3))
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), ((nspins*10) + 6)..(((nspins*10) + 6) + (nspins*(nspins - 1))))] = ((2*aim)*gstiled3)
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), (((nspins*10) + 6) + 1)..(((nspins*10) + 6) + (nspins*(nspins - 1))))] = ((-2*are)*gstiled3)
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), (((nspins*10) + 6) + (nspins*(nspins - 1)))..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))))] = ((-2*aim)*gstiled3)
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), ((((nspins*10) + 6) + (nspins*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((2*nspins)*(nspins - 1))))] = ((2*are)*gstiledswitch)
fill_diagonal(np, J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), (((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))], -(deltas[indexi]));
fill_diagonal(np, J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), ((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))], (-(gpar) - gamma));
J[(((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))), (((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))))] = (aim*(transpose(gmatjacre2) + transpose(gmatjacim2)))
J[((((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))), 0)] = ((gsreal[indexj2]*(((8*smre[indexj2])*szre[indexi2]) - (4*szsmre[indexswitchred]))) + (gsreal[indexi2]*(((8*smre[indexi2])*szre[indexj2]) - (4*szsmre[indexred]))))
J[((((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))), 1)] = ((gsreal[indexj2]*(((8*smim[indexj2])*szre[indexi2]) - (4*szsmim[indexswitchred]))) + (gsreal[indexi2]*(((8*smim[indexi2])*szre[indexj2]) - (4*szsmim[indexred]))))
J[((((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))), 2..)] = 0.0
J[((((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))), ((nspins*2) + 6)..((nspins*4) + 6))] = multiply(np, gstiled, (-4*(szre1.transpose() + szre2.transpose())))
J[((((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))), ((nspins*4) + 6)..((nspins*6) + 6))] = multiply(np, (gmatsmsprereal.transpose() + gmatsmspimreal.transpose()), (-4*(smre1.transpose() + smre2.transpose())))
J[((((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))), (((nspins*4) + 6) + 1)..((nspins*6) + 6))] = multiply(np, (gmatsmsprereal.transpose() + gmatsmspimreal.transpose()), (-4*(smim1.transpose() + smim2.transpose())))
J[((((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))), ((nspins*6) + 6)..((nspins*8) + 6))] = multiply(np, gstiled, ((-4*(aszre1.transpose() + aszre2.transpose())) + ((8*are)*(szre1.transpose() + szre2.transpose()))))
J[((((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))), (((nspins*6) + 6) + 1)..((nspins*8) + 6))] = multiply(np, gstiled, ((-4*(aszim1.transpose() + aszim2.transpose())) + ((8*aim)*(szre1.transpose() + szre2.transpose()))))
fill_diagonal(np, J[((((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))), ((nspins*8) + 6)..((nspins*10) + 6))], -(gpar));
J[((((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))), ((nspins*8) + 6)..((nspins*10) + 6))] = (multiply(np, (gmatsmsprereal.transpose() + gmatsmspimreal.transpose()), (((-4*(adagsmre1.transpose() + adagsmre2.transpose())) + ((8*aim)*(smim1.transpose() + smim2.transpose()))) + ((8*are)*(smre1.transpose() + smre2.transpose())))) - (gparmat*gpar))
J[((((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))), (((nspins*10) + 6) + ((2*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))] = ((-4*are)*(gmatjacre2 + gmatjacim2))
J[((((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))), ((((nspins*10) + 6) + ((2*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((4*nspins)*(nspins - 1))))] = ((-4*aim)*(gmatjacre2 + gmatjacim2))
fill_diagonal(np, J[((((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))), (((nspins*10) + 6) + ((4*nspins)*(nspins - 1)))..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))))], (-2*gpar));
J[(((((nspins*10) + 6) + ((4*nspins)*(nspins - 1))) + 1)..(((nspins*10) + 6) + ((5*nspins)*(nspins - 1))), ..)] = 0.0
return J
end


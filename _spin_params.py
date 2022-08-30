import numpy as np


#cavity parameters
quali = 1363.8800995204592
fr = 3099.7889160238033
kappa = np.pi*fr/quali


#hybrid parameters
gcoll = 28.290549673769124
#CAREFUL HERE: a positive deltac means that the cavity frequency is larger(!!) than the probe. so the x axis in a frequency plot are reversed (at least intuitively)
deltac = 0*2*np.pi #detuning drive to cavity wc-wp
delta = 0*2*np.pi #detuning cavity to spin central frequency ws-wc

#spin parameters
q = 1.31
gperp = 2*np.pi*0.3 #HWHM
gpar = 2*np.pi*.0001#0.001 #HWHM
gammaq = 58.93042792447484
spin_dist_width = 60 #MHz how far frequencies should be sampled

# numerical parameters
nbins = 1301#701#5000#20
g_number = 1 #how many bins for inhomogeneous coupling
nreal = 3.6e13 #estimated real number of spins

#fano parameters
offset = -0.0009444275261222643
phi = 8.276302849184598
phi_offset = 0.7740617467429286
dt = 0.05624458203702167


#reference trace folder

folder="/mnt/samba2/experiments/Quantum InterConnect/Cryogenics/Dilution Fridge/Measurements/20220811_DCR_n-diamond_40/reftraces/"
reference_trace = "referenceTrace_1.mat"
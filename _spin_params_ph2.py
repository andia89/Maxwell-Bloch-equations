import numpy as np


#cavity parameters
quali = 12878.399027616591
fr = 3714.9216773600947
kappa = np.pi*fr/quali


#hybrid parameters
gcoll = 8.543498083462055
#CAREFUL HERE: a positive deltac means that the cavity frequency is larger(!!) than the probe. so the x axis in a frequency plot are reversed (at least intuitively)
deltac = 0*2*np.pi #detuning drive to cavity wc-wp
delta = 0*2*np.pi #detuning cavity to spin central frequency ws-wc

#spin parameters
q = 1.61
gperp = 2*np.pi*0.3 #HWHM
gpar = 2*np.pi*.0001#0.001 #HWHM
gammaq = 16.446376077520007
spin_dist_width = 60 #MHz how far frequencies should be sampled

# numerical parameters
nbins = 1301#701#5000#20
g_number = 1 #how many bins for inhomogeneous coupling
nreal = 3.6e13 #estimated real number of spins

#fano parameters
offset = 0.0018425645775328413
phi = 1.0141624824749207
phi_offset = -0.5297840676511922
dt = -0.03802483171150982
ampl = -0.0010838703567443142

#reference trace folder

folder="/mnt/samba2/experiments/Quantum InterConnect/Cryogenics/ParaHydrogen/Measurements/Measurement_Run181_20220623/Mag field scan/"
reference_trace = "trace_300_5.pkl"
high_power_trace = "trace_off_res.pkl"
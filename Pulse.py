import numpy as np

"""
Note: all pulses in rotating frame. Representations are modulations of (arbitrary) carrier.
"""

class Pulse:
    def __init__(self):
        pass
    # todo: make this an abstract class

    def get_amp_t_q(self):
        pass

    def get_amp_t_i(self):
        pass


class AmpPhasePulse:
    """
    Pulse represented by amplitude and phase as one complex number.
    """
    def __init__(self):
        self.amps_c = None      # complex amplitudes
        self.t = None
        self.dt = None

    def to_iq(self):
        # todo: double check. Correct?
        i = np.real(self.amps_c) * np.cos(np.imag(self.amps_c))
        q = np.real(self.amps_c) * np.sin(np.imag(self.amps_c))

        return (i, q)

    def set_t_axis(self, t):
        self.t = t
        self.dt = t[1] - t[0]


class FourierPulse:
    """
    Pulse represented by I and Q Fourier components
    """
    def __init__(self, w_base, n_fcomps, a_max=10):
        self.w_base = w_base
        self.n_fcomps = n_fcomps
        self.a_max = a_max
        self.f_comps = None # [n_fcomps x 2]: [[amplitude_i,...], [amplitude_q,...]]

    def get_fcomps_i(self):
        return self.f_comps[:, 0]

    def get_fcomps_q(self):
        return self.f_comps[:, 1]

    def get_amp_t_q(self, t):
        return self._fcomps_2_t(t, self.get_fcomps_q())

    def get_amp_t_i(self, t):
        return self._fcomps_2_t(t, self.get_fcomps_i())

    def _fcomps_2_t(self, t, fcomps):
        """
        Convert Fourier components to time domain
        :param t:  Time axis (us)
        :param fcomp: Fourier component of I or Q.
        :return:
        """
        pulse_t = 0
        for i in range(len(fcomps)):
            pulse_t = pulse_t + (fcomps[i]) * np.sin((i + 1) * self.w_base * t)
        return pulse_t

    def pulse_t_from_f(self, t):

        pulse_i = self.get_amp_t_i(t)
        pulse_q = self.get_amp_t_q(t)

        return (pulse_i, pulse_q)

    def init_random(self):
        self.f_comps = np.random.uniform(-self.a_max, self.a_max, (self.n_fcomps, 2))
        self.f_comps = self.clip_amps(self.f_comps)

    def clip_amps(self, f_comps):
        return np.clip(f_comps , -self.a_max, self.a_max)


if __name__ == "__main__":
    import scipy.io
    import matplotlib.pyplot as plt

    path = 'C:\\Users\\timo.joas\\OneDrive\\_Promotion\\Software\\Easyspin CompensatedPulse\\output'
    file = '180612_compensated_pi_200ns.mat'
    pulse_raw = scipy.io.loadmat(path + '\\' + file)
    t = pulse_raw['t_comp_in']
    y_complex = pulse_raw['y_comp_in']

    pulse = AmpPhasePulse()
    pulse.t = t[0,:]
    pulse.amps_c = y_complex[0,:]

    plt.plot(pulse.t, np.real(pulse.amps_c))
    plt.plot(pulse.t, np.imag(pulse.amps_c))

    pulse_i, pulse_q = pulse.to_iq()
    plt.plot(pulse.t, pulse_i)
    plt.plot(pulse.t, pulse_q)

    plt.show()
    exit()
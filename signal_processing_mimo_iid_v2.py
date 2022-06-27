import math as mt
import numpy as np


class MonteCarlo():
    def __init__(self, MC_RUNS, N_RX, N_TX):
        self.mc_runs = MC_RUNS
        self.n_rx = N_RX
        self.n_tx = N_TX

        # Initialize variables
        self.rx_signal = np.zeros([MC_RUNS, N_RX], dtype=complex)

    def signal_process(self, M, symb_QAM, std):
        # Choose which class of vector signal to transmit. Classes occurrences
        # are random (i.i.d) and equiprobable
        symb_idx = np.random.randint(M, size=[self.mc_runs, self.n_tx])

        # Initialize complex Gaussian channel coefficients (Rayleigh fading)
        # TODO: divide by N_RX to normilize to HTH = I?
        H = (1 / mt.sqrt(2 * self.n_rx)) * (np.random.randn(self.mc_runs, self.n_rx, self.n_tx) +
                                            1j * np.random.randn(self.mc_runs, self.n_rx, self.n_tx))

        # ro = 0.99
        # ro = np.sqrt(1 - (1 + (1 / std[:, np.newaxis, np.newaxis]**2))**(-1))
        # Eps = (1 / mt.sqrt(2 * self.n_rx)) * (np.random.randn(self.mc_runs, self.n_rx, self.n_tx) +
        #                           1j * np.random.randn(self.mc_runs, self.n_rx, self.n_tx))
        
        # H_hat = ro * H + np.sqrt(1 - ro**2) * Eps

        
        # Initialize complex AWG noise at the receiver
        noise = (std[:, np.newaxis] / mt.sqrt(2)) * (np.random.randn(self.mc_runs, self.n_rx) +
                                                      1j * np.random.randn(self.mc_runs, self.n_rx))

        # Corrupt the received signal with channel impairments
        self.rx_signal = np.squeeze(H @ symb_QAM[symb_idx, np.newaxis]) + noise

        return self.rx_signal, symb_idx, H

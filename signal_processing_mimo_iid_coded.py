import math as mt
import numpy as np


class MonteCarlo_coded():
    def __init__(self, MC_RUNS, N_RX, N_TX, M, n, k):
        self.mc_runs = MC_RUNS
        self.n_rx = N_RX
        self.n_tx = N_TX
        self.n, self.k = n, k

        # Initialize variables
        self.n_symb = n // int(mt.sqrt(M))
        self.channel_input = np.zeros([N_TX, MC_RUNS, self.n_symb], dtype=complex)

    def signal_process(self, encoder, modulator, M, std):
        mess = np.random.randint(0, 2, size=(self.n_tx, self.mc_runs, self.k))
        
        for i in range(self.n_tx):
            self.channel_input[i] = \
                modulator.modulate_symbols(encoder.encode_messages(mess[i]), m=M) * \
                    mt.sqrt(3 / (2 * (M - 1)))

        # Initialize complex Gaussian channel coefficients (Rayleigh fading)
        H = (1 / mt.sqrt(2 * self.n_rx)) * \
            (np.random.randn(self.mc_runs, self.n_rx, self.n_tx) +
             1j * np.random.randn(self.mc_runs, self.n_rx, self.n_tx))

        # Initialize complex AWG noise at the receiver
        noise = (std[:, np.newaxis] / mt.sqrt(2)) * \
            (np.random.randn(self.mc_runs, self.n_rx, self.n_symb) +
             1j * np.random.randn(self.mc_runs, self.n_rx, self.n_symb))

        # Corrupt the received signal with channel impairments
        rx_signal = H @ self.channel_input.transpose([1, 0, 2]) + noise

        return rx_signal, mess, H

""" Deep Unfolded Probability Association Detector (DU-PDA) for MIMO Systems

    Notes
    -----
    Please, refer to [R1] for all citations made in this code, namely of
    sections, equations, and so forth.

    Reference
    ----------

    .. [R1] Pedro Henrique Carneiro Souza, Luciano Leonel Mendes. Low-complexity
            Deep Unfolded Neural Network Receiver for MIMO Systems Based on the
            Probability Data Association Detector, 24 March 2022, PREPRINT
            (Version 1) available at Research Square
            [https://doi.org/10.21203/rs.3.rs-1439479/v1]

    Created on Fri May 28/2021
    Last rev. on Wed Jun 22
    © 2022 Pedro H. C. de Souza
"""
import math as mt
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import special
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from digcommpy import messages, encoders, decoders, modulators
from signal_processing_mimo_iid_v2 import MonteCarlo
from signal_processing_mimo_iid_coded import MonteCarlo_coded
start_time = time.time()


class SubLayer(layers.Layer):
    """ The customized deep unfolded detector uses the Tensorflow Functional API

        Notes
        -----
        See 'Making new Layers and Models via subclassing' on https://www.tensorflow.org/
    """
    def __init__(self, units_TX, units_RX, coords, sic_l, sub_layer):
        super().__init__()
        self.dn_t = 2 * units_TX  # number of transmitting atennas (Nt) (real valued notation)
        self.dn_r = 2 * units_RX  # number of receiving atennas (Nr) (real valued notation)
        # Coordinates from the normalized M-QAM constellation
        self.q_coord = tf.convert_to_tensor(coords[:, tf.newaxis], dtype=tf.float32)
        self.s_sic = sic_l  # detection order of received symbols (discontinued)
        self.ell = sub_layer  # layer index being accessed

    def build(self, input_shape):
        # Learnable weight initialization
        self.theta_l = self.add_weight(
            shape=(1,),  # it is a scalar
            # initializer=keras.initializers.GlorotUniform(seed=42),
            trainable=True
        )

    def call(self, x_l, P_l, r_l, H_l, std_l):
        def cov(A_l, res_z):
            # Calculations of the empirical MSE estimator of the covariance matrix

            # Comment the following lines to implement the simplified DU-PDA
            # -------------------------------------------------------------------------------------
            # FH_l = tf.expand_dims(tf.linalg.trace(tf.matmul(H_l, H_l, transpose_a=True)), axis=1)

            # See Eq. (24)
            # v2_l = tf.divide((tf.reduce_sum(tf.square(res_z), axis=1, keepdims=True) -
            #                   self.dn_r * tf.square(std_l) / 2.), FH_l)
            # v2_l = tf.maximum(v2_l, 1e-9)
            # v2_l = tf.expand_dims(v2_l, axis=2)

            # B_l = tf.eye(self.dn_t, batch_shape=[H_l.shape[0]]) - tf.matmul(A_l, H_l)

            # See Eq. (23)
            # cov_l = (1 / self.dn_t) *\
            #     (tf.reshape(tf.linalg.trace(tf.matmul(B_l, B_l, transpose_a=True)), [-1, 1, 1]) *
            #       v2_l + tf.square(tf.reshape(std_l, [-1, 1, 1])) / 2 *
            #       tf.reshape(tf.linalg.trace(tf.matmul(A_l, A_l, transpose_a=True)), [-1, 1, 1]))
            # -------------------------------------------------------------------------------------
            # The variable 'cov_l' must be returned if the above lines are not commented

            return tf.square(tf.reshape(std_l, [-1, 1, 1])) / 2

        # Calculation of posterior probabilities of the PDA detector
        def alpha(P_j, z_l, A_l, res):
            v_l = tf.matmul(P_j, self.q_coord)
            Omega = (tf.matmul(P_j, self.q_coord**2) - v_l**2) + cov(A_l, res)  # see Eq. (22)

            u_l = z_l - tf.squeeze(v_l)

            # See Eq. (20)
            alpha_l = tf.math.reduce_sum(
                tf.linalg.matrix_transpose((u_l[:, :, tf.newaxis] -
                                            0.5 * e_i * tf.squeeze(self.q_coord)) *
                                            (e_i * tf.squeeze(self.q_coord) / Omega)), 2)

            return alpha_l - tf.math.reduce_max(alpha_l, 1, keepdims=True)

        res = r_l - tf.squeeze(tf.matmul(H_l, x_l[:, :, tf.newaxis]))
        A_l = tf.linalg.matrix_transpose(self.theta_l * H_l)
        z_l = x_l + tf.squeeze(tf.matmul(A_l, res[:, :, tf.newaxis]))  # see Eq. (19)

        e_i = tf.zeros_like(P_l)

        # Initialize posterior probabilities matrix
        P_l = tf.where(self.s_sic[self.ell], tf.zeros_like(P_l), P_l)

        # Build up the vector [0...x...0]^T
        e_i = tf.where(self.s_sic[self.ell], tf.ones_like(e_i), e_i)

        p_l = tf.nn.softmax(alpha(P_l, z_l, A_l, res), 1)  # see Eq. (19)

        # See Eq. (26)
        z_l = tf.where(self.s_sic[self.ell, :, :, 0], tf.matmul(p_l, self.q_coord), z_l)

        # Lines commented below implement the hard combining version of Eq. (26) (discontinued)
        # z_l = tf.where(self.s_sic[self.ell, :, :, 0],
        #                 tf.gather(self.q_coord, tf.math.argmax(p_l, 1)), z_l)

        # Assign values for the posterior probabilities matrix accordingly
        P_l = tf.where(self.s_sic[self.ell], tf.tile(p_l[:, tf.newaxis, :],
                                                      [1, self.dn_t, 1]), P_l)

        return P_l, z_l


# Initialize system model parameters
N_POINTS = 11  # number of BER curve points
M = 4  # order of the sqaure M-QAM modulation
L = int(mt.sqrt(M))
N_RX = 8  # number of atennas (Nr) at the receiver
N_TX = 4  # number of atennas (Nt) at the transmitter
DN_TX = 2 * N_TX  # number of transmitting atennas for the real valued notation
n, kmess = 256, 128  # number of bits transmitted (nb) and codeword size
C_RATE = kmess / n  # computation of the code rate
n_symb = n // int(L)   # number of symbols to be transmitted (nb/log2(M))
snr = np.linspace(-2, 15, N_POINTS)  # definition of the SNR range

# Initialize hyperparameters of the deep unfolded neural network
N_TRAIN = 10**5  # training set size
N_LAYER = 2  # number of layers (multiplier of Nt)
SOLVER = 'Adam'
INIT_ETA = 10**-3  # initial learning rate

MAX_EPOCHS = 200
BATCH_SIZE = 500  # mini-batch set size
N_ITER = N_TRAIN // BATCH_SIZE  # number of iterations at each epoch
TOL = 10**-4  # reference value used for convergence
N_ITER_NO_CHANGE = 10  # reference value used for aborting training

# Generation of M possible bits combinations
all_messages = messages.generate_data(L, number=None, binary=True)

# Calculation of the normalized M-QAM constellation
# E_zero = 3 / (2 * (M[i] - 1) * N_TX)
E_zero = 3 / (2 * (M - 1))
j_idx = np.arange(1, L + 1)
x_coords = (2 * j_idx - L - 1) * np.sqrt(E_zero)
x_QAM = np.zeros(M, complex)
x_PDA = np.zeros([M, 2 * L])
k = 0
for x in range(L):
    for y in range(L):
        x_QAM[k] = x_coords[x] + 1j * x_coords[y]
        x_PDA[k, x] =  1
        x_PDA[k, L + y] =  1
        k += 1

# Configuration of parameters for simulating baseband MIMO signal transmissions
params = MonteCarlo(BATCH_SIZE, N_RX, N_TX)

# Initialize DU-PDA parameters for architecture compilation
P_zero = keras.Input(shape=(DN_TX, L), batch_size=(BATCH_SIZE), name='P')
x_zero = keras.Input(shape=(DN_TX), batch_size=(BATCH_SIZE), name='x_tilde')
r = keras.Input(shape=(2 * N_RX), batch_size=(BATCH_SIZE), name='r')
H = keras.Input(shape=(2 * N_RX, DN_TX), batch_size=(BATCH_SIZE), name='H')
std = keras.Input(shape=(1,), name='sigma')

# Define the detection order of received symbols (discontinued)
sic = np.zeros([DN_TX, BATCH_SIZE, DN_TX, L], dtype=bool)
for j in range(DN_TX):
    sic[j, :, j] = True

# Compilate (generate) the DU-PDA architecture
P_ell = P_zero
x_ell = x_zero
outputs = []
for L_primary in range(N_LAYER):
    for ell in range(DN_TX):
        P_ell, x_ell = SubLayer(N_TX, N_RX, x_coords, sic, ell)(x_ell, P_ell, r, H, std)

    outputs.append(P_ell)

# Instantiate Keras and initialize the compiled DU-PDA architecture
model = keras.Model(inputs=(x_zero, P_zero, r, H, std), outputs=outputs)

optimizer = keras.optimizers.Adam(learning_rate=INIT_ETA)  # instantiate the solver

# Instantiate the loss function (see Eq. (27))
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

#--------------------------------TRAINING STAGE--------------------------------
# See 'Writing a training loop from scratch' on https://www.tensorflow.org/
P_zero_batch = np.ones([BATCH_SIZE, DN_TX, L]) / L
x_zero_batch = np.zeros([BATCH_SIZE, DN_TX])
best_loss = np.inf
NO_IMP_COUNT = 0
try:
    for epoch in range(MAX_EPOCHS):
        print("\nEpoch #%d" % (epoch,))

        for step in range(N_ITER):
            # SNR values are drawn from a uniform distribution (see §4.1)
            snr_uniform = np.random.uniform(snr.min(), snr.max(), BATCH_SIZE)

            # Compute the system signal-to-noise ratio (SNR) per bit
            std_batch = np.sqrt(N_TX / ((10**(snr_uniform / 10)) * np.log2(M) * N_RX))

            # Simulate the baseband MIMO signal transmissions for training
            rx_signal, symb_idx, ch_coef = params.signal_process(M, x_QAM, std_batch)

            # Define the training ground truth
            label_batch = np.block([[x_PDA[symb_idx][:, :, 0:L]],
                                    [x_PDA[symb_idx][:, :, L:(2 * L)]]])

            # Convert complex vectors and matrices to the real valued notation
            r_batch = np.block([np.real(rx_signal), np.imag(rx_signal)])
            H_batch = np.block([[np.real(ch_coef), -np.imag(ch_coef)],
                                [np.imag(ch_coef), np.real(ch_coef)]])

            # Apply automatic differentiation and train the DU-PDA
            with tf.GradientTape() as tape:
                out = model((x_zero_batch, P_zero_batch, r_batch,
                              H_batch, std_batch), training=True)
                loss_value = loss_fn(tf.tile(label_batch[tf.newaxis],
                                              [N_LAYER, 1, 1, 1]), out) / L

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            sys.stdout.write('\r')
            sys.stdout.write("%.1f" % (100 / N_ITER * (step + 1)))
            sys.stdout.flush()

        sys.stdout.write("\n Loss: {:.4f}".format(loss_value))

        # Abort training if a given set of codditions are met
        if loss_value > best_loss - TOL:
            NO_IMP_COUNT += 1
            if NO_IMP_COUNT >= N_ITER_NO_CHANGE:
                sys.stdout.write("\n Loss: {:.4f}".format(loss_value))
                sys.stdout.flush()
                break
        else:
            NO_IMP_COUNT = 0
        if loss_value < best_loss:
            best_loss = loss_value

        sys.stdout.write("\n No improvement count: {:.1f}".format(NO_IMP_COUNT))
        sys.stdout.write("\n Best Loss: {:.4f}".format(best_loss))
        sys.stdout.flush()

except KeyboardInterrupt:
    no_improvement_count = np.inf
#--------------------------------TRAINING STAGE--------------------------------

# Reconfiguration of parameters for simulating baseband MIMO signal transmissions
params = MonteCarlo_coded(BATCH_SIZE, N_RX, N_TX, M, n, kmess)

#--------------------------DETECTION (TEST) STAGE------------------------------
# Generation of all M constellation symbols according to Gray mapping
modulator = modulators.QamModulator()
all_mod_symbols = np.squeeze(modulator.modulate_symbols(all_messages, M))

idx = np.arange(M)
est_bits = np.zeros([N_TX, BATCH_SIZE, n_symb, L])
est_mess = np.zeros([N_TX, BATCH_SIZE, kmess])
pe = np.zeros([N_POINTS])

# Compute Monte Carlo iterations for each curve point
for j in range(N_POINTS):
    # Generate the progress bar for evaluation purposes. It does not have
    # any impact on the results
    sys.stdout.write('\r')
    sys.stdout.write("[{:{}}] {:.1f}%  ".format("=" * j, N_POINTS - 1,
                                                (100 / (N_POINTS - 1) * j)))
    sys.stdout.flush()

    lin_snr = 10**(snr[j] / 10)

    # Definition of the AWGN standard deviation. Note here that the system
    # SNR per bit is considered, where it is assumed that the channel matrix is
    # normalized by 1/sqrt(Nr)
    std = np.array([np.sqrt(N_TX / (lin_snr * C_RATE * np.log2(M) * N_RX))])

    # See Besser, K.: Digcommpy 0.9. https://pypi.org/project/digcommpy/
    # For the sake of simplicity, we assume that the reference bit error
    # probability used by the Polar encoder is given by the Gaussian tail area
    pflip = 0.5 - 0.5 * special.erf(np.sqrt(C_RATE * lin_snr))
    encoder = encoders.PolarEncoder(n, kmess, "BSC", pflip)
    decoder = decoders.PolarDecoder(n, kmess, "BSC", pflip)

    # Clear cumulative error counter
    ERR = 0
    MC_RUNS = 0
    while ERR < 10**3:
        # Simulate the MIMO transmission of M-QAM symbols over NtNr slow-flat
        # Rayleigh channels and contaminate the received signal with AWGN
        rx_signal, mess, ch_coef = params.signal_process(encoder, modulator, M, std)

        # Convert matrices and vectors of complex numbers to concatenated real values
        r = np.block([[np.real(rx_signal)], [np.imag(rx_signal)]])
        H = np.block([[np.real(ch_coef), -np.imag(ch_coef)],
                      [np.imag(ch_coef), np.real(ch_coef)]])

        for cbit in range(n_symb):
            # Symbol detection (see Line 15 of Algorithm 2)
            P_L = model((x_zero_batch, P_zero_batch,
                         r[:, :, cbit], H, std), training=False)[N_LAYER - 1]

            # Final decision in favor of the symbol coordinate associated with the
            # higher probability value by the DU-PDA detector
            x_QAM_hat = np.round((x_coords[np.argmax(P_L, 2)][:, 0:N_TX] + \
                         1j * x_coords[np.argmax(P_L, 2)][:, N_TX:DN_TX]) / np.sqrt(E_zero))

            # Convert decided symbols into bits
            for tx in range(N_TX):
                idx_bool = x_QAM_hat[:, tx, np.newaxis] == all_mod_symbols
                est_bits[tx, :, cbit] = all_messages[idx_bool @ idx]

        # Decoding via the Polar decoder
        est_codewords = est_bits.reshape([N_TX, BATCH_SIZE, n])
        for tx in range(N_TX):
            est_mess[tx] = decoder.decode_messages(est_codewords[tx])

        ERR += np.sum(mess != est_mess)
        MC_RUNS += BATCH_SIZE * N_TX * kmess

    # Estimate the probability of error
    pe[j] = ERR / MC_RUNS
#--------------------------DETECTION (TEST) STAGE------------------------------

# Estimate runtime for evaluation purposes
print("\n %f seconds" % (time.time() - start_time))

# Plot results
fig, ax = plt.subplots()
plt.plot(snr, pe, '--sr', linewidth=1.25, fillstyle='none', markersize=3.75)

# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(), fontsize='x-small')
# plt.legend()

props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
TEXTSTR = '\n'.join(('Rayleigh (slow-flat)',
                      'Polar: (%.i,%.i)' % (n, kmess),
                      'Ideal'))
TEXTSTR_ = '\n'.join(('DU-PDA parameters:',
                      r'$N_{\mathcal{S}_{TR}} = %.E$' % N_TRAIN,
                      r'$L = %.i$' % N_LAYER,
                      r'$\eta_{init} = %.E$' % INIT_ETA,
                      SOLVER))
plt.text(0.025, 0.025, TEXTSTR, transform=ax.transAxes, fontsize='x-small',
          verticalalignment='bottom', bbox=props)
plt.text(0.025, 0.575, TEXTSTR_, transform=ax.transAxes, fontsize='x-small',
          verticalalignment='bottom', bbox=props)

plt.axis([snr.min(), snr.max(), 10**-5, 1])
plt.semilogy()
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title(r'MIMO %.ix%.i, %.i-QAM ' % (N_TX, N_RX, M))
plt.grid(which='major', linestyle='--')
plt.grid(which='minor', linestyle='--', linewidth=0.5)

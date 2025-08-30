import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, fft, fftfreq
from scipy.signal import spectrogram, welch

## argparse define line

parser = argparse.ArgumentParser(description="signal detect with plot")

parser.add_argument('sig', help = 'Signal file name')
parser.add_argument('fs', help = 'Sampling rate')
parser.add_argument('-spec', default=False, help = 'Show spectrogram')
parser.add_argument('-ks', default=False, help = 'Show Am Detect')
parser.add_argument('-time_domain', default=False, help = "show energy plot")


args = parser.parse_args()
sig = args.sig
fs = args.fs
spect = args.spect
ks = args.ks
time_domain = args.time_domain




class Signal:
    def __init__(self, sig, fs, dt):
        self.name = sig
        sig = np.fromfile(sig, dtype=dt)

        if dt[-1:] == '1':
            self.sig = np.float32(sig) / 127
        elif dt[-1:] == '2':
            self.sig = np.float32(sig) / 32768

        self.fs = fs

    def spectrogram(self, IQ = False, nperseg = 2*11, noverlab = 2*10):
        if IQ:
            iq = self.sig.reshape(-1, 2)
            z = iq[:,0] + 1j*iq[:,1]    # I + jQ

        f, t, s = spectrogram(self.sig, self.fs, nperseg=nperseg, noverlab=noverlab)

        if IQ:
            s = np.fft.fftshift(s, axes=0) # 복소평면 데이터 셋으로 쉬프트
            f = np.fft.fftshift(f)
            plt.ylim(self.fs/8 * (-1), self.fs/8)
        else:
            plt.ylim(0, self.fs/2)
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(t, f, 20 * np.log10(s + 1e-12), shading='gouraud', cmap='binary')
        plt.colorbar(label='dB:')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequncy [Hz]')
        plt.title(f'{self.name} Spectrogram')
        plt.show()

    def time_domain(self, nperseg = 2*11, noverlab = 2*10):
        t = np.arange(len(self.sig))
        plt.figure(figsize=(9, 3))
        plt.plot(t, self.sig, linewidth=0.6)
        plt.xlabel("Time [s]");
        plt.ylabel("Amplitude")
        plt.title("2-FSK (time domain, full 2 s)")
        plt.tight_layout()
        plt.show()

    def ks_detector(self):
        # ---------- Utilities ----------
        def rrc_filter(beta, span, sps):
            """
            Root-Raised-Cosine (RRC) filter taps.
            beta: roll-off (0~1)
            span: filter length in symbols (e.g., 8~12)
            sps: samples per symbol (oversampling factor)
            """
            N = span * sps
            t = (np.arange(-N / 2, N / 2 + 1) / sps).astype(float)  # in symbol times
            h = np.zeros_like(t, dtype=float)
            for i, ti in enumerate(t):
                if np.isclose(ti, 0.0):
                    h[i] = 1.0 - beta + 4 * beta / np.pi
                elif beta != 0 and np.isclose(abs(ti), 1 / (4 * beta)):
                    # l'Hospital for singularity at t = ±T/(4β)
                    h[i] = (beta / np.sqrt(2)) * (
                            ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))) +
                            ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
                    )
                else:
                    num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
                    den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
                    h[i] = num / den
            # Normalize to unit energy
            h /= np.sqrt(np.sum(h ** 2))
            return h

        def psk_symbols(Nsym):
            bits = np.random.randint(0, 2, size=2 * Nsym)
            # Gray mapping: 00->(1+1j), 01->(-1+1j), 11->(-1-1j), 10->(1-1j)
            b0 = bits[0::2]
            b1 = bits[1::2]
            I = 1 - 2 * b0
            Q = 1 - 2 * b1
            # Rotate by 45° so points at π/4, 3π/4, 5π/4, 7π/4 (optional, common)
            const = (I + 1j * Q) / np.sqrt(2)
            return const

        def upsample(symbols, sps):
            up = np.zeros(len(symbols) * sps, dtype=complex)
            up[::sps] = symbols
            return up

        def am_detector_symbol_rate(x, fs):
            env = np.abs(x)
            env = env - np.mean(env)
            f, Pxx = welch(env, fs=fs, nperseg=4096)
            idx = np.argmax(Pxx[1:]) + 1  # skip DC
            Rs_est = f[idx]
            return Rs_est, f, Pxx

        # ---------- Parameters ----------
        fs = self.fs  # sampling rate [Hz]
        Rs_true = 3000.0  # symbol rate [Hz]
        sps = int(round(fs / Rs_true))
        Rs_true = fs / sps  # ensure integer sps compatibility
        Nsym = 4000  # number of symbols
        beta = 0.35  # RRC roll-off
        span = 10  # RRC length in symbols
        snr_db = 20  # AWGN SNR in dB

        # ---------- Transmitter ----------
        syms = psk_symbols(Nsym)
        x_up = upsample(syms, sps)
        h = rrc_filter(beta, span, sps)
        x_bb = np.convolve(x_up, h, mode='same')

        # ---------- AWGN ----------
        sig_power = np.mean(np.abs(x_bb) ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*x_bb.shape) + 1j * np.random.randn(*x_bb.shape))
        x_rx = x_bb + noise

        # ---------- AM-detector-based Rs estimation ----------
        Rs_est, f, Pxx = am_detector_symbol_rate(self.sig, fs)

        # ---------- Visualizations ----------
        # PSD of the envelope with estimated Rs marker
        plt.figure(figsize=(9, 4.8))
        plt.semilogy(f, Pxx, label="Envelope PSD (Welch)")
        plt.axvline(Rs_est, linestyle='--', label=f"Estimated Rs ≈ {Rs_est:.2f} Hz")
        plt.title("AM Detector Envelope Spectrum (PSK type example)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power Spectral Density")
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()

        print(f"True Rs: {Rs_true:.2f} Hz | Estimated Rs: {Rs_est:.2f} Hz | sps={sps}")

    def test_func(self):
        return 0



if __name__ == '__main__':
    if spect:
        Signal.spectrogram()

    if ks:
        Signal.ks_detector()

    if time_domain:
        Signal.time_domain()

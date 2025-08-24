import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, upfirdn, hilbert, welch, spectrogram

# -------------------------------
# 1) Zero-stuffing + LPF 업샘플링
# -------------------------------

def upsample_zero_pad_manual(x, L, cutoff=None, numtaps=None, beta=8.6):
    """
    샘플 사이에 (L-1)개 0을 삽입(제로패딩) 후, LPF로 보간하는 업샘플링.
    - x: 실수 1D ndarray
    - L: 업샘플 배수(정수)
    - cutoff: FIR LPF 컷오프(정규화, Nyquist=1). 기본 0.45/L
    - numtaps: FIR 탭 수. 기본 64*L + 1 (홀수 권장)
    - beta: Kaiser 윈도우 파라미터(≈60 dB는 8.6)
    반환: y (업샘플 결과, 길이 ≈ len(x)*L)
    """
    x = np.asarray(x, dtype=np.float32)
    if L < 1 or int(L) != L:
        raise ValueError("L은 1 이상의 정수여야 합니다.")
    L = int(L)

    # 1) zero-stuffing
    y_zs = np.zeros(len(x) * L, dtype=np.float32)
    y_zs[::L] = x

    # 2) LPF 설계 (이상적으론 π/L 컷오프; firwin 정규화는 Nyquist=1 기준)
    if cutoff is None:
        cutoff = 0.45 / L  # 여유를 둔 컷오프
    if numtaps is None:
        numtaps = 64 * L + 1  # 배수가 커질수록 긴 필터 권장
    if numtaps % 2 == 0:
        numtaps += 1

    h = firwin(numtaps, cutoff, window=('kaiser', beta))
    # 업샘플 보상 스케일 (DC 게인 L)
    h = h * L

    # 3) 보간 필터링
    y = lfilter(h, [1.0], y_zs).astype(np.float32)
    return y


def upsample_zero_pad_upfirdn(x, L, cutoff=None, numtaps=None, beta=8.6):
    """
    upfirdn으로 제로패딩+LPF를 한번에 처리하는 업샘플링.
    매개변수 의미는 위와 동일.
    """
    x = np.asarray(x, dtype=np.float32)
    if L < 1 or int(L) != L:
        raise ValueError("L은 1 이상의 정수여야 합니다.")
    L = int(L)

    if cutoff is None:
        cutoff = 0.45 / L
    if numtaps is None:
        numtaps = 64 * L + 1
    if numtaps % 2 == 0:
        numtaps += 1

    h = firwin(numtaps, cutoff, window=('kaiser', beta))
    h = h * L  # DC 게인 보상
    y = upfirdn(h, x, up=L, down=1).astype(np.float32)
    return y


# -----------------------------------
# 2) N-th power spectrum / spectrogram
# -----------------------------------

def nth_power_spectrum(x, fs, n=4, use_hilbert=True, nperseg=4096):
    """
    실수 신호 x -> (analytic 변환) -> x^n -> Welch PSD
    - use_hilbert=True: 힐버트로 analytic 신호(단측대역) 생성 권장
    반환: f(정렬됨), Pxx(정렬됨)
    """
    x = np.asarray(x, dtype=np.float32)
    if use_hilbert:
        xa = hilbert(x).astype(np.complex64)
    else:
        xa = x.astype(np.complex64)

    xn = xa ** n
    f, Pxx = welch(xn, fs=fs, nperseg=nperseg, return_onesided=False,
                   detrend=False, scaling='density')
    # 0 Hz 중심으로 재정렬
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    return f, Pxx


def nth_power_spectrogram(x, fs, n=4, nperseg=2048, overlap=0.75, to_db=True):
    """
    실수 신호 x -> analytic -> x^n -> 스펙트로그램
    반환: f(정렬), t, S(dB 또는 선형, 정렬)
    """
    x = np.asarray(x, dtype=np.float32)
    xa = hilbert(x).astype(np.complex64)  # 실수 → analytic (권장)
    xn = xa ** n

    noverlap = int(nperseg * overlap)
    f, t, S = spectrogram(xn, fs=fs, nperseg=nperseg, noverlap=noverlap,
                          nfft=nperseg, return_onesided=False,
                          detrend=False, scaling='density', mode='psd')
    f = np.fft.fftshift(f)
    S = np.fft.fftshift(S, axes=0)
    if to_db:
        S = 10.0 * np.log10(S + 1e-12)
    return f, t, S


# ---------------------
# 3) 간단한 사용 데모
# ---------------------
if __name__ == "__main__":
    # (가짜 실수 신호) 두 톤 + 느린 AM
    fs_in = 8_000
    t = np.arange(0, 1.5, 1.0/fs_in, dtype=np.float32)
    x = 0.6*np.sin(2*np.pi*600*t) + 0.4*np.sin(2*np.pi*900*t)
    x *= 1.0 + 0.3*np.sin(2*np.pi*5*t)  # 약한 AM

    # A) zero-stuffing 업샘플 (수동/UPFIRDN 두 가지 중 택1)
    L = 4
    fs_out = fs_in * L

    y1 = upsample_zero_pad_manual(x, L)      # 수동 방식
    y2 = upsample_zero_pad_upfirdn(x, L)     # upfirdn 방식 (권장)

    # B) N-th power spectrum (Welch)
    f1, Pxx1 = nth_power_spectrum(y2, fs=fs_out, n=4, use_hilbert=True, nperseg=4096)

    plt.figure()
    plt.semilogy(f1, Pxx1)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.title("4th-Power Spectrum (upsampled)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # (선택) C) N-th power spectrogram
    f2, tt, S = nth_power_spectrogram(y2, fs=fs_out, n=4, nperseg=1024, overlap=0.75, to_db=True)
    plt.figure()
    plt.pcolormesh(f2, tt, S, shading='gouraud')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Time [s]")
    plt.title("4th-Power Spectrogram (upsampled)")
    cbar = plt.colorbar()
    cbar.set_label("dB")
    plt.tight_layout()
    plt.show()
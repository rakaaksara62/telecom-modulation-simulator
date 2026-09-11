import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

st.set_page_config(page_title="Simulasi Modulasi & Demodulasi Analog Lengkap", layout="wide")

st.title("Simulasi Real-Time Modulasi & Demodulasi Analog (AM, FM, PM)")
st.caption("Visualisasi pemrosesan transmitter, perkalian lokal pada receiver, hingga ekstraksi data.")

# --- SIDEBAR KONTROL PARAMETER ---
st.sidebar.header("Konfigurasi Parameter Sinyal")
mod_type = st.sidebar.selectbox(
    "Pilih Jenis Modulasi:", 
    ["PM (Phase Modulation)", "AM (Amplitude Modulation)", "FM (Frequency Modulation)"]
)

fm = st.sidebar.slider("Frekuensi Pesan fm (Hz)", min_value=1, max_value=8, value=2)
fc = st.sidebar.slider("Frekuensi Carrier fc (Hz)", min_value=20, max_value=80, value=35)
duration = 1.0  # durasi detik
fs = 4000       # frekuensi sampling simulasi (Hz)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Sinyal dasar
Am = 1.0
Ac = 1.0
m_t = Am * np.sin(2 * np.pi * fm * t)          # Pesan informasi
carrier_pure = Ac * np.cos(2 * np.pi * fc * t) # Carrier referensi murni

# Desain filter LPF Butterworth
cutoff_lpf = 2.5 * fm
b_lpf, a_lpf = butter(4, cutoff_lpf / (fs / 2), btype='low')

# ==============================================================================
# PEMROSESAN SINYAL
# ==============================================================================
if "PM" in mod_type:
    kp = st.sidebar.slider("Deviasi Fasa kp (radian)", 0.2, 2.5, 1.2)
    
    # 1. Transmitter: s(t) = Ac * cos(2*pi*fc*t + kp*m(t))
    tx_signal = Ac * np.cos(2 * np.pi * fc * t + kp * m_t)
    tx_eq = r"$s(t) = A_c \cos(2\pi f_c t + k_p m(t))$"
    
    # 2. Receiver - Local Oscillator tergeser -90 derajat (kuadratur): -sin(2*pi*fc*t)
    local_osc = -np.sin(2 * np.pi * fc * t)
    local_eq = r"$c_{local}(t) = -\sin(2\pi f_c t)$"
    
    # 3. Receiver - Multiplier (Mixer)
    # Secara trigonometri:
    # cos(A) * (-sin(B)) = -0.5 * [sin(A+B) + sin(A-B)]
    # A = 2*pi*fc*t + kp*m(t), B = 2*pi*fc*t
    # v_mult(t) = 0.5 * sin(kp * m(t)) - 0.5 * sin(4*pi*fc*t + kp * m(t))
    multiplier_out = tx_signal * local_osc
    mult_eq = r"$v_{mult}(t) = s(t) \times c_{local}(t) = \frac{1}{2}\sin(k_p m(t)) - \frac{1}{2}\sin(4\pi f_c t + k_p m(t))$"
    
    # 4. Receiver - Low Pass Filter (membuang komponen frekuensi tinggi 2*fc)
    rx_recovered = filtfilt(b_lpf, a_lpf, multiplier_out)
    rx_recovered = rx_recovered - np.mean(rx_recovered)
    if np.max(rx_recovered) > 0:
        rx_recovered = rx_recovered / np.max(rx_recovered)
    rec_eq = r"$m_{rec}(t) = \text{LPF}\{v_{mult}(t)\} \approx \frac{1}{2} k_p m(t)$"

elif "AM" in mod_type:
    ka = st.sidebar.slider("Indeks Modulasi ka", 0.1, 1.0, 0.7)
    
    # Transmitter AM Standar
    tx_signal = Ac * (1 + ka * m_t) * np.cos(2 * np.pi * fc * t)
    tx_eq = r"$s(t) = A_c [1 + k_a m(t)] \cos(2\pi f_c t)$"
    
    # Receiver AM Menggunakan Detektor Koheren (Pengali Lokal In-Phase)
    local_osc = 2 * np.cos(2 * np.pi * fc * t)
    local_eq = r"$c_{local}(t) = 2 \cos(2\pi f_c t)$"
    
    multiplier_out = tx_signal * local_osc
    mult_eq = r"$v_{mult}(t) = A_c [1 + k_a m(t)] + A_c [1 + k_a m(t)] \cos(4\pi f_c t)$"
    
    rx_filtered = filtfilt(b_lpf, a_lpf, multiplier_out)
    rx_recovered = rx_filtered - np.mean(rx_filtered)  # Blokir komponen DC
    if np.max(rx_recovered) > 0:
        rx_recovered = rx_recovered / np.max(rx_recovered)
    rec_eq = r"$m_{rec}(t) = \text{LPF}\{v_{mult}(t)\} - V_{DC} \propto m(t)$"

else:  # FM
    kf = st.sidebar.slider("Sensitivitas Frekuensi kf (Hz/V)", 5, 30, 15)
    
    # Transmitter FM
    integral_m = np.cumsum(m_t) / fs
    tx_signal = Ac * np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * integral_m)
    tx_eq = r"$s(t) = A_c \cos\left(2\pi f_c t + 2\pi k_f \int_0^t m(\tau) d\tau\right)$"
    
    # Receiver FM - Slope Detector / Frequency Discriminator
    diff_signal = np.abs(np.diff(tx_signal, prepend=tx_signal[0])) * (fs / (2 * np.pi * fc))
    local_osc = np.cos(2 * np.pi * fc * t)
    local_eq = r"$\text{Referensi Frekuensi Pusat } f_c = " + f"{fc}" + r"\text{ Hz}$"
    
    multiplier_out = diff_signal
    mult_eq = r"$v_{disc}(t) = \left| \frac{d}{dt} s(t) \right| \propto [2\pi f_c + 2\pi k_f m(t)]$"
    
    rx_recovered = filtfilt(b_lpf, a_lpf, diff_signal)
    rx_recovered = rx_recovered - np.mean(rx_recovered)
    if np.max(rx_recovered) > 0:
        rx_recovered = rx_recovered / np.max(rx_recovered)
    rec_eq = r"$m_{rec}(t) = \text{LPF}\{v_{disc}(t)\} - V_{DC} \propto m(t)$"

# ==============================================================================
# PLOTTING GRAFIK LENGKAP
# ==============================================================================
fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)

# 1. Sinyal Pesan
axes[0].plot(t, m_t, color='blue', lw=1.8)
axes[0].set_title(f"1. Sinyal Informasi / Pesan Asli: $m(t) = A_m \sin(2\pi f_m t)$", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Amplitudo (V)")
axes[0].grid(True, linestyle='--', alpha=0.5)

# 2. Sinyal Termodulasi (TX Out)
axes[1].plot(t, tx_signal, color='red', lw=1.2)
if "AM" in mod_type:
    axes[1].plot(t, Ac * (1 + ka * m_t), 'k--', alpha=0.5, label='Envelope')
    axes[1].plot(t, -Ac * (1 + ka * m_t), 'k--', alpha=0.5)
    axes[1].legend(loc='upper right')
axes[1].set_title(f"2. Sinyal Transmisi Termodulasi: {tx_eq}", fontsize=12, fontweight='bold')
axes[1].set_ylabel("Amplitudo")
axes[1].grid(True, linestyle='--', alpha=0.5)

# 3. Osilator Lokal Receiver
axes[2].plot(t, local_osc, color='purple', lw=1.2, linestyle='-')
axes[2].set_title(f"3. Sinyal Pengali Lokal Receiver (Carrier Generator): {local_eq}", fontsize=12, fontweight='bold')
axes[2].set_ylabel("Amplitudo")
axes[2].grid(True, linestyle='--', alpha=0.5)

# 4. Hasil Perkalian (Mixer Output / Selisih Fasa & Frekuensi Ganda)
axes[3].plot(t, multiplier_out, color='darkorange', lw=1.0)
axes[3].set_title(f"4. Hasil Pengali Mixer: {mult_eq}", fontsize=11, fontweight='bold')
axes[3].set_ylabel("Tegangan Mixer")
axes[3].grid(True, linestyle='--', alpha=0.5)

# 5. Sinyal Hasil Demodulasi (Keluaran LPF)
axes[4].plot(t, rx_recovered, color='green', lw=2.0, label='Sinyal Hasil Demodulasi')
axes[4].plot(t, m_t / np.max(m_t), color='blue', linestyle='--', alpha=0.4, label='Pesan Target Asli (Normalisasi)')
axes[4].set_title(f"5. Sinyal Diterima Akhir (Output LPF): {rec_eq}", fontsize=12, fontweight='bold')
axes[4].set_xlabel("Waktu (detik)")
axes[4].set_ylabel("Amplitudo")
axes[4].grid(True, linestyle='--', alpha=0.5)
axes[4].legend(loc='upper right')

plt.tight_layout()
st.pyplot(fig)

# ==============================================================================
# PENJELASAN MATEMATIS DETAIL TENTANG DETEKTOR FASA (PM)
# ==============================================================================
st.markdown("---")
st.subheader("Buku Catatan Teknis: Mengapa Pengali Lokal Mengekstrak Fasa Secara Real-Time?")
st.markdown(r"""
Pada modulasi fasa, sinyal yang masuk ke antena penerima adalah:
$$s(t) = A_c \cos(2\pi f_c t + k_p m(t))$$

Di penerima, sinyal ini langsung dikalikan dengan **osilator lokal** berkondisi kuadratur (berbeda fasa $90^\circ$ atau $-\sin(2\pi f_c t)$):
$$v_{mult}(t) = \left[ A_c \cos(2\pi f_c t + k_p m(t)) \right] \times \left[ -\sin(2\pi f_c t) \right]$$

Menggunakan identitas trigonometri: $\cos(\alpha)\sin(\beta) = \frac{1}{2}[\sin(\alpha+\beta) - \sin(\alpha-\beta)]$:
$$v_{mult}(t) = \underbrace{\frac{1}{2} A_c \sin(k_p m(t))}_{\text{Sinyal Baseband (Selisih Fasa)}} - \underbrace{\frac{1}{2} A_c \sin(4\pi f_c t + k_p m(t))}_{\text{Frekuensi Tinggi ganda } 2f_c}$$

* **Komponen Frekuensi Ganda ($2f_c$):** Frekuensinya sangat tinggi (pada grafik 4 terlihat seperti gerigi berosilasi rapat). Komponen ini langsung **dibuang** oleh Low Pass Filter (LPF).
* **Komponen Baseband:** Untuk deviasi fasa kecil ($k_p m(t) \ll 1\text{ rad}$), berlaku aproksimasi $\sin(\theta) \approx \theta$. Sehingga:
$$m_{rec}(t) \approx \frac{1}{2} A_c k_p m(t)$$
Tegangan listrik seketika yang keluar dari filter **berbanding lurus langsung dengan sinyal suara $m(t)$** tanpa harus menunggu gelombang memotong sumbu nol.
""")
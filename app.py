import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt

st.set_page_config(page_title="Simulasi Modulasi & Demodulasi Analog", layout="wide")

st.title("Simulasi Real-Time Modulasi & Demodulasi Analog (AM, FM, PM)")
st.caption("Visualisasi pemrosesan sinyal sesaat (instantaneous) dari sisi Transmitter hingga Receiver.")

# --- SIDEBAR KONTROL PARAMETER ---
st.sidebar.header("Konfigurasi Sinyal")
mod_type = st.sidebar.selectbox("Pilih Jenis Modulasi:", ["AM (Amplitude Modulation)", "FM (Frequency Modulation)", "PM (Phase Modulation)"])

fm = st.sidebar.slider("Frekuensi Sinyal Informasi / Suara (Hz)", min_value=1, max_value=10, value=2)
fc = st.sidebar.slider("Frekuensi Carrier (Hz)", min_value=20, max_value=100, value=40)
duration = 1.0  # durasi detik
fs = 2000       # frekuensi sampling simulasi (Hz)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Sinyal dasar
m_t = np.sin(2 * np.pi * fm * t)          # Sinyal informasi
carrier = np.cos(2 * np.pi * fc * t)     # Sinyal carrier unmodulated

# --- RUMUS PEMROSESAN TRANSMITTER & RECEIVER ---
if "AM" in mod_type:
    ka = st.sidebar.slider("Indeks Modulasi (ka)", 0.1, 1.0, 0.7)
    
    # 1. Transmitter: Perkalian instan envelope
    tx_signal = (1 + ka * m_t) * np.cos(2 * np.pi * fc * t)
    
    # 2. Receiver (Detektor Selubung: Dioda Rectifier + Low Pass Filter)
    rectified = np.maximum(0, tx_signal) # Efek dioda penyearah
    # Filter LPF orde 4
    b, a_coeff = butter(4, (2 * fm * 2) / fs, btype='low')
    rx_recovered = filtfilt(b, a_coeff, rectified)
    rx_recovered = (rx_recovered - np.mean(rx_recovered)) # Hapus offset DC
    if np.max(rx_recovered) > 0:
        rx_recovered = rx_recovered / np.max(rx_recovered) # Normalisasi

    tx_desc = f"Amplitudo sesaat carrier langsung dikalikan nilai tegangan: A(t) = [1 + {ka} * m(t)]."
    rx_desc = "Penerima menggunakan Dioda (memotong lembah negatif) lalu Kapasitor/LPF menahan muatan puncak sesaat (envelope)."

elif "FM" in mod_type:
    kf = st.sidebar.slider("Deviasi Frekuensi kf (Hz/V)", 5, 30, 15)
    
    # 1. Transmitter: Frekuensi sesaat mengendalikan laju fasa osilator (VCO)
    integral_m = np.cumsum(m_t) / fs
    tx_signal = np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * integral_m)
    
    # 2. Receiver (Diskriminator Frekuensi / Slope Detector)
    # Turunan sinyal d/dt mengubah variasi frekuensi menjadi variasi amplitudo
    diff_signal = np.abs(np.diff(tx_signal, prepend=tx_signal[0]))
    b, a_coeff = butter(4, (2 * fm * 2) / fs, btype='low')
    rx_recovered = filtfilt(b, a_coeff, diff_signal)
    rx_recovered = (rx_recovered - np.mean(rx_recovered))
    if np.max(rx_recovered) > 0:
        rx_recovered = rx_recovered / np.max(rx_recovered)

    tx_desc = "VCO berputar lebih cepat saat sinyal pesan bernilai positif dan melambat saat negatif secara real-time."
    rx_desc = "Rangkaian slope/diferensiator mengukur laju putar gelombang sesaat, lalu LPF mengekstrak kembali tegangannya."

else: # PM
    kp = st.sidebar.slider("Deviasi Fasa kp (Radian)", 0.5, 3.14, 1.5)
    
    # 1. Transmitter: Sudut fasa digeser langsung proporsional terhadap m(t)
    tx_signal = np.cos(2 * np.pi * fc * t + kp * m_t)
    
    # 2. Receiver (Phase Detector: Mengalikan sinyal RX dengan carrier lokal quadrature)
    # Menggunakan sinyal acuan lokal tergeser 90 derajat (-sin)
    local_ref = -np.sin(2 * np.pi * fc * t)
    mixed = tx_signal * local_ref
    b, a_coeff = butter(4, (2 * fm * 2) / fs, btype='low')
    rx_recovered = filtfilt(b, a_coeff, mixed)
    rx_recovered = (rx_recovered - np.mean(rx_recovered))
    if np.max(rx_recovered) > 0:
        rx_recovered = rx_recovered / np.max(rx_recovered)

    tx_desc = f"Rangkaian penunda fasa memajukan/memundurkan sudut carrier sebesar {kp} * m(t) pada detik yang sama."
    rx_desc = "Detektor Fasa mengalikan sinyal yang datang dengan sinyal osilator lokal untuk mengukur selisih sudut fasa seketika."

# --- TAMPILAN GRAFIK ---
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Plot 1: Sinyal Asli
axes[0].plot(t, m_t, color='blue', label='m(t) Sinyal Pesan')
axes[0].set_title("1. Sinyal Informasi Asli (Tegangan Mikrofon / Suara)")
axes[0].set_ylabel("Amplitudo (V)")
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(loc='upper right')

# Plot 2: Carrier Murni
axes[1].plot(t, carrier, color='gray', linestyle=':', alpha=0.7, label='Carrier Murni (fc)')
axes[1].set_title(f"2. Gelombang Pembawa Murni ({fc} Hz)")
axes[1].set_ylabel("Amplitudo")
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(loc='upper right')

# Plot 3: Sinyal Hasil Modulasi (Transmitter Out)
axes[2].plot(t, tx_signal, color='red', label='Sinyal Termodulasi')
if "AM" in mod_type:
    envelope_up = 1 + ka * m_t
    envelope_down = -envelope_up
    axes[2].plot(t, envelope_up, 'g--', alpha=0.7, label='Envelope Atas')
    axes[2].plot(t, envelope_down, 'g--', alpha=0.7)
axes[2].set_title(f"3. Sinyal yang Dipancarkan Antena TX [{tx_desc}]")
axes[2].set_ylabel("Amplitudo")
axes[2].grid(True, linestyle='--', alpha=0.6)
axes[2].legend(loc='upper right')

# Plot 4: Sinyal Hasil Demodulasi (Receiver Out)
axes[3].plot(t, rx_recovered, color='green', linewidth=2, label='m_rec(t) Hasil Deteksi')
axes[3].plot(t, m_t, color='blue', linestyle='--', alpha=0.4, label='Target Asli')
axes[3].set_title(f"4. Sinyal Keluaran Speaker RX [{rx_desc}]")
axes[3].set_xlabel("Waktu (detik)")
axes[3].set_ylabel("Amplitudo (V)")
axes[3].grid(True, linestyle='--', alpha=0.6)
axes[3].legend(loc='upper right')

plt.tight_layout()
st.pyplot(fig)

# --- PENJELASAN MEKANISME REAL-TIME ---
st.markdown("---")
st.subheader("Mekanisme Pengambilan Data Sesaat")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Di Sisi Pemancar (Transmitter):**")
    st.write(tx_desc)
    st.info("Nilai tegangan $m(t)$ pada detik itu langsung mengubah fisik sinyal pembawa tanpa harus mengumpulkan sampel berdurasi panjang.")

with col2:
    st.markdown("**Di Sisi Penerima (Receiver):**")
    st.write(rx_desc)
    st.info("Penerima tidak menunggu gelombang melintasi $x=0$, melainkan memanfaatkan sifat pengisian muatan RC atau filter frekuensi untuk membaca nilai tegangan seketika.")
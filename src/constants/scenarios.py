"""
src/constants/scenarios.py
==========================
Blast Furnace mülakat / test senaryoları.

Her senaryo, farklı bir çalışma rejimini temsil eder.
Engineered feature'lar (rolling mean, lag, diff vb.) BFPredictor
içinde otomatik hesaplandığı için sadece base sensör değerleri
yeterlidir.

Senaryolar feature importance analizlerine göre kurgulanmıştır:
  - Si, R, Fb, Th, dP, CO2, H2  →  en dominant temel sensörler
  - Tt1-Tt4, Tp3-Tp4            →  termal profil
  - Tc, Ph, Pc, Pt              →  soğutma / basınç dağılımı
"""

from typing import Dict, Any

# ---------------------------------------------------------------------------
# Base değer aralıkları (referans)
# ---------------------------------------------------------------------------
# Fb  : 3500 – 4500  Nm³/h   (hava debisi)
# Th  : 1050 – 1200  °C      (sıcak hava sıcaklığı)
# R   : 3.5  – 5.5           (cevher / kok oranı)
# dP  : 0.8  – 1.5   bar     (basınç farkı)
# CO2 : 20   – 25    %       (CO2 oranı)
# H2  : 3    – 8     %       (H2 oranı)
# Si  : 0.2  – 0.8   %       (silikon)
# Tc  : 60   – 130   °C      (soğutma suyu)
# Ph  : 2.0  – 3.5   bar     (üst basınç)
# Pc  : 1.5  – 3.0   bar     (orta basınç)
# Pt  : 1.0  – 2.0   bar     (alt basınç)
# Tt  : 100  – 250   °C      (üst sıcaklıklar)
# Tp  : 150  – 500   °C      (pres sıcaklıklar)
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Dict[str, Any]] = {

    # =====================================================================
    # 1. NORMAL OPERATION
    # =====================================================================
    "normal_operation": {
        "description": (
            "Stabil çalışma rejimi. Ortalama Si (~0.45), dengeli R/Fb/Th, "
            "normal gaz kompozisyonu ve termal profil."
        ),
        "expected_si_range": [0.40, 0.50],
        "sensor_values": {
            # Besleme
            "Fb": 4100.0,   # Nm³/h  — orta-yüksek hava
            "Th": 1120.0,   # °C     — standart sıcak hava
            "R": 4.50,      #        — orta cevher/kok oranı
            # Basınç
            "dP": 1.20,     # bar    — normal basınç farkı
            # Kimya
            "CO2": 22.5,    # %      — normal CO2
            "H2": 5.0,      # %      — normal H2
            # Mevcut Si
            "Si": 0.45,     # %      — stabil silikon
            # Soğutma / basınç dağılımı
            "Tc": 95.0,     # °C
            "Ph": 2.50,     # bar
            "Pc": 2.20,     # bar
            "Pt": 1.50,     # bar
            # Üst sıcaklıklar (Tt)
            "Tt1": 145.0, "Tt2": 152.0, "Tt3": 148.0, "Tt4": 150.0,
            # Pres sıcaklıklar (Tp) — düzgün dağılım
            "Tp1": 280.0, "Tp2": 310.0, "Tp3": 340.0, "Tp4": 365.0, "Tp5": 380.0,
            "Tp6": 390.0, "Tp7": 385.0, "Tp8": 370.0, "Tp9": 350.0, "Tp10": 320.0,
        },
    },

    # =====================================================================
    # 2. HIGH SILICON RISK
    # =====================================================================
    "high_silicon_risk": {
        "description": (
            "Yüksek Si riski senaryosu. Düşük R (daha fazla kok), düşük Fb, "
            "yüksek Th, yüksek H2, düşük CO2. Si ~0.72 seviyelerinde. "
            "Feature importance'da R, Si_roll_mean ve Fb dominant."
        ),
        "expected_si_range": [0.65, 0.80],
        "sensor_values": {
            # Besleme — daha fazla kok (düşük R), az hava, sıcak hava yüksek
            "Fb": 3650.0,   # Nm³/h  — düşük hava debisi
            "Th": 1185.0,   # °C     — yüksek sıcak hava
            "R": 3.75,      #        — düşük cevher/kok = fazla kok
            # Basınç — biraz düşük (daha az hava)
            "dP": 1.05,     # bar
            # Kimya — reduksiyon güçlü, CO2 düşük, H2 yüksek
            "CO2": 20.8,    # %
            "H2": 7.2,      # %
            # Mevcut Si — zaten yüksek
            "Si": 0.72,     # %
            # Soğutma / basınç
            "Tc": 125.0,    # °C     — yüksek sıcaklık yükü
            "Ph": 2.85,     # bar
            "Pc": 2.60,     # bar
            "Pt": 1.75,     # bar
            # Üst sıcaklıklar — yüksek
            "Tt1": 195.0, "Tt2": 205.0, "Tt3": 198.0, "Tt4": 202.0,
            # Pres sıcaklıklar — yüksek, özellikle orta-bölge (Tp3-Tp5)
            "Tp1": 320.0, "Tp2": 360.0, "Tp3": 410.0, "Tp4": 445.0, "Tp5": 465.0,
            "Tp6": 470.0, "Tp7": 460.0, "Tp8": 440.0, "Tp9": 415.0, "Tp10": 375.0,
        },
    },

    # =====================================================================
    # 3. LOW SILICON RISK
    # =====================================================================
    "low_silicon_risk": {
        "description": (
            "Düşük Si riski senaryosu. Yüksek R (daha az kok), yüksek Fb, "
            "düşük Th, düşük H2, yüksek CO2. Si ~0.28 seviyelerinde. "
            "Oksidasyon güçlü, reduksiyon zayıf."
        ),
        "expected_si_range": [0.22, 0.35],
        "sensor_values": {
            # Besleme — az kok (yüksek R), çok hava, düşük sıcaklık
            "Fb": 4450.0,   # Nm³/h  — yüksek hava debisi
            "Th": 1070.0,   # °C     — düşük sıcak hava
            "R": 5.20,      #        — yüksek cevher/kok = az kok
            # Basınç — yüksek (çok hava)
            "dP": 1.42,     # bar
            # Kimya — oksidasyon güçlü, CO2 yüksek, H2 düşük
            "CO2": 24.2,    # %
            "H2": 3.4,      # %
            # Mevcut Si — düşük
            "Si": 0.28,     # %
            # Soğutma / basınç
            "Tc": 72.0,     # °C     — düşük termal yük
            "Ph": 2.15,     # bar
            "Pc": 1.90,     # bar
            "Pt": 1.25,     # bar
            # Üst sıcaklıklar — düşük
            "Tt1": 115.0, "Tt2": 122.0, "Tt3": 118.0, "Tt4": 120.0,
            # Pres sıcaklıklar — düşük
            "Tp1": 220.0, "Tp2": 245.0, "Tp3": 270.0, "Tp4": 290.0, "Tp5": 305.0,
            "Tp6": 310.0, "Tp7": 300.0, "Tp8": 285.0, "Tp9": 265.0, "Tp10": 240.0,
        },
    },

    # =====================================================================
    # 4. THERMAL IMBALANCE / ABNORMAL PRESSURE
    # =====================================================================
    "thermal_imbalance": {
        "description": (
            "Termal dengesizlik + anormal basınç dağılımı. dP düşük, "
            "Tp profili bozuk (parçalı), Tt1-Tt3 arası farklı. "
            "Si ~0.55, stabil olmayan rejim. Feature importance'da "
            "dP_roll_mean, pressure_distribution_ratio ve Tp4 öne çıkar."
        ),
        "expected_si_range": [0.50, 0.62],
        "sensor_values": {
            # Besleme — kararsız
            "Fb": 3900.0,   # Nm³/h
            "Th": 1140.0,   # °C
            "R": 4.20,      #        — biraz düşük R
            # Basınç — düşük fark, dağılım bozuk
            "dP": 0.92,     # bar    — anormal düşük basınç farkı
            # Kimya — dengesiz
            "CO2": 21.5,    # %
            "H2": 6.1,      # %
            # Mevcut Si
            "Si": 0.55,     # %
            # Soğutma / basınç — anormal
            "Tc": 110.0,    # °C
            "Ph": 2.30,     # bar
            "Pc": 2.80,     # bar    — orta basınç yüksek (tıkanma)
            "Pt": 1.10,     # bar    — alt basınç düşük
            # Üst sıcaklıklar — dengesiz
            "Tt1": 135.0, "Tt2": 175.0, "Tt3": 128.0, "Tt4": 168.0,
            # Pres sıcaklıklar — bozuk dağılım (channeling işareti)
            "Tp1": 260.0, "Tp2": 290.0, "Tp3": 380.0, "Tp4": 295.0, "Tp5": 420.0,
            "Tp6": 310.0, "Tp7": 400.0, "Tp8": 275.0, "Tp9": 360.0, "Tp10": 300.0,
        },
    },

    # =====================================================================
    # 5. EXTREME HIGH SILICON (edge case)
    # =====================================================================
    "extreme_high_si": {
        "description": (
            "Aşırı yüksek Si senaryosu (~0.78). Çok düşük R, çok düşük Fb, "
            "çok yüksek Th. RED alert tetiklenmeli. Edge-case testi."
        ),
        "expected_si_range": [0.75, 0.85],
        "sensor_values": {
            "Fb": 3550.0,
            "Th": 1195.0,
            "R": 3.60,
            "dP": 0.98,
            "CO2": 20.2,
            "H2": 7.8,
            "Si": 0.78,
            "Tc": 128.0,
            "Ph": 2.95,
            "Pc": 2.75,
            "Pt": 1.80,
            "Tt1": 210.0, "Tt2": 225.0, "Tt3": 215.0, "Tt4": 220.0,
            "Tp1": 340.0, "Tp2": 390.0, "Tp3": 450.0, "Tp4": 485.0, "Tp5": 495.0,
            "Tp6": 490.0, "Tp7": 475.0, "Tp8": 455.0, "Tp9": 430.0, "Tp10": 395.0,
        },
    },
}

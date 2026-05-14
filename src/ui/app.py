"""
src/ui/app.py — BF Intelligence Streamlit Paneli
=================================================
Çalıştırma:
  streamlit run src/ui/app.py --server.port 8501

Multi-horizon:
  - Tahmin ufku seçici eklendi (2h / 4h / 6h / 8h)
  - Sonuç kartı seçilen ufku gösteriyor
  - SHAP sekmesi horizon'a göre ilgili grafikleri gösteriyor
  - /health'ten yüklü horizon listesi alınıyor
"""

import streamlit as st
import requests
import os
import time
from pathlib import Path

from src.constants.scenarios import SCENARIOS

# ---------------------------------------------------------------------------
# Konfigürasyon
# ---------------------------------------------------------------------------
API_URL     = os.getenv("BF_API_URL",   "http://api:8000")
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/app/reports"))

st.set_page_config(
    page_title="BF Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "last_res":       None,
    "si_now":         0.45,
    "request_ts":     None,
    "horizon_hours":  8,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("API'ye ulaşılamıyor. Servisin ayakta olduğundan emin olun.")
    except requests.exceptions.HTTPError as e:
        st.error(f"API Hatası ({e.response.status_code}): {e.response.text}")
    except Exception as e:
        st.error(f"Beklenmedik hata: {e}")
    return None


def _api_get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_status = st.columns([4, 1])
with col_title:
    st.title("🏗️ Yüksek İzleme Paneli")

with col_status:
    health = _api_get("/health")
    if health:
        loaded_horizons = health.get("loaded_horizons", [])
        icon = "🟢" if health.get("model_ready") else "🟡"
        st.metric(
            "API Durumu",
            f"{icon} {'Hazır' if health.get('model_ready') else 'Model Bekleniyor'}"
        )
        if loaded_horizons:
            st.caption(f"Yüklü modeller: {', '.join(str(h)+'h' for h in loaded_horizons)}")
        if st.button("🔄 Modeli Yenile", help="Retrain sonrası modeli sıcak yükler"):
            res = _api_post("/reload", {})
            if res:
                st.success(f"Model yeniden yüklendi. Horizon'lar: {res.get('horizons')}")
                time.sleep(1)
                st.rerun()
    else:
        st.metric("API Durumu", "🔴 Erişilemiyor")

st.markdown("---")

# ---------------------------------------------------------------------------
# Sekmeler
# ---------------------------------------------------------------------------
t_main, t_autopsy, t_shap = st.tabs([
    "🚀 Canlı Tahmin",
    "🩺 Fırın Otopsisi",
    "🧠 Model Zekası (SHAP)",
])


# ── TAB 1: CANLI TAHMİN ────────────────────────────────────────────────────
with t_main:
    col_inp, col_res = st.columns([1, 1], gap="large")

    with col_inp:
        st.subheader("📡 Sensör Girişi")

        # ── Tahmin ufku seçici ──────────────────────────────────────────────
        st.markdown("#### ⏱️ Tahmin Ufku")
        horizon_options = {
            "2 Saat Sonra":  2,
            "4 Saat Sonra":  4,
            "6 Saat Sonra":  6,
            "8 Saat Sonra":  8,
        }

        # Yüklü horizon'lardan hangisi varsa aktif et
        loaded = health.get("loaded_horizons", [2, 4, 6, 8]) if health else [2, 4, 6, 8]
        available_options = {k: v for k, v in horizon_options.items() if v in loaded}

        if not available_options:
            st.warning("Hiçbir model yüklü değil.")
            available_options = horizon_options  # fallback — tümünü göster

        selected_label = st.radio(
            label="Kaç saat sonrasını tahmin etmek istiyorsunuz?",
            options=list(available_options.keys()),
            index=list(available_options.values()).index(
                st.session_state["horizon_hours"]
                if st.session_state["horizon_hours"] in available_options.values()
                else list(available_options.values())[-1]
            ),
            horizontal=True,
        )
        selected_horizon = available_options[selected_label]
        st.session_state["horizon_hours"] = selected_horizon

        st.markdown("#### 🎭 Çalışma Senaryosu")
        from src.constants.scenarios import SCENARIOS # Import'u buraya veya en üste ekleyebilirsin
        
        selected_scenario = st.selectbox(
            "Hazır bir senaryo yüklemek ister misiniz?",
            options=list(SCENARIOS.keys()),
            index=0,
            format_func=lambda x: x.replace("_", " ").title(),
            key="scenario_selector" # Bu sabit kalabilir
        )
        
        # Seçilen senaryonun verilerini al
        scenario_data = SCENARIOS[selected_scenario]["sensor_values"]
        st.info(f"💡 {SCENARIOS[selected_scenario]['description']}")

        s_suffix = f"_{selected_scenario}_{selected_horizon}"

        st.markdown("---")

        # ── Sensör parametreleri ─────────────────────────────────────────────
        with st.expander("Besleme Parametreleri", expanded=True):
            # Senaryodan veriyi çekiyoruz, yoksa fallback olarak senin verdiğin default değeri kullanıyoruz
            fb  = st.number_input("Hava Debisi (Fb)", 
                                value=float(scenario_data.get("Fb", 3800.0)), 
                                step=10.0, key=f"fb{s_suffix}")
            
            th  = st.number_input("Sıcak Hava Sıcaklığı (Th)", 
                                value=float(scenario_data.get("Th", 1150.0)), 
                                step=5.0, key=f"th{s_suffix}")
            
            r   = st.number_input("Cevher/Kok Oranı (R)", 
                                value=float(scenario_data.get("R", 3.8)), 
                                step=0.1, key=f"r{s_suffix}")
            
            tc  = st.number_input("Soğutma Suyu Sıcaklığı (Tc)", 
                                value=float(scenario_data.get("Tc", 65.0)), 
                                step=1.0, key=f"tc{s_suffix}")

        with st.expander("Gaz & Basınç"):
            dp  = st.number_input("Basınç Farkı (dP)", 
                                value=float(scenario_data.get("dP", 1.45)), 
                                step=0.05, key=f"dp{s_suffix}")
            
            co2 = st.number_input("CO2 Oranı (%)", 
                                value=float(scenario_data.get("CO2", 18.5)), 
                                step=0.5, key=f"co2{s_suffix}")
            
            h2  = st.number_input("H2 Oranı (%)", 
                                value=float(scenario_data.get("H2", 4.2)), 
                                step=0.1, key=f"h2{s_suffix}")
            
            ph  = st.number_input("Üst Basınç (Ph)", 
                                value=float(scenario_data.get("Ph", 2.1)), 
                                step=0.1, key=f"ph{s_suffix}")
            
            pc  = st.number_input("Orta Basınç (Pc)", 
                                value=float(scenario_data.get("Pc", 1.8)), 
                                step=0.1, key=f"pc{s_suffix}")
            
            pt  = st.number_input("Alt Basınç (Pt)", 
                                value=float(scenario_data.get("Pt", 1.2)), 
                                step=0.1, key=f"pt{s_suffix}")
            
        with st.expander("Gövde & Tepe Sıcaklıkları (Thermal Sensors)"):
            st.info("Bu sensörler fırın içi sıcaklık dağılımını (Imbalance) ölçer.")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Gövde Sıcaklıkları (Tp)**")
                tp_vals = []
                for i in range(1, 11):
                    # Tp1, Tp2... şeklinde dinamik key oluşturup senaryodan çekiyoruz
                    tp_key = f"Tp{i}"
                    val = st.number_input(f"Tp{i}", 
                                        value=float(scenario_data.get(tp_key, 450.0)), 
                                        step=5.0, key=f"tp{i}{s_suffix}")
                    tp_vals.append(val)
                    
            with col2:
                st.write("**Tepe Sıcaklıkları (Tt)**")
                tt_vals = []
                for i in range(1, 5):
                    # Tt1, Tt2... şeklinde dinamik key oluşturup senaryodan çekiyoruz
                    tt_key = f"Tt{i}"
                    val = st.number_input(f"Tt{i}", 
                                        value=float(scenario_data.get(tt_key, 120.0)), 
                                        step=5.0, key=f"tt{i}{s_suffix}")
                    tt_vals.append(val)    

        # Mevcut Silicon girişi
        si_now = st.number_input(
            "🧪 Mevcut Silicon (Si %)", 
            value=float(scenario_data.get("Si", 0.45)), 
            step=0.01, 
            key=f"si_now{s_suffix}"
        )

        if st.button("⚡ Analiz Et", type="primary", use_container_width=True):
            payload = {
                "horizon_hours": selected_horizon,
                "Fb": fb,
                "Th": th,
                "R": r,
                "Tc": tc,
                "dP": dp,
                "CO2": co2,
                "H2": h2,
                "Ph": ph,
                "Pc": pc,
                "Pt": pt,
                "Si": si_now,
            }


            for i, val in enumerate(tp_vals, 1):
                payload[f"Tp{i}"] = val

            for i, val in enumerate(tt_vals, 1):
                payload[f"Tt{i}"] = val    

            with st.spinner(f"{selected_horizon} saatlik tahmin hesaplanıyor..."):
                res = _api_post("/predict", payload)
                if res:
                    st.session_state["last_res"]   = res
                    st.session_state["request_ts"] = time.strftime("%H:%M:%S")

                    

    with col_res:
        st.subheader("🎯 Tahmin Sonucu")
        res = st.session_state.get("last_res")

        if res:
            pred          = res["prediction"]
            delta         = res["delta"]
            alert         = res["alert"]
            alert_msg     = res["alert_msg"]
            horizon_hours = res.get("horizon_hours", selected_horizon)
            ts            = st.session_state.get("request_ts", "")

            st.caption(f"Son güncelleme: {ts}")
            st.metric(
                label=f"{horizon_hours} Saat Sonra Beklenen Si",
                value=f"{pred:.4f} %",
                delta=f"{delta:+.4f}",
                delta_color="inverse",
            )

            st.markdown("---")

            if alert == "RED":
                st.error(f"🚨 **KRİTİK ALARM**\n\n{alert_msg}")
            elif alert == "YELLOW":
                st.warning(f"⚠️ **DİKKAT**\n\n{alert_msg}")
            else:
                st.success(f"✅ **Stabil**\n\n{alert_msg}")

            with st.expander("Detaylar"):
                st.json({
                    "horizon_hours": horizon_hours,
                    "prediction":    pred,
                    "delta":         delta,
                    "alert":         alert,
                    "si_input":      st.session_state["si_now"],
                })
        else:
            st.info(
                "Tahmin almak için:\n"
                "1. Tahmin ufkunu seçin (2h / 4h / 6h / 8h)\n"
                "2. Sensör değerlerini girin\n"
                "3. **Analiz Et** butonuna basın"
            )


# ── TAB 2: FIRIN OTOPSİSİ ─────────────────────────────────────────────────
with t_autopsy:
    st.subheader("🩺 Fırın Sağlık Raporu")

    autopsy_img = REPORTS_DIR / "furnace_autopsy.png"
    if autopsy_img.exists():
        st.image(str(autopsy_img), caption="Duruşlar ve Termal Şok Analizi", use_container_width=True)
    else:
        st.warning(
            f"Otopsi raporu henüz oluşturulmamış.\n\n"
            f"Çalıştır: `python analysis/furnace_autopsy.py`\n\n"
            f"Beklenen konum: `{autopsy_img}`"
        )

    st.markdown("#### 🔍 Yorumlama Rehberi")
    st.markdown("""
- **Gri çizgi** — Silicon'un tüm döküm süreci boyunca akışı  
- **Kırmızı noktalar** — Isolation Forest'ın anomali olarak işaretlediği döküm anları  
- **Siyah kesikli çizgiler** — Fırının durduğu ve yeniden başladığı geçiş noktaları  
- **Uyanış Şoku** — Duruş sonrasındaki ilk dökümlerde Silicon'un neden sıçradığını gösterir  
""")


# ── TAB 3: MODEL ZEKASI ───────────────────────────────────────────────────
with t_shap:
    st.subheader("🧠 Model Neden Bu Kararı Verdi?")

    # Horizon seçici — SHAP grafikleri horizon'a özgü
    shap_horizon = st.radio(
        "Hangi modelin SHAP analizini görmek istiyorsunuz?",
        options=[2, 4, 6, 8],
        format_func=lambda h: f"{h} Saat",
        horizontal=True,
        index=[2, 4, 6, 8].index(st.session_state.get("horizon_hours", 8)),
    )

    col_bar, col_bee = st.columns(2)

    shap_bar = REPORTS_DIR / f"shap_importance_{shap_horizon}h.png"
    shap_bee = REPORTS_DIR / f"shap_beeswarm_{shap_horizon}h.png"

    with col_bar:
        st.markdown(f"**Global Feature Importance — {shap_horizon}h**")
        if shap_bar.exists():
            st.image(str(shap_bar), use_container_width=True)
        else:
            st.warning(
                f"Bulunamadı: `{shap_bar}`\n\n"
                f"Çalıştır: `python analysis/shap_explainer.py --horizon {shap_horizon}`"
            )

    with col_bee:
        st.markdown(f"**Etki Yönü (Beeswarm) — {shap_horizon}h**")
        if shap_bee.exists():
            st.image(str(shap_bee), use_container_width=True)
        else:
            st.warning(f"Bulunamadı: `{shap_bee}`")

    st.markdown("---")
    st.markdown("""
**Grafik nasıl okunur?**

- **Bar plot (sol)** — her sensörün modelin kararına ortalama katkısı. En üstteki en belirleyici.  
- **Beeswarm (sağ)** — her nokta bir döküm. Kırmızı = o sensörün yüksek değeri, Mavi = düşük değer.  
  Noktanın x eksenindeki konumu Si tahminini artırıyor mu azaltıyor mu gösterir.  
""")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
if health:
    thresholds = health.get("si_thresholds", {})
    loaded     = health.get("loaded_horizons", [])
    st.caption(
        f"BF Intelligence v3 · "
        f"Modeller: {', '.join(str(h)+'h' for h in loaded)} · "
        f"Si Alert: [{thresholds.get('low', '?')} – {thresholds.get('high', '?')}] · "
        f"API: {API_URL}"
    )
else:
    st.caption(f"BF Intelligence v3 · API: {API_URL} (bağlantı yok)")
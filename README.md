# bf_intelligence

> Yüksek fırın operasyonları için çok ufuklu silisyum tahmini (Multi-Horizon Silicon Prediction).

Yüksek fırındaki sıvı demir silisyum (Si) içeriğini **2h, 4h, 6h ve 8h** ufukları için tahmin eden, üretim ortamına hazır bir veri & ML sistemi. Ham sensör verisinden canlı API tahminine kadar tüm süreç otomatize edilmiştir.

---

## Mimari Genel Bakış

```
Ham Veri (Excel/CSV)
        │
        ▼
  Kafka Streamer          ← src/bronze/kafka_streamer.py           (sensör + Si verileri Kafka topic'e akar)
        │
        ▼
  Bronze Katmanı          ← src/bronze/ingest_to_bronze.py         (Spark Structured Streaming → Parquet)
        │
        ▼
  Silver Katmanı          ← src/silver/                            (toplu temizleme, tip dönüşümü, kalite filtresi)
        │
        ▼
  Gold Katmanı            ← src/features/feature_engineering.py    (lag, diff, velocity, rolling window feature'ları)
        │
        ▼
  Model Eğitimi           ← src/models/train.py                    (LightGBM × 4 ufuk | Optuna | TimeSeriesSplit | MLflow)
        │
        ▼
  Fırın Otopsisi          ← src/analysis/furnace_autopsy.py        (IsolationForest anomali tespiti, duruş şoku analizi)
  SHAP Açıklanabilirlik   ← src/analysis/shap_explainer.py
        │
        ▼
  FastAPI Servisi         ← src/api/main.py                        (POST /predict → anlık Si tahmini + alert)
        │
        ▼
  Model İzleme            ← src/models/monitor.py                  (7 günlük rolling MAE drift tespiti → otomatik yeniden eğitim)
```

---

## Özellikler

- **Çok ufuklu tahmin** — 2h / 4h / 6h / 8h için bağımsız LightGBM modelleri; her ufuk kendi Optuna çalışmasıyla ayrı tune edilir.
- **Medallion mimarisi** — Bronze → Silver → Gold katmanlı veri akışı; Spark + Kafka ile ingest, Silver toplu (batch) işleme.
- **Online feature güncelleme** — Canlı tahminde yalnızca anlık hesaplanabilen feature'lar (lag, diff, velocity, acceleration) yeniden üretilir; ağır rolling window feature'ları veritabanından doğrudan servis edilir.
- **Otomatik drift tespiti** — Hem mutlak MAE eşiği hem de baseline'a göre %25 bozulma kriteri; drift algılandığında hafifletilmiş bir Optuna yeniden eğitim turu otomatik başlar.
- **SHAP açıklanabilirlik** — Her ufuk için bağımsız TreeExplainer çalışması; bar plot + beeswarm görsellerle hangi sensörün tahmini ne yönde etkilediği raporlanır.
- **FastAPI servisi** — `/predict` uç noktası GREEN / YELLOW / RED alert seviyeli yanıt döner; `/health` ve `/reload` ops uç noktaları dahil.
- **MLflow takibi** — Tüm eğitim koşuları için parametre, metrik, feature importance ve model artifact otomatik loglanır.
- **Duruş şoku analizi** — Uzun fırın duruşları (>50h) sonrası Si davranışı otomatik olarak raporlanır.

---

## Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| Veri akışı | Apache Kafka, Apache Spark (Structured Streaming) |
| Depolama | PostgreSQL, Parquet (Medallion / Lakehouse) |
| Makine öğrenmesi | LightGBM, Optuna, SHAP, scikit-learn |
| Deney takibi | MLflow |
| API | FastAPI, Pydantic, Uvicorn |
| Arayüz | Streamlit |
| Konfigürasyon | YAML (paths / settings / schemas) |
| Loglama | Python logging (modül bazlı dosya + konsol) |
| Konteynerizasyon | Docker, Docker Compose |

---

## Proje Yapısı

```
bf_intelligence/
├── config/
│   ├── paths.yaml          # Veri ve model dizin yolları
│   ├── settings.yaml       # DB, Kafka, Spark, MLflow, model ayarları
│   └── schemas.yaml        # Sensör veri sözleşmesi (tipler + açıklamalar)
│
├── src/
│   ├── bronze/
│   │   ├── excel_extractor.py      # Excel → CSV (ham veri ayrıştırma)
│   │   ├── kafka_streamer.py       # CSV → Kafka topic
│   │   └── ingest_to_bronze.py     # Kafka → Parquet (Spark Streaming)
│   │
│   ├── silver/
│   │   ├── silver_refiner.py       # Toplu temizleme & tip dönüşümü
│   │   └── silver_stream.py        # Spark Streaming silver katmanı
│   │
│   ├── gold/
│   │   └── gold_refiner.py         # Feature store üretimi
│   │
│   ├── features/
│   │   └── feature_engineering.py  # Lag, diff, velocity, rolling, thermal feature'lar
│   │
│   ├── models/
│   │   ├── train.py        # Çok ufuklu LightGBM eğitimi (Optuna + MLflow)
│   │   ├── predict.py      # Canlı tahmin pipeline'ı
│   │   └── monitor.py      # Drift tespiti & otomatik yeniden eğitim
│   │
│   ├── analysis/
│   │   ├── furnace_autopsy.py  # Anomali tespiti, duruş şoku analizi
│   │   └── shap_explainer.py   # SHAP feature açıklanabilirliği
│   │
│   ├── pipeline/
│   │   ├── main_pipeline.py    # Ana giriş noktası (argparse + mod dispatch)
│   │   ├── modes.py            # full / medallion / train / ui / extract modları
│   │   ├── orchestrator.py     # Medallion katman orkestrasyonu
│   │   └── analysis.py         # SHAP + autopsy pipeline adımı
│   │
│   ├── api/
│   │   └── main.py         # FastAPI uygulaması
│   │
│   ├── ui/
│   │   └── app.py          # Streamlit arayüzü
│   │
│   └── utils/
│       ├── config_loader.py
│       ├── database_manager.py
│       ├── kafka_utils.py
│       ├── spark_utils.py
│       ├── logger.py
│       ├── pipeline_utils.py
│       ├── cleanup.py
│       └── io_helper.py
│
├── data/
│   ├── raw/source/         # Ham Excel dosyaları
│   ├── bronze/             # Parquet — ham ingest
│   ├── silver/             # Parquet — temizlenmiş
│   └── gold/               # Parquet — feature store
│
├── models/                 # Eğitilmiş .joblib modelleri
├── reports/                # Feature importance, SHAP, diagnostik görseller
├── logs/
└── docker-compose.yml
```

---

## Sensör Değişkenleri

| Grup | Değişken | Açıklama |
|---|---|---|
| Besleme | `Fb` | Hava debisi (Nm³/h) |
| | `Th` / `Tc` | Sıcak / soğuk hava sıcaklığı (°C) |
| | `Fo` | Oksijen oranı |
| | `R` | Cevher/kok oranı |
| Basınç | `Ph`, `Pc`, `Pt` | Giriş / merkez / tepe basıncı (bar) |
| | `dP`, `dPu`, `dPl` | Toplam / üst / alt basınç farkı (bar) |
| Kimya | `CO2`, `H2` | Gaz bileşimi (%) |
| Termal | `Tt1–Tt4` | Üst sıcaklık profili (°C) |
| | `Tp1–Tp10` | Probe sıcaklık profili (°C) |
| Hedef | `Si` | Silisyum içeriği (%) |

---

## Kurulum

**Gereksinimler:** Docker & Docker Compose

```bash
docker-compose up -d
```

Kafka, Spark, PostgreSQL, MLflow ve uygulama container'ları ayağa kalkar.

---

## Kullanım

Tüm komutlar `bf_orchestrator` container'ı içinde çalıştırılır.

### Pipeline Modları

```bash
# Tam pipeline: extract → bronze → silver → gold → train → analiz → arayüz
docker exec bf_orchestrator python -m src.pipeline.main_pipeline

# Önceki verileri/modelleri sıfırlayıp baştan çalıştır
docker exec bf_orchestrator python -m src.pipeline.main_pipeline --fresh

# Sadece veri katmanları (bronze → silver → gold), eğitim yok
docker exec bf_orchestrator python -m src.pipeline.main_pipeline --mode medallion

# Sadece model eğitimi + SHAP/autopsy analizi
docker exec bf_orchestrator python -m src.pipeline.main_pipeline --mode train

# Eski modelleri ve raporları silip yeniden eğit (MLflow DB korunur)
docker exec bf_orchestrator python -m src.pipeline.main_pipeline --mode train --fresh

# Analiz adımlarını atlayarak sadece eğit
docker exec bf_orchestrator python -m src.pipeline.main_pipeline --mode train --skip-analysis

# Sadece Excel extraction + Kafka stream
docker exec bf_orchestrator python -m src.pipeline.main_pipeline --mode extract

# Sadece Streamlit arayüzü
docker exec bf_orchestrator python -m src.pipeline.main_pipeline --mode ui
```

### API Servisi

```bash
docker exec bf_orchestrator uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Tahmin isteği:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "horizon_hours": 4,
    "Fb": 3800.0, "Th": 1150.0, "R": 3.8,
    "dP": 1.45, "CO2": 18.5, "H2": 4.2, "Si": 0.45
  }'
```

### Pipeline Durum Kontrolü

```bash
docker exec bf_orchestrator python -m src.check_pipeline
```

---

## API Uç Noktaları

| Metot | Uç Nokta | Açıklama |
|---|---|---|
| `GET` | `/health` | Model durumu, yüklü ufuklar, Si eşikleri |
| `POST` | `/predict` | Sensör verisiyle Si tahmini + alert seviyesi |
| `POST` | `/reload` | Model dosyalarını diskten yeniden yükle |

Alert seviyeleri: `GREEN` (stabil) → `YELLOW` (±0.10 sapma bekleniyor) → `RED` (Si eşik dışına çıktı)

---

## Servis Adresleri

| Servis | Adres |
|---|---|
| Streamlit Arayüzü | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| MLflow | http://localhost:5000 |

---

## Model Performansı

Tüm modeller Optuna ile tune edilmiş, TimeSeriesSplit cross-validation ve MLflow ile takip edilmiştir.

| Ufuk | MAE | RMSE | R² | Baseline MAE | İyileşme |
|---|---|---|---|---|---|
| 2h | 0.0823 | 0.1312 | 0.577 | 0.1037 | %20.7 |
| 4h | 0.0861 | 0.1362 | 0.589 | 0.1114 | %22.7 |
| 6h | 0.0907 | 0.1454 | 0.510 | 0.1179 | %23.0 |
| 8h | 0.0923 | 0.1475 | 0.481 | 0.1212 | %23.8 |

> Baseline MAE: bir önceki Si ölçümünü tahmin olarak kullanan naive model.

**8h ufku için en baskın feature'lar** (SHAP): `Si_roll_mean_4h`, `Si`, `Si_roll_mean_8h`, `Si_roll_mean_24h`, `R_roll_mean_4h`, `month`

---

## Raporlar & Çıktılar

Her eğitim ve analiz koşusu sonunda `reports/` dizinine üretilen dosyalar:

| Dosya | İçerik |
|---|---|
| `feature_importance_{h}h.png / .csv` | Top 30 LightGBM feature importance |
| `model_diagnostics_{h}h.png` | Tahmin vs Gerçek, Artık, Hata Dağılımı, Zaman Serisi |
| `shap_importance_{h}h.png` | SHAP bar plot |
| `shap_beeswarm_{h}h.png` | SHAP beeswarm (etki yönü + büyüklüğü) |
| `furnace_autopsy.png` | Anomali haritası + duruş işaretleri |
| `model_drift_history.csv` | Zaman serisi MAE geçmişi (Grafana/PowerBI uyumlu) |
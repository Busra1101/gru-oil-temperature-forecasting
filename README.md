# GRU ile Transformatör Yağ Sıcaklığı Tahmini

Bu depo, üç kişilik bir akademik ekip projesinde geliştirdiğim **GRU bileşenini** içermektedir. Projenin genel amacı; ETTh1 zaman serisi veri seti üzerinde LSTM, GRU ve Transformer mimarilerinin performanslarını karşılaştırmaktır.

## Projedeki Katkım

- PyTorch kullanarak GRU tabanlı zaman serisi tahmin modelini geliştirdim ve eğittim.
- Geçmiş 96 saatlik veriyi kullanarak bir sonraki saatin transformatör yağ sıcaklığını (OT) tahmin ettim.
- Modeli MSE, RMSE, MAE ve R² metrikleriyle değerlendirdim.
- Eğitim ve doğrulama kayıplarını görselleştirdim.
- Gerçek ve tahmin edilen değerleri karşılaştıran grafikler oluşturdum.
- GRU deneyinin teknik dokümantasyonuna ve proje raporuna katkı sağladım.

Bu depoda bulunan veri ön işleme akışı, ekipteki modellerin aynı koşullarda karşılaştırılabilmesi amacıyla kullanılan ortak yaklaşımı yansıtmaktadır.

## Elde Edilen Sonuçlar

| Metrik | Değer |
|---|---:|
| MSE | 0.007553 |
| RMSE | 0.086907 |
| MAE | 0.064953 |
| R² | 0.9524 |
| Parametre sayısı | 39.041 |

## Kullanılan Teknolojiler

Python, PyTorch, pandas, NumPy, scikit-learn, Matplotlib ve joblib.

## Proje Yapısı

```text
.
├── data/
│   └── ETTh1.csv
├── results/
│   ├── gru_loss_pred1.png
│   ├── gru_prediction_pred1.png
│   └── metrics.txt
├── preprocessing.py
├── gru_pred1.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Kurulum ve Çalıştırma

Sanal ortam oluşturun:

```bash
python -m venv .venv
```

Windows için sanal ortamı etkinleştirin:

```bash
.venv\Scripts\activate
```

Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

Veri ön işleme adımını çalıştırın:

```bash
python preprocessing.py
```

GRU modelini eğitmek ve değerlendirmek için:

```bash
python gru_pred1.py
```

## Akademik Proje Bağlamı

Projenin tamamında LSTM, GRU ve Transformer modelleri aynı ETTh1 veri seti ve ortak veri ön işleme yaklaşımı kullanılarak karşılaştırılmıştır. Bu depo yalnızca geliştirdiğim GRU bileşenine odaklanmaktadır.


## Proje Raporu

LSTM, GRU ve Transformer mimarilerinin ETTh1 veri seti üzerindeki karşılaştırmasını, deneyleri ve elde edilen sonuçları içeren nihai proje raporuna aşağıdaki bağlantıdan ulaşabilirsiniz.

[Proje raporunu görüntüle](gru-zaman-serisi-proje-raporu.pdf)

Bu depo, akademik ekip projesinde geliştirdiğim GRU bileşenine odaklanmaktadır. Proje raporunun hazırlanması, ekip çalışmalarının bütünleştirilmesi ve nihai düzenlemesi tarafımdan gerçekleştirilmiştir.

**Hazırlayan:** Büşra Demirkıran  
**Bölüm:** Bartın Üniversitesi Bilgisayar Mühendisliği

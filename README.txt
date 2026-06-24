═════════════════════════════════════════════════════════════════
RIRSU PROJEKT - NAPOVEDOVANJE PM10 DELCEV Z GRU NEVRONSKO MREŽO
═════════════════════════════════════════════════════════════════

OPIS PROJEKTA
─────────────
Ta projekt temelji na napovedovanju koncentracije PM10 delcev v zraku
z uporabo GRU (Gated Recurrent Unit) nevronske mreže. Model je
narejen v Python-u s knjižnico TensorFlow/Keras.

NAMESTITEV
──────────

1. KLONIRAJTE REPOZITORIJ:
   git clone https://github.com/Anns2209/RIRSU_projekt.git
   cd RIRSU_projekt

2. NAMESTITE POTREBNE KNJIŽNICE:
   pip install -r requirements.txt


STRUKTURA PROJEKTA
──────────────────
├── data/                      # Podatkovne datoteke
├── notebooks/                 # Jupyter ključe (analiza in vizualizacija)
├── src/                       # Glavna koda
│   ├── model.py              # Definicija GRU modela
│   ├── preprocessing.py       # Priprava podatkov
│   └── train.py              # Učenje modela
├── artifacts/                 # Shranjeni modeli
│   └── pm10_gru_model.keras  # Naučen GRU model
├── requirements.txt           # Odvisnosti
└── README.txt                # Ta datoteka

UPORABA
───────

Napovedovanje z naučenim modelom:
   python src/predict.py --input data/test_data.csv

Učenje novega modela:
   python src/train.py --epochs 50 --batch_size 32

ZAHTEVE
───────
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn

═════════════════════════════════════════════════════════════════

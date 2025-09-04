# 🧠 Apartment Price NN (Moscow) — README

Predict apartment prices in **Moscow (RUB)** from tabular features using a compact **neural network (MLP + scikit-learn)**.
Made student-friendly: quick start in **Google Colab** or locally, clean validation, and user input mode.

---

## ✨ Features

* ✅ Works with CSV (commas **or** semicolons, commas as decimal separator supported)
* ✅ Target = **`price`** (RUB) with **log-transform** for stability
* ✅ Automatic preprocessing (impute, scale, one-hot)
* ✅ **Fast mode** for quick checks, **full mode** for best quality
* ✅ **User input** interface (via `input()` or a Python dict)
* ✅ **Lat/Lon guardrails** for Moscow to avoid garbage inputs
* ✅ Optional plots (can be **disabled**)

---

## 📦 Repository layout (suggested)

```
.
├─ data/
│  └─ dsdExcelnew.csv          # your dataset (not tracked)
├─ src/
│  ├─ train.py                 # training entrypoint (NN + fast mode)
│  ├─ predict.py               # CLI/user input prediction
│  └─ utils.py                 # CSV reader, validators, constants
├─ notebooks/
│  └─ colab_template.ipynb     # ready-to-run Colab notebook (optional)
├─ requirements.txt
└─ README.md
```

> You can keep a single script if you prefer (e.g., `model_apartment_price.py`). The commands below assume the split above; adapt names if you use a single file.

---

## 🗃️ Data format (columns)

**Required features** (example names from your file):

* `wallsMaterial` *(int code, see mapping below)*
* `floorNumber` *(int)*
* `floorsTotal` *(int)*
* `totalArea` *(float, m²)*
* `kitchenArea` *(float, m²)*
* `latitude` *(float, degrees)*
* `longitude` *(float, degrees)*
* **`price`** *(float, RUB — target)*

### Walls material codes (use in UI/help)

| Code | Material (RU)     | Material (EN) |
| ---: | ----------------- | ------------- |
|    1 | Кирпич            | Brick         |
|    2 | Панель            | Panel         |
|    3 | Монолит           | Monolith      |
|    4 | Блок              | Block         |
|    5 | Дерево            | Wood          |
|    6 | Смешанный         | Mixed         |
|    7 | Другое/не указано | Other/unknown |

---

## 🌍 Moscow coordinate guardrails

Use these bounds to validate user inputs:

* **Latitude:** `55.30 ≤ lat ≤ 56.00`
* **Longitude:** `37.30 ≤ lon ≤ 37.90`

This covers Moscow (within/near MKAD) and prevents the model from receiving out-of-city coordinates.

---

## 🔧 Installation (local)

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**`requirements.txt`**

```
scikit-learn>=1.4
pandas>=2.0
numpy>=1.24
joblib>=1.3
matplotlib>=3.7
```

> **Note (scikit-learn 1.4+):** use `OneHotEncoder(..., sparse_output=False)` (not `sparse=False`).

---

## 🚀 Quick start in Google Colab (fastest path)

1. Open **Google Colab** → *New Notebook*.
2. Install libs:

```python
!pip install scikit-learn pandas numpy joblib matplotlib
```

3. Upload your CSV:

```python
from google.colab import files
uploaded = files.upload()  # choose dsdExcelnew.csv
```

4. Paste your **training** code (full or fast mode).
5. Run **prediction** code (see below) to test user inputs.

> Full Colab snippets were provided earlier; you can paste them as-is. In Colab, training usually takes **1–5 minutes** depending on mode and data.

---

## 🏋️‍♂️ Train (local)

```bash
python src/train.py --data data/dsdExcelnew.csv --out model/price_model.pkl \
  --no-plots \
  --random-state 42
```

**Useful flags**

* `--fast` — fast baseline/NN on a subsample (seconds)
* `--no-plots` — disable matplotlib plots during training
* `--val-size 0.2` — test split size

**What you get**

* Saved model: `model/price_model.pkl`
* Console metrics: **MAE, RMSE, R²**
* (Optional) Plot: Predicted vs Actual

---

## 👤 Predict (user input)

### A) CLI with `input()` (interactive)

```python
# src/predict.py
import joblib, numpy as np, pandas as pd
import sys

MOSCOW_LAT_MIN, MOSCOW_LAT_MAX = 55.30, 56.00
MOSCOW_LON_MIN, MOSCOW_LON_MAX = 37.30, 37.90

def check_coords(lat, lon):
    if not (MOSCOW_LAT_MIN <= lat <= MOSCOW_LAT_MAX):
        sys.exit("❌ Latitude out of Moscow range (55.30–56.00)")
    if not (MOSCOW_LON_MIN <= lon <= MOSCOW_LON_MAX):
        sys.exit("❌ Longitude out of Moscow range (37.30–37.90)")

def main():
    model = joblib.load("model/price_model.pkl")  # path to your saved model
    print("Enter apartment parameters (Moscow):")

    wallsMaterial = int(input("wallsMaterial (1=brick, 2=panel, 3=monolith, 4=block, 5=wood, 6=mixed, 7=other): "))
    floorNumber  = int(input("floorNumber: "))
    floorsTotal  = int(input("floorsTotal: "))
    totalArea    = float(input("totalArea (m2): "))
    kitchenArea  = float(input("kitchenArea (m2): "))
    latitude     = float(input("latitude (e.g., 55.75): "))
    longitude    = float(input("longitude (e.g., 37.62): "))

    check_coords(latitude, longitude)

    row = pd.DataFrame([{
        "wallsMaterial": wallsMaterial,
        "floorNumber": floorNumber,
        "floorsTotal": floorsTotal,
        "totalArea": totalArea,
        "kitchenArea": kitchenArea,
        "latitude": latitude,
        "longitude": longitude
    }])

    pred_log = model.predict(row)[0]
    price = float(np.expm1(pred_log))
    print(f"\n💰 Estimated price: {price:,.2f} RUB")

if __name__ == "__main__":
    main()
```

Run:

```bash
python src/predict.py
```

### B) Programmatic (dict → price)

```python
import joblib, numpy as np, pandas as pd
model = joblib.load("model/price_model.pkl")

sample = {
  "wallsMaterial": 2, "floorNumber": 5, "floorsTotal": 12,
  "totalArea": 48.0, "kitchenArea": 9.5,
  "latitude": 55.75, "longitude": 37.62
}
pred_log = model.predict(pd.DataFrame([sample]))[0]
price = float(np.expm1(pred_log))
print("Price (RUB):", round(price, 2))
```

---

## 🔍 Training script notes

* **No plots**: pass `--no-plots` (or set `SHOW_PLOTS=False` in code).
* **Fast mode**: pass `--fast` (uses subsample & lighter NN/Ridge for instant feedback).
* The NN uses `MLPRegressor` with `early_stopping=True` and quick `RandomizedSearchCV`
* Target is log-transformed (`np.log1p(price)`), predictions are inverted via `np.expm1(...)`

---

## 🧪 Input validation (recommended)

* Areas: `totalArea > 8`, `kitchenArea > 0`, `kitchenArea <= totalArea`
* Floors: `1 ≤ floorNumber ≤ floorsTotal ≤ 100`
* Materials: `wallsMaterial ∈ {1..7}`
* Coordinates: **Moscow bounds** (see above)

You can implement these checks either in `predict.py` or upstream (form/UI).

---

## ❗ Troubleshooting

* **`TypeError: OneHotEncoder(..., sparse=...)`**
  Use `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` with scikit-learn ≥ 1.4.

* **CSV read errors / wrong delimiter**
  Try `sep=';'` first (common in RU CSV). If decimals use comma, convert with
  `df[col] = pd.to_numeric(df[col].astype(str).replace(',', '.', regex=True), errors='coerce')`.

* **Poor metrics**
  Use full mode (disable `--fast`), increase search iterations, try more hidden units, or switch to gradient boosting (XGBoost/LightGBM) for tabular data.

---

## 🔒 Notes & limitations

* Trained **only for Moscow**; other cities will produce unreliable results.
* Heavily depends on **data quality** (outliers, missing values, duplicates).
* This is an educational project; not financial advice.

---

## 📄 License

MIT 

---

## 📣 How to describe the user module (English)

> **User-friendly integration for apartment price prediction.**
> The project includes an interactive CLI for end-users to enter apartment attributes and receive an estimated price in RUB. Inputs are validated (including Moscow-specific latitude/longitude ranges) to ensure robust predictions.


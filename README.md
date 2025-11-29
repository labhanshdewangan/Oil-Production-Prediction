# Oil Production Prediction using Machine Learning

This project applies machine learning models to predict **daily oil production (Sm³/day)** using real production parameters from the **Volve oil field dataset** (released by Equinor).

The goal is to demonstrate how regression techniques can be used in petroleum engineering to estimate oil output from well data.

---

## 📌 Features
- Three trained models are included:
  - **Linear Regression**
  - **Polynomial Regression (degree 4)**
  - **Neural Network (ANN with 3 hidden layers)**
- Predictions based on production parameters such as temperature, pressure, choke size, and water volume.
- Pre-trained models are provided, so you can run predictions directly.

---

## 📂 Files in this Project
- `lin_reg.pkl` – Trained linear regression model  
- `poly_reg.pkl` – Trained polynomial regression model  
- `reg_all.pkl` – Combined regression model  
- `scaler.pkl` – Feature scaler (normalization)  
- `neural_cnn.h5` – Trained neural network model  
- `Volve production data.xlsx` – Sample dataset (Well 5351, Volve field)  
- `requirements.txt` – Python dependencies  

---

## 🚀 How to Use
1. Clone or download this repository:
   ```bash
   git clone https://github.com/cout-raj-explorer/Oil_Production_Predictor.git
   cd Oil_Production_Predictor
   ```


Install the required dependencies:
pip install -r requirements.txt
Import the models in your Python script and call the prediction functions. Example:


## 📖 Dataset
The data comes from the Volve oil field (Norwegian North Sea), publicly released by Equinor.
Input features include:

ON_STREAM_HRS

AVG_DOWNHOLE_TEMPERATURE

AVG_ANNULUS_PRESS

AVG_CHOKE_SIZE_P

AVG_WHP_P

AVG_WHT_P

DP_CHOKE_SIZE

BORE_WAT_VOL

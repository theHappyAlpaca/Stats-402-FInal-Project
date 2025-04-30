# Stats-402-Final-Project
## PM₂.₅ Forecasting Using LSTM, BiLSTM, and BiLSTM+XGBoost

### Project Description

This project implements and compares three machine learning models for short-term PM₂.₅ concentration forecasting based on multivariate air pollutant data.  
The models include a standard LSTM (Long Short-Term Memory) network using TensorFlow, a BiLSTM (Bidirectional LSTM) implemented in PyTorch, and a hybrid BiLSTM + XGBoost architecture that combines deep sequential encoding with tree-based regression.  
The dataset consists of hourly observations of major atmospheric pollutants such as SO₂, NO₂, NO, NOₓ, CO, and PM₁₀, with PM₂.₅ as the prediction target.  
All models are trained and evaluated using a consistent time-series sliding window framework and standard regression metrics.

### Setup Instructions and Dependencies

This project requires Python 3.8+ and the following libraries:

- numpy  
- pandas  
- matplotlib  
- scikit-learn  
- tensorflow  
- torch  
- xgboost  
- openpyxl  
- scipy  

To install dependencies:

```bash
pip install -r requirements.txt
```
### The structure of this project
```bash
PM25_Forecasting_Project/
├── README.md
├── data/
│   └── 2223NEWDATA.xlsx
├── main/
│   ├── train_lstm.py
│   ├── train_bilstm.py
│   └── train_bilstm_xgb.py
└── outputs/
    └── plots/
```
### How to run the code 
```bash
python main/train_lstm.py       # Train LSTM model
python main/train_bilstm.py     # Train BiLSTM model
python main/train_bilstm_xgb.py # Train BiLSTM + XGBoost model
```
### Issues and Limitations
Missing values in the dataset are interpolated using cubic splines.
BiLSTM training is executed on CPU by default.
The hybrid BiLSTM + XGBoost model may overfit and perform worse on unseen test data.
There is no built-in hyper-parameter tuning in this version.

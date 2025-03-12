# Credit Card Default Prediction using Logistic Regression  

This project uses a **Logistic Regression** model to predict whether a credit card holder will default on their payment. The dataset includes customer attributes like payment history, credit limit, and past bill statements. A **Streamlit UI** is provided for easy interaction and real-time predictions.  

## 📂 Project Structure  

```
credit_card_default/
│── data/
│   ├── default_of_credit_card_clients.csv
│── model/
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│── train_model.py       # Train the model
│── predict_ui.py        # Streamlit UI for predictions
│── requirements.txt     # Dependencies
│── .gitignore           # Ignore unnecessary files
│── README.md            # Project documentation
```

## 🔧 Installation & Setup  

### 1️⃣ Clone the Repository  

```bash
git clone https://github.com/yourusername/credit-card-default.git
cd credit-card-default
```

### 2️⃣ Create a Virtual Environment  

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```

### 3️⃣ Install Dependencies  

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model  

```bash
python train_model.py
```

- Loads and preprocesses the dataset.  
- Trains a **Logistic Regression** model using **scikit-learn**.  
- Saves the model, scaler, and feature names in the `model/` directory.  

### 5️⃣ Run the Streamlit UI  

```bash
streamlit run predict_ui.py
```

- Launches a **web-based interface** for predictions.  
- Users can paste a **comma-separated input list** and get a **default prediction**.  

## 🚀 How It Works  

1. **Model Training:** `train_model.py` reads the dataset, applies preprocessing, and trains a **Logistic Regression model**.  
2. **Feature Scaling:** StandardScaler is used to normalize features.  
3. **Prediction UI:** `predict_ui.py` takes a user-provided CSV-style input and predicts the **probability of default**.  
4. **Confidence Score:** The model outputs the likelihood of default along with the prediction.  

## 🔎 Example Input & Prediction  

Paste the following into the UI:  

```
50000,1,2,1,57,-1,0,-1,0,0,0,8617,5670,35835,20940,19146,19131,2000,36681,10000,9000,689,679,0
```

### Example Output:  

```
⚠️ Likely to Default (Confidence: 72.4%)  
```

or  

```
✅ No Default Expected (Confidence: 89.6%)  
```

## 🛠️ Technologies Used  

- **Python**  
- **scikit-learn** (Logistic Regression)  
- **pandas** (Data Handling)  
- **numpy** (Numerical Computations)  
- **Streamlit** (UI for Predictions)  
- **joblib** (Saving Model & Scaler)  

## 🔥 Future Improvements  

- Implement additional ML models like **Random Forest** or **XGBoost**.  
- Deploy as a **Flask** or **FastAPI** web service.  
- Improve feature engineering for better accuracy.  

## 🤝 Contributing  

Fork the repository, make improvements, and submit a **pull request**! 🚀

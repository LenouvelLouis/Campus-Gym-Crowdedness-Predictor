title: Campus Gym Crowd Predictor 
emoji:🏋️ 
colorFrom: blue 
colorTo: indigo 
sdk: gradio 
sdk_version: 5.0.1 
app_file: app.py 
pinned: false 
license: mit

# 🏋️ Campus Gym Crowdedness Predictor

This project uses **Machine Learning** to predict how crowded a campus gym will be at any given time.  
It includes a complete data science workflow (**EDA, Feature Engineering, Model Tuning**) and a user-friendly web interface built with **Gradio**.

---

## 📌 Project Overview

The goal is to help students plan their workouts by avoiding peak hours.  
The model analyzes historical data (hour, weekday, temperature, semester schedule, etc.) to estimate how many people will be in the gym.

### 🔑 Key Features
- **Advanced Feature Engineering**: cyclical time (Sin/Cos) + interaction features.
- **Model Comparison**: Random Forest vs. Histogram Gradient Boosting.
- **Interactive Web App**: Clean real-time Gradio interface.
- **Portable Model**: Exported as a `.pkl` file for easy deployment.

---

## 📂 Project Structure

```
├── app.py # Gradio web application
├── best_gym_crowd_model.pkl # Trained ML model (generated)
├── crowdedness-at-the-campus-gym.csv # Dataset (Kaggle)
├── Campus_Gym_Crowdedness_Prediction.ipynb # Colab training notebook
├── README.md # Documentation
└── requirements.txt # Dependencies
```
---

## 🚀 Getting Started

### ✔️ Prerequisites
You need **Python 3.8+**
```bash
python --version
```
### ✔️ Install Dependencies
```bash
pip install -r requirements.txt
```
Required packages: scikit-learn, pandas, numpy, gradio, joblib
### ✔️ Run the Web App
```bash
python app.py
```
---

## 🧠 Model Details

### 📥 Input Features

| Feature | Type | Notes |
|--------|------|-------|
| Hour of Day | 0–23 | Cyclical encoding |
| Month | Jan–Dec | Cyclical encoding |
| Day of Week | Mon–Sun | Categorical |
| Temperature | Float | Fahrenheit |
| Semester Status | Start / During / Break | Critical factor |
| Holiday | Boolean | True / False |

---

### 🔬 Feature Engineering Highlight

We improved model performance using **interaction features**:

- **`weekend_hour`**: distinguishes weekday peaks vs. quieter weekend periods.
- **`semester_temp`**: captures relationship between weather and gym visits specifically during active semester periods.

## 📝 License
This project is open-source and available for educational purposes.

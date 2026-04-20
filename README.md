# 🚢 Titanic Survival Prediction Model

A machine learning-powered web application that predicts whether a passenger would have survived the Titanic disaster based on key features like age, gender, ticket class, and more.

This project demonstrates an end-to-end ML pipeline — from data preprocessing and model training to deployment using Streamlit.

---

## 📌 Project Overview

* **Model Used:** Random Forest Classifier
* **Target Variable:** `Survived`

  * `0` → Did Not Survive
  * `1` → Survived
* **Dataset:** Titanic dataset (`titanic.csv`)

---

## ⚙️ Features Used

The model uses the following input features:

| Feature        | Description                                        |
| -------------- | -------------------------------------------------- |
| **Pclass**     | Passenger class (1 = First, 2 = Second, 3 = Third) |
| **Sex**        | Gender (0 = Male, 1 = Female)                      |
| **Age**        | Passenger's age                                    |
| **Fare**       | Ticket fare                                        |
| **Embarked**   | Port of embarkation (S=0, C=1, Q=2)                |
| **FamilySize** | Total family members onboard (`SibSp + Parch + 1`) |
| **IsAlone**    | 1 if alone, else 0                                 |

---

## 🧹 Data Preprocessing

The dataset underwent several preprocessing steps:

* **Handling Missing Values:**

  * `Age` → filled with median
  * `Embarked` → filled with mode
  * `Fare` → filled with median

* **Feature Engineering:**

  * Created `FamilySize`
  * Created `IsAlone`

* **Encoding:**

  * `Sex` and `Embarked` converted into numeric values

* **Data Cleaning:**

  * Ensured correct data types (e.g., Age as integer)

---

## 🧠 Model Details

* **Algorithm:** Random Forest Classifier
* Handles non-linear relationships effectively
* Robust to overfitting compared to single decision trees

---

## 📁 Project Structure

```
├── Titanic_model.ipynb   # Model training & preprocessing
├── titanic_model.pkl    # Trained model file
├── titanic.py           # Streamlit web app
├── titanic.csv          # Dataset
└── README.md            # Project documentation
```

---

## 🚀 Running the Application

### 1. Install dependencies

```bash
pip install streamlit pandas scikit-learn joblib
```

### 2. Run the app

```bash
streamlit run titanic.py
```

---

## 🖥️ Web App Features

* Interactive UI for entering passenger details
* Real-time prediction of survival
* Clean and simple interface using Streamlit

---

## 📊 Future Improvements

* Add model performance metrics (Accuracy, Precision, Recall)
* Display prediction probability (`predict_proba`)
* Improve UI/UX with charts and better styling
* Add feature importance visualization
* Deploy the app online (Streamlit Cloud / Render)

---

## 💡 Key Learnings

* Feature engineering plays a crucial role in model performance
* End-to-end ML projects require both modeling and deployment skills
* Simplicity in UI can still deliver powerful functionality

---

## 🧾 License

This project is for educational purposes and can be freely used and modified.

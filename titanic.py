import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Predictor", page_icon="🚢", layout="centered")

# Minimal CSS
st.markdown("""
<style>
.main {
    background-color: #0b0f19;
}

.block-container {
    padding-top: 2rem;
}

h1 {
    text-align: center;
    font-weight: 600;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #111827;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("titanic_model.pkl")

# Header
st.title("🚢 Titanic Survival Predictor")
st.caption("Clean prediction. No drama.")

st.markdown("---")

# Card container
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Class", [1,2,3])
    sex = st.radio("Gender", ["Male","Female"])
    age = st.slider("Age", 0, 80, 25)

with col2:
    fare = st.slider("Fare", 0, 500, 50)
    embarked = st.selectbox("Embarked", ["S","C","Q"])
    sibsp = st.number_input("SibSp", 0, 10)
    parch = st.number_input("Parch", 0, 10)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")

# Convert
sex_val = 1 if sex == "Female" else 0
embarked_val = {"S":0,"C":1,"Q":2}[embarked]

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# Button
if st.button("Predict"):

    input_data = pd.DataFrame({
        "Pclass":[pclass],
        "Sex":[sex_val],
        "Age":[age],
        "Fare":[fare],
        "Embarked":[embarked_val],
        "FamilySize":[family_size],
        "IsAlone":[is_alone]
    })

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction[0] == 1:
        st.success(f"Survived • {prob*100:.1f}%")
        st.progress(int(prob*100))
    else:
        st.error(f"Not Survived • {(1-prob)*100:.1f}%")
        st.progress(int((1-prob)*100))

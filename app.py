# app.py
import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import time

# Load model and label encoder
with open("model.pkl", "rb") as f:
    model, label_encoder = pickle.load(f)

st.markdown("""
    <head>
   <link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lobster&display=swap" rel="stylesheet">
    </head>
    <h1 style='text-align: center; color: white; font-size: 40px; font-family: "Lobster", cursive;'>
        🌸 Iris Flower Species Predictor
    </h1>
""", unsafe_allow_html=True)




st.markdown(
    "<h3 style='font-size: 24px;'>Enter the flower's measurements below to predict its species:</h3>",
    unsafe_allow_html=True
)


# Input sliders for each feature
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Create feature array in the correct order
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict species
prediction = model.predict(features)[0]
prediction_label = label_encoder.inverse_transform([prediction])[0]
probabilities = model.predict_proba(features)[0]

# Show prediction
st.subheader("Prediction:")
st.success(f"The predicted species is: **{prediction_label}**")

# Show prediction probabilities
st.subheader("📊 Prediction Probabilities:")
for i, class_name in enumerate(label_encoder.classes_):
    st.write(f"{class_name}: {probabilities[i]:.2f}")


# Simulated animation using st.empty()
chart_placeholder = st.empty()

for step in range(1, len(probabilities) + 1):
    animated_probs = list(probabilities[:step]) + [0] * (len(probabilities) - step)

    fig = go.Figure(go.Bar(
        x=label_encoder.classes_,
        y=animated_probs,
        marker_color=['#f4a261', '#2a9d8f', '#e76f51'],
        text=[f"{p:.2f}" for p in animated_probs],
        textposition='outside',
        hovertemplate=
            "<b>Species:</b> %{x}<br>" +
            "<b>Confidence:</b> %{y:.2%}<extra></extra>",
        width=0.6,
    ))

    fig.update_layout(
        title='xPrediction Probabilities',
        title_font=dict(size=24, ),
        xaxis_title='Flower Species',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        bargap=0.4,
        height=400,
    )

    chart_placeholder.plotly_chart(fig)
    time.sleep(0.3) 

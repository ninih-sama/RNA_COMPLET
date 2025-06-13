import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Nettoyage des valeurs problématiques
def clean_values(x_values):
    x_values = np.nan_to_num(x_values, nan=0.0, posinf=1.0, neginf=-1.0)
    x_values = np.clip(x_values, 0, 1)  # Pour la logistique, x est dans [0,1]
    return x_values

# Génération de la suite logistique
def logistic_sequence(A=4.2, x0=0.1, n_steps=500):
    x_values = np.zeros(n_steps)
    x_values[0] = x0
    for n in range(1, n_steps):
        x_values[n] = A * x_values[n-1] * (1 - x_values[n-1])
    x_values = clean_values(x_values)
    return x_values

# Entraînement modèle LSTM
def train_lstm_model(x_values, sequence_length=10, epochs=50):
    x_values = clean_values(x_values)
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x_values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(x_scaled) - sequence_length):
        X.append(x_scaled[i:i + sequence_length])
        y.append(x_scaled[i + sequence_length])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)

    return model, scaler, X, y

# Question 4.2 : affichage suite logistique
def question_4_2():
    A = 4.2
    x_values = logistic_sequence(A=A)

    st.write(f"Valeurs min/max : {np.min(x_values):.5f} / {np.max(x_values):.5f}")

    st.write("Affichage des 500 premières valeurs réelles de la suite logistique :")
    # Afficher dans un dataframe pour un scroll agréable
    import pandas as pd
    df_all = pd.DataFrame({"Valeurs réelles": np.round(x_values, 5)})
    st.dataframe(df_all, height=300)  # hauteur fixée pour scrollbar

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x_values, marker='o', linestyle='dotted', markersize=3, color='blue')
    ax.set_title(f"Suite logistique : 500 valeurs avec A={A}")
    ax.set_xlabel("n")
    ax.set_ylabel("x_n")
    ax.grid()
    st.pyplot(fig)


# Question 4.3b/c : prédiction à un pas + affichage erreurs
def question_4_3b_c():
    A = 4.2
    x_values = logistic_sequence(A=A)

    model, scaler, X, y = train_lstm_model(x_values)

    predictions = model.predict(X, verbose=0)
    predicted = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y.reshape(-1, 1))

    # Calcul erreurs
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)

    st.write(f"Erreur MSE : {mse:.6f}")
    st.write(f"Erreur MAE : {mae:.6f}")

    st.write("Comparaison des 10 premières valeurs réelles et prédites :")
    df_compare = {
        "Valeurs réelles": np.round(actual[:10].flatten(), 5),
        "Valeurs prédites": np.round(predicted[:10].flatten(), 5)
    }
    st.table(df_compare)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(actual, label='Réel')
    ax.plot(predicted, label='Prédit')
    ax.set_title("Prédiction à un pas avec apprentissage du LSTM")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

# Question 4.3d : prédiction multi-pas + affichage erreurs et comparaison valeurs
def question_4_3d():
    A = 4.2
    x_values = logistic_sequence(A=A)

    model, scaler, X, y = train_lstm_model(x_values)

    horizons = [3, 10, 20]
    input_seq = X[-1].copy()  # Copier pour éviter modification accidentelle

    for n_future in horizons:
        predictions = []

        seq = input_seq.copy()
        for _ in range(n_future):
            pred = model.predict(seq.reshape(1, -1, 1), verbose=0)
            predictions.append(pred[0][0])
            seq = np.append(seq[1:], [[pred[0][0]]], axis=0)

        predictions = np.array(predictions).reshape(-1, 1)
        predictions_inv = scaler.inverse_transform(predictions)
        predictions_inv = np.clip(np.nan_to_num(predictions_inv, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)

        real_future = x_values[-n_future:]
        real_future = np.nan_to_num(real_future, nan=0.0, posinf=1.0, neginf=0.0)

        mse = mean_squared_error(real_future, predictions_inv.flatten())
        mae = mean_absolute_error(real_future, predictions_inv.flatten())

        st.write(f"### Prédiction multi-pas : {n_future} pas")
        st.write(f"Erreur MSE : {mse:.6f}")
        st.write(f"Erreur MAE : {mae:.6f}")

        # Affichage comparaison 10 premières valeurs réelles/prédites (ou moins si horizon < 10)
        nb_affiche = min(10, n_future)
        st.write(f"Comparaison des {nb_affiche} premières valeurs réelles futures et prédites :")
        df_compare = {
            "Valeurs réelles": np.round(real_future[:nb_affiche], 5),
            "Valeurs prédites": np.round(predictions_inv[:nb_affiche].flatten(), 5)
        }
        st.table(df_compare)

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(real_future, label='Réel')
        ax.plot(predictions_inv, label=f'Prévision {n_future} pas')
        ax.set_title(f"Prédiction multi-pas ({n_future} pas futurs)")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

# Interface Streamlit unique
st.title("Simulation Logistique et Prédiction architecture LSTM avec A=4.2")

page = st.radio("Choisissez une question :", [
    "Question 4.2",
    "Question 4.3b / 4.3c",
    "Question 4.3d"
])

if page == "Question 4.2":
    question_4_2()
elif page == "Question 4.3b / 4.3c":
    question_4_3b_c()
elif page == "Question 4.3d":
    question_4_3d()

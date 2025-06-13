import numpy as np
import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2

# Génération des données logistiques
def generate_logistic_data(A=2, x0=0.1, n=500):
    x = [x0]
    for _ in range(n - 1):
        next_val = A * x[-1] * (1 - x[-1])
        next_val = np.clip(next_val, 0, 1)  # Empêche les dépassements
        x.append(next_val)
    return np.array(x)

# Préparation des séquences pour LSTM
def prepare_sequences(data, seq_len=20):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    X = np.array(X).reshape((-1, seq_len, 1))
    y = np.array(y)
    return X, y

# Création et entraînement du modèle
def train_model(X, y):
    model = Sequential([
        Bidirectional(LSTM(150, activation='tanh', return_sequences=True, input_shape=(X.shape[1], 1), kernel_regularizer=l2(0.002))),
        Dropout(0.3),
        Bidirectional(LSTM(100, activation='tanh', kernel_regularizer=l2(0.002))),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=RMSprop(learning_rate=5e-5), loss='mse')
    model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2, verbose=0)
    return model

# Question 2 - Affichage des 500 valeurs
# Question 2 - Affichage des 500 valeurs
def question_2():
    st.subheader("Question 2 : Les 500 premières valeurs de la suite logistique (A = 2, x₀ = 0.1)")

    # Génération des données
    A = 2
    x0 = 0.1
    n = 500
    x_values = [x0]
    for _ in range(n - 1):
        next_val = A * x_values[-1] * (1 - x_values[-1])
        next_val = np.clip(next_val, 0, 1)
        x_values.append(next_val)
    
    x_array = np.array(x_values)

    # Affichage statistique
    st.write(f"Valeur min : {x_array.min():.8f}")
    st.write(f"Valeur max : {x_array.max():.8f}")
    st.write(f"Valeur moyenne : {x_array.mean():.8f}")

    # Graphique
    st.line_chart(pd.DataFrame(x_array, columns=["xₙ"]))

    # Tableau
    df = pd.DataFrame({"n": np.arange(1, n + 1), "xₙ": x_array})
    st.dataframe(df.style.format({"xₙ": "{:.8f}"}), use_container_width=True)

# Prédiction à un pas (3b & 3c)
def question_3bc():
    st.subheader("Question 3b & 3c :Apprentissage du reseau et Prédiction à un pas")
    data = generate_logistic_data()
    X, y = prepare_sequences(data)
    model = train_model(X, y)

    # Prédiction de 10 valeurs
    index = len(data) - 10 - X.shape[1]
    input_seq = data[index:index+X.shape[1]].reshape(1, X.shape[1], 1)
    true_values = data[index + X.shape[1]:index + X.shape[1] + 10]

    predictions = []
    for _ in range(10):
        pred = model.predict(input_seq, verbose=0)[0, 0]
        pred = np.clip(pred, 0, 1)
        predictions.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    df = pd.DataFrame({"Valeur réelle": true_values, "Valeur prédite": predictions})
    st.dataframe(df.style.format("{:.8f}"))

    st.line_chart(df)

    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    st.success(f"Erreur quadratique moyenne (MSE) : {mse:.8f}")
    st.info(f"Erreur absolue moyenne (MAE) : {mae:.8f}")

# Prédiction multi-pas (3d)
def question_3d():
    st.subheader("Question 3d : Prédictions à plusieurs pas")
    data = generate_logistic_data()
    X, y = prepare_sequences(data)
    model = train_model(X, y)

    for pas in [3, 10, 20]:
        st.markdown(f"### 🔮 Prédiction à {pas} pas")
        index = len(data) - pas - X.shape[1]
        input_seq = data[index:index+X.shape[1]].reshape(1, X.shape[1], 1)
        true_values = data[index + X.shape[1]: index + X.shape[1] + pas]

        predictions = []
        for _ in range(pas):
            pred = model.predict(input_seq, verbose=0)[0, 0]
            pred = np.clip(pred, 0, 1)
            predictions.append(pred)
            input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

        df = pd.DataFrame({"Valeur réelle": true_values, "Valeur prédite": predictions})
        st.dataframe(df.style.format("{:.8f}"))
        st.line_chart(df)

        mse = mean_squared_error(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        st.success(f"MSE : {mse:.8f}")
        st.info(f"MAE : {mae:.8f}")

# Interface principale Streamlit
st.title("📈 Projet RNA - Suite logistique  Utilisation de LSTM avec A=2")
choix = st.radio("Sélectionnez une question :", ["Question 2", "Question 3b & 3c", "Question 3d"])

if choix == "Question 2":
    question_2()
elif choix == "Question 3b & 3c":
    question_3bc()
elif choix == "Question 3d":
    question_3d()

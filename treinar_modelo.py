import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --- 1. Hiperparâmetros e Configurações ---
# Tamanho da janela de tempo (quantos passos no tempo vamos usar para prever o próximo)
LOOK_BACK = 30

# Parâmetros de treinamento
EPOCHS = 5
BATCH_SIZE = 128

# Nomes dos arquivos
TRAIN_FILE = 'Data/dados_treino.csv'
VALIDATION_FILE = 'Data/dados_validacao.csv'
MODEL_FILE = 'modelo_ethereum.h5' # O Keras salva modelos no formato .h5

# --- 2. Carregar e Preparar os Dados ---
print("Carregando dados de treino e validação...")
df_train = pd.read_csv(TRAIN_FILE)
df_val = pd.read_csv(VALIDATION_FILE)

# Vamos usar apenas a coluna 'close' para a previsão
train_data = df_train['close'].values.reshape(-1, 1)
val_data = df_val['close'].values.reshape(-1, 1)

# Normalização dos dados
# É CRUCIAL treinar o scaler APENAS com os dados de treino para evitar vazamento de dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_val_data = scaler.transform(val_data) # Usa o mesmo scaler para transformar a validação

# --- 3. Função para Criar Sequências ---
# Esta função transforma uma lista de preços em pares de input/output para o modelo
def create_sequences(data, look_back=LOOK_BACK):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

print("Criando sequências de treino e validação...")
X_train, y_train = create_sequences(scaled_train_data)
X_val, y_val = create_sequences(scaled_val_data)

# O LSTM espera uma entrada no formato [amostras, timesteps, features]
# Então, precisamos remodelar os dados
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

print(f"Formato dos dados de treino: {X_train.shape}")
print(f"Formato dos dados de validação: {X_val.shape}")

# --- 4. Construção do Modelo LSTM ---
print("\nConstruindo o modelo LSTM...")
model = Sequential()

# Camada LSTM 1 com Dropout para evitar overfitting
model.add(LSTM(units=50, return_sequences=True, input_shape=(LOOK_BACK, 1)))
model.add(Dropout(0.2))

# Camada LSTM 2
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Camada de Saída (Dense)
# A saída é 1, pois queremos prever um único valor (o próximo preço de 'close')
model.add(Dense(units=1))

# Compilação do modelo
# Usamos 'adam' como otimizador e 'mean_squared_error' como função de perda para regressão
model.compile(optimizer='adam', loss='mean_squared_error')

# Mostra um resumo da arquitetura do modelo
model.summary()

# --- 5. Treinamento do Modelo ---
print("\nIniciando o treinamento do modelo...")

# Callbacks para melhorar o treinamento:
# EarlyStopping: para o treino se a perda na validação não melhorar após N épocas
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# ModelCheckpoint: salva o melhor modelo observado durante o treino
model_checkpoint = ModelCheckpoint(MODEL_FILE, monitor='val_loss', save_best_only=True)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

print("\nTreinamento concluído!")
print(f"O melhor modelo foi salvo em '{MODEL_FILE}'.")

# --- 6. Visualizar o Histórico de Treinamento ---
print("Gerando gráfico do histórico de perdas...")
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Perda no Treino (loss)')
plt.plot(history.history['val_loss'], label='Perda na Validação (val_loss)')
plt.title('Histórico de Perda do Modelo')
plt.xlabel('Época')
plt.ylabel('Perda (Mean Squared Error)')
plt.legend()
plt.savefig('historico_perda_modelo.png')
plt.show()
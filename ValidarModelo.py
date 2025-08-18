import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# --- 1. Configurações ---
# Nomes dos arquivos
TRAIN_FILE = 'Data/dados_treino.csv'
VALIDATION_FILE = 'Data/dados_validacao.csv'
TEST_FILE = 'Data/dados_teste.csv'
MODEL_FILE = 'modelo_ethereum.h5'

# O LOOK_BACK deve ser o MESMO usado no treinamento
LOOK_BACK = 30

# --- 2. Carregar o Modelo e os Dados ---
print("Carregando o modelo treinado...")
try:
    model = load_model(MODEL_FILE)
except IOError:
    print(f"Erro: O arquivo do modelo '{MODEL_FILE}' não foi encontrado. Execute o script de treinamento primeiro.")
    exit()

print("Carregando dados de teste e treino...")
try:
    df_test = pd.read_csv(TEST_FILE)
    df_train = pd.read_csv(TRAIN_FILE)
except FileNotFoundError as e:
    print(f"Erro: Arquivo de dados não encontrado. Detalhes: {e}")
    exit()

# --- 3. Preparar os Dados de Teste ---
# Pega a coluna 'close' dos dados de teste
test_data = df_test['close'].values.reshape(-1, 1)

# **PASSO CRUCIAL:** Usamos o scaler ajustado (fitted) nos dados de TREINO para transformar os dados de TESTE.
# Isso simula o cenário real onde não conhecemos o futuro (os dados de teste).
scaler = MinMaxScaler(feature_range=(0, 1))
# Ajusta o scaler com os dados de treino
scaler.fit(df_train['close'].values.reshape(-1, 1)) 
# Transforma os dados de teste
scaled_test_data = scaler.transform(test_data)

# Função para criar as sequências (a mesma do script de treino)
def create_sequences(data, look_back=LOOK_BACK):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

print("Criando sequências de teste...")
X_test, y_test = create_sequences(scaled_test_data)

# Remodelar os dados para o formato esperado pelo LSTM [amostras, timesteps, features]
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# --- 4. Fazer as Previsões ---
print("Realizando previsões com os dados de teste...")
predicted_prices_scaled = model.predict(X_test)

# --- 5. Reverter a Normalização ---
# Agora, transformamos os preços previstos (e os reais) de volta para a escala original (dólares)
predicted_prices = scaler.inverse_transform(predicted_prices_scaled)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

print("Previsões revertidas para a escala original.")

# --- 6. Avaliar o Modelo ---
# Calcular o RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
print(f"\nRoot Mean Squared Error (RMSE) nos dados de teste: {rmse:.4f}")
# O RMSE te dá uma ideia do erro médio do modelo em dólares. Um RMSE de 50 significa que, em média,
# a previsão do modelo erra por $50.

# Visualizar os resultados
print("Gerando gráfico de comparação...")
plt.figure(figsize=(16, 8))
plt.plot(real_prices, color='blue', label='Preço Real do Ethereum')
plt.plot(predicted_prices, color='red', label='Preço Previsto pelo Modelo', alpha=0.7)
plt.title('Comparação: Preço Real vs. Preço Previsto')
plt.xlabel('Tempo (em minutos, a partir do início do conjunto de teste)')
plt.ylabel('Preço do Ethereum (USD)')
plt.legend()
plt.savefig('validacao_modelo_plot.png')
plt.show()
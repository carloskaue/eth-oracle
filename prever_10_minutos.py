import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time

# --- 1. Configurações ---
LOOK_BACK = 30
FORECAST_HORIZON = 10 # Quantos minutos à frente queremos prever

# Nomes dos arquivos
TRAIN_FILE = 'Data/dados_treino.csv'
VALIDATION_FILE = 'Data/dados_validacao.csv'
TEST_FILE = 'Data/dados_teste.csv'
MODEL_FILE = 'modelo_ethereum.h5'

# --- 2. Carregar o Modelo e Preparar o Scaler ---
print("Carregando o modelo e preparando o normalizador...")
try:
    model = load_model(MODEL_FILE)
    df_train = pd.read_csv(TRAIN_FILE)
except Exception as e:
    print(f"Erro ao carregar arquivos: {e}")
    exit()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_train['close'].values.reshape(-1, 1))

# --- 3. Buscar Dados Iniciais e os Dados Reais Futuros ---
points_to_fetch = LOOK_BACK + FORECAST_HORIZON
print(f"Buscando os últimos {points_to_fetch} minutos de dados da Binance...")
try:
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('ETH/USDT', '1m', limit=points_to_fetch + 5)
    
    df_live = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_live = df_live.tail(points_to_fetch)

    if len(df_live) < points_to_fetch:
        print("Não foi possível obter dados suficientes. Tente novamente.")
        exit()
except Exception as e:
    print(f"Erro ao buscar dados da Binance: {e}")
    exit()

# Separa a sequência inicial (para começar a prever) e os valores reais (para comparar)
initial_sequence = df_live.head(LOOK_BACK)
actual_future_prices = df_live.tail(FORECAST_HORIZON)

print("Dados obtidos. Iniciando o processo de previsão autorregressiva...")

# --- 4. Processo de Previsão Autorregressiva ---
# Normaliza a sequência inicial
current_sequence_scaled = scaler.transform(initial_sequence['close'].values.reshape(-1, 1))

# Lista para armazenar as previsões
future_predictions_scaled = []

for i in range(FORECAST_HORIZON):
    # Prepara a sequência para o formato do LSTM
    X_predict = np.reshape(current_sequence_scaled, (1, LOOK_BACK, 1))
    
    # Faz a previsão de 1 passo à frente
    next_prediction_scaled = model.predict(X_predict, verbose=0)
    
    # Armazena a previsão
    future_predictions_scaled.append(next_prediction_scaled[0][0])
    
    # ** A Mágica Autorregressiva **
    # Atualiza a sequência: remove o primeiro valor e adiciona a nova previsão no final
    current_sequence_scaled = np.append(current_sequence_scaled[1:], next_prediction_scaled, axis=0)

# --- 5. Pós-processamento e Comparação ---
# Reverte a normalização das previsões para a escala de dólares
predicted_prices = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))

# Pega os preços reais para comparação
real_prices = actual_future_prices['close'].values

# --- 6. Mostrar Resultados e Plotar Gráfico ---
print("\n" + "="*60)
print("       Comparação: Preços Reais vs. Previsões para 10 Minutos")
print("="*60)

comparison_df = pd.DataFrame({
    'Timestamp': pd.to_datetime(actual_future_prices['timestamp'], unit='ms'),
    'Preço Real': real_prices,
    'Preço Previsto': predicted_prices.flatten() # .flatten() para transformar em 1D array
})
print(comparison_df)
print("="*60)

# Plotar o gráfico
print("Gerando gráfico de comparação...")
plt.figure(figsize=(16, 8))
# Index para o eixo X
time_index = np.arange(len(real_prices))

plt.plot(time_index, real_prices, 'bo-', label='Preço Real do Ethereum', markersize=5)
plt.plot(time_index, predicted_prices, 'ro-', label='Preço Previsto pelo Modelo', alpha=0.7, markersize=5)

plt.title('Previsão Autorregressiva para os Próximos 10 Minutos')
plt.xlabel('Minutos no Futuro')
plt.ylabel('Preço do Ethereum (USD)')
plt.xticks(time_index, labels=[f't+{i+1}' for i in time_index]) # Rótulos como t+1, t+2...
plt.legend()
plt.grid(True)
plt.savefig('previsao_10_minutos_plot.png')
plt.show()
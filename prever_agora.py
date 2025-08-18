import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import time

# --- 1. Configurações ---
# O LOOK_BACK DEVE ser o mesmo usado no treinamento
LOOK_BACK = 30 

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
    print(f"Erro ao carregar arquivos necessários: {e}")
    print("Certifique-se que os arquivos 'modelo_ethereum.h5' e 'dados_treino.csv' estão na pasta.")
    exit()

# Ajusta o scaler com os dados de TREINO para garantir consistência
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_train['close'].values.reshape(-1, 1))

# --- 3. Buscar os Dados Mais Recentes da Binance ---
# Precisamos de 31 pontos: 30 para a entrada (input) e o 31º como o valor real a ser comparado.
points_to_fetch = LOOK_BACK + 1
print(f"\nBuscando os últimos {points_to_fetch} minutos de dados do Ethereum na Binance...")
try:
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('ETH/USDT', '1m', limit=points_to_fetch + 5) # Pega um pouco a mais por segurança

    # Garante que temos exatamente o número de pontos necessários
    df_live = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_live = df_live.tail(points_to_fetch)

    if len(df_live) < points_to_fetch:
        print(f"Erro: A API retornou apenas {len(df_live)} pontos. Tente novamente em um minuto.")
        exit()

except Exception as e:
    print(f"Erro ao buscar dados da Binance: {e}")
    exit()

print("Dados recentes obtidos com sucesso.")

# --- 4. Separar Dados de Entrada e o Valor Real ---
# Os primeiros 30 pontos são a nossa sequência de entrada para o modelo
input_df = df_live.iloc[:-1] 

# O último ponto (o 31º) é o nosso "alvo", o valor real que queremos comparar
actual_df = df_live.iloc[-1:]

# Extrair os valores de 'close'
input_data_close = input_df['close'].values.reshape(-1, 1)
actual_price_real = actual_df['close'].iloc[0]
actual_time = pd.to_datetime(actual_df['timestamp'].iloc[0], unit='ms')

# --- 5. Pré-processar, Prever e Pós-processar ---
# Normaliza a sequência de entrada
scaled_input_data = scaler.transform(input_data_close)

# Remodela para o formato do LSTM [1, look_back, 1]
X_predict = np.reshape(scaled_input_data, (1, LOOK_BACK, 1))

# Faz a previsão
print("Realizando a previsão com base nos 30 minutos anteriores...")
predicted_price_scaled = model.predict(X_predict)

# Reverte a previsão para a escala de dólares
predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

# --- 6. Comparar e Mostrar o Resultado ---
diferenca_abs = predicted_price - actual_price_real
diferenca_perc = (diferenca_abs / actual_price_real) * 100

print("\n" + "="*50)
print("       VERIFICAÇÃO DA ÚLTIMA PREVISÃO")
print("="*50)
print(f"Dados de entrada:       Últimos 30 min antes de {actual_time}")
print(f"Previsão para o minuto: {actual_time}")
print("-"*50)
print(f"Preço Real:             ${actual_price_real:.2f}")
print(f"Preço Previsto:         ${predicted_price:.2f}")
print("-"*50)
print(f"Diferença Absoluta:     ${diferenca_abs:.2f}")
print(f"Diferença Percentual:   {diferenca_perc:.4f}%")
print("="*50)
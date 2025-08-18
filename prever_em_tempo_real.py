import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta

# --- 1. Configurações ---
LOOK_BACK = 30
RUN_DURATION_MINUTES = 10

# Nomes dos arquivos
TRAIN_FILE = 'Data/dados_treino.csv'
VALIDATION_FILE = 'Data/dados_validacao.csv'
TEST_FILE = 'Data/dados_teste.csv'
MODEL_FILE = 'modelo_ethereum.h5'

# --- 2. Função de Previsão (igual à anterior) ---
def fazer_previsao(model, scaler):
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv('ETH/USDT', '1m', limit=LOOK_BACK + 5)
        df_live = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_live = df_live.tail(LOOK_BACK)

        if len(df_live) < LOOK_BACK:
            return None, None, None

        last_known_price = df_live['close'].iloc[-1]
        last_known_time = pd.to_datetime(df_live['timestamp'].iloc[-1], unit='ms')

        live_data_close = df_live['close'].values.reshape(-1, 1)
        scaled_live_data = scaler.transform(live_data_close)
        X_predict = np.reshape(scaled_live_data, (1, LOOK_BACK, 1))

        predicted_price_scaled = model.predict(X_predict, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

        return last_known_price, last_known_time, predicted_price
    except Exception as e:
        print(f"  Erro durante a previsão: {e}")
        return None, None, None

# --- 3. Script Principal com Gráfico ---
if __name__ == "__main__":
    print("Carregando o modelo e preparando o normalizador...")
    try:
        model = load_model(MODEL_FILE)
        df_train = pd.read_csv(TRAIN_FILE)
    except Exception as e:
        print(f"Erro fatal ao carregar arquivos: {e}")
        exit()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_train['close'].values.reshape(-1, 1))
    
    # --- Configuração do Gráfico Interativo ---
    plt.ion() # LIGA o modo interativo
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Listas para armazenar o histórico de preços da sessão
    timestamps_history = []
    real_prices_history = []
    predicted_prices_history = []
    
    # Inicia a sessão de previsão
    start_time = time.time()
    end_time = start_time + (RUN_DURATION_MINUTES * 60)
    
    print("\n" + "="*50)
    print(f"Iniciando sessão com gráfico por {RUN_DURATION_MINUTES} minutos.")
    print("="*50 + "\n")

    while time.time() < end_time:
        now = datetime.now()
        
        # 1. Faz a previsão para o próximo minuto
        last_price, last_time, prediction = fazer_previsao(model, scaler)
        
        if prediction is not None:
            prediction_time = last_time + timedelta(minutes=1)
            print(f"({now.strftime('%H:%M:%S')}) PREVISÃO para {prediction_time.strftime('%H:%M:%S')}: ${prediction:.2f}")

            # 2. Aguarda o minuto da previsão passar para buscar o preço real
            seconds_to_wait = 62 - datetime.now().second
            if time.time() + seconds_to_wait > end_time: break
            print(f"  Aguardando {seconds_to_wait}s para buscar o preço real...")
            time.sleep(seconds_to_wait)
            
            # 3. Busca o preço real que acabou de se formar
            try:
                exchange = ccxt.binance()
                latest_candle = exchange.fetch_ohlcv('ETH/USDT', '1m', limit=1)
                actual_price = latest_candle[0][4] # O 4º índice é o 'close'
                actual_time = pd.to_datetime(latest_candle[0][0], unit='ms')

                print(f"  PREÇO REAL às {actual_time.strftime('%H:%M:%S')}: ${actual_price:.2f}\n")

                # 4. Adiciona os dados ao histórico para plotagem
                timestamps_history.append(actual_time)
                real_prices_history.append(actual_price)
                predicted_prices_history.append(prediction)

                # 5. Atualiza o gráfico
                ax.clear() # Limpa o gráfico anterior
                ax.plot(timestamps_history, real_prices_history, 'bo-', label='Preço Real', markersize=5)
                ax.plot(timestamps_history, predicted_prices_history, 'ro-', label='Preço Previsto', alpha=0.7, markersize=5)
                
                # Formatação
                ax.set_title(f"Previsão em Tempo Real (Última Atualização: {datetime.now().strftime('%H:%M:%S')})")
                ax.set_ylabel('Preço (USD)')
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=30)
                plt.tight_layout()
                
                # Redesenha o gráfico
                fig.canvas.draw()
                fig.canvas.flush_events()
                
            except Exception as e:
                print(f"  Erro ao buscar preço real: {e}\n")

        else:
            # Se a previsão falhar, espera um minuto para tentar de novo
            time.sleep(60)

    print("="*50)
    print("Sessão de previsão concluída.")
    print("="*50)

    # Salva o gráfico final
    plt.savefig('previsao_final_plot.png')
    plt.ioff() # DESLIGA o modo interativo
    plt.show() # Mostra o gráfico final e bloqueia o script até fechar
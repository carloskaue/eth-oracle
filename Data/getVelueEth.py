import ccxt
import pandas as pd
import time

# --- Parte 1: Configuração e Coleta de Dados para 2017 ---
hora_inicial = 'T00:00:00Z'
hora_final = 'T23:59:59Z'

anos =[2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
meses= {'Janeiro':1, 'Fevereiro':2, 'Março':3, 'Abril':4, 'Maio':5, 'Junho':6,
        'Julho':7, 'Agosto':8, 'Setembro':9, 'Outubro':10, 'Novembro':11, 'Dezembro':12}
dias = {'Janeiro':31, 'Fevereiro':28, 'Março':31, 'Abril':30, 'Maio':31, 'Junho':30,
        'Julho':31, 'Agosto':31, 'Setembro':30, 'Outubro':31, 'Novembro':30, 'Dezembro':31} 


# Instancia a corretora
exchange = ccxt.binance()
# Define o símbolo e o timeframe
symbol = 'ETH/USDT'
timeframe = '1m'
limit = 1000 # Limite de dados por requisição da API

# Define a data de início (1 de janeiro de 2017)
# Define a data de término (hoje)
for ano in anos:
    # for mes in meses.keys():
        # since = exchange.parse8601('2025-08-01T00:00:00Z')
        since = exchange.parse8601(f'{ano}-{meses["Janeiro"]:02d}-01{hora_inicial}')
        until = exchange.parse8601(f'{ano}-{meses['Dezembro']:02d}-31{hora_final}')
        print(since, until)
        all_ohlcv = []
        print(f"Iniciando a coleta de dados da Binance para o ano de {ano}.")
        while since < until:
            try:
                # Busca os dados a partir do 'since'
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)

                # Se a API não retornar mais dados, o loop para
                if not ohlcv:
                    break

                # Filtra os dados para garantir que não ultrapassem a data final
                ohlcv_filtered = [candle for candle in ohlcv if candle[0] <= until]
                all_ohlcv.extend(ohlcv_filtered)

                # Atualiza o 'since' para a próxima requisição
                since = ohlcv[-1][0] + 1

                # Imprime o progresso
                last_date = pd.to_datetime(ohlcv_filtered[-1][0], unit='ms')
                print(f"Buscando... {len(all_ohlcv)} candles coletados. Última data: {last_date}")

            except Exception as e:
                print(f"Ocorreu um erro: {e}. O script vai pausar por 10 segundos e tentar novamente.")
                time.sleep(10)

        print(f"\nColeta de dados de {ano} finalizada!")

        # Converte para DataFrame do Pandas, se dados foram coletados
        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # --- Parte 2: Salvando o DataFrame em Disco ---

            # Define o nome do arquivo para refletir o conteúdo
            filename = 'eth_usdt_1m_data_'+str(ano)+'.csv'

            # Salva o DataFrame no arquivo CSV
            print(f"\nSalvando os dados no arquivo '{filename}'...")
            df.to_csv(filename, index=False)

            print(f"DataFrame com {len(df)} linhas salvo com sucesso!")
        else:
            print("Nenhum dado foi coletado para o período especificado.")
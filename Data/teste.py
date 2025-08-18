import pandas as pd

# Nome do seu arquivo CSV
filename = 'ethereum_close_and_delta.csv'

try:
    # Lê o arquivo CSV para um DataFrame
    df = pd.read_csv(filename)

    # Usa o método .tail(1) para pegar a última linha
    ultima_linha_df = df.tail(1)

    print("--- Última linha como DataFrame do Pandas ---")
    print(ultima_linha_df)

    # Se você quiser a linha como um dicionário (muito útil)
    ultima_linha_dict = ultima_linha_df.to_dict('records')[0]
    print("\n--- Última linha como Dicionário ---")
    print(ultima_linha_dict)

    # Para acessar um valor específico, como o 'close'
    ultimo_close = ultima_linha_dict['close']
    print(f"\nValor de fechamento da última linha: {ultimo_close}")

except FileNotFoundError:
    print(f"Erro: O arquivo '{filename}' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro: {e}")
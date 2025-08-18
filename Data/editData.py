import pandas as pd
import glob
import os

# --- 1. Configuração ---
# Padrão para encontrar seus arquivos.
# Ajuste se os nomes forem diferentes (ex: 'dados_eth_*.csv')
file_pattern = '*.csv'

# Nome do arquivo de saída
output_filename = 'ethereum_close_and_delta.csv'

# --- 2. Encontrar e Ler os Arquivos ---
# Pega o caminho de todos os arquivos que correspondem ao padrão na pasta atual
all_files = glob.glob(file_pattern)

# Remove o próprio arquivo de saída da lista, caso ele já exista e o script seja rodado novamente
if output_filename in all_files:
    all_files.remove(output_filename)

print(f"Encontrados {len(all_files)} arquivos para processar:")
print(all_files)

# Cria uma lista para armazenar os DataFrames de cada arquivo
df_list = []

# Loop para ler cada arquivo
for filename in all_files:
    print(f"Lendo o arquivo: {filename}...")
    try:
        df = pd.read_csv(filename)
        # Verifica se as colunas necessárias existem no arquivo
        if 'timestamp' in df.columns and 'close' in df.columns:
            df_list.append(df)
        else:
            print(f"  Aviso: O arquivo '{filename}' não contém as colunas 'timestamp' e 'close' e será ignorado.")
    except Exception as e:
        print(f"  Erro ao ler o arquivo '{filename}': {e}")


# --- 3. Combinar e Processar os Dados ---
if not df_list:
    print("\nNenhum dado válido foi carregado. Saindo do script.")
else:
    # Combina todos os DataFrames da lista em um só
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"\nTotal de {len(combined_df)} linhas combinadas.")

    # Mantém apenas as colunas de interesse
    processed_df = combined_df[['timestamp', 'close']].copy()

    # Garante que a coluna 'timestamp' está no formato de data correto
    processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])

    # ** PASSO CRUCIAL: Ordena os valores pela data/hora **
    # Sem isso, o cálculo do 'delta' estaria incorreto.
    processed_df.sort_values(by='timestamp', inplace=True)
    
    # Remove duplicatas, mantendo a primeira ocorrência
    processed_df.drop_duplicates(subset='timestamp', keep='first', inplace=True)

    # --- 4. Calcular a Coluna 'delta' ---
    # A função .shift(1) pega o valor da linha anterior
    previous_close = processed_df['close'].shift(1)
    
    # Fórmula: (valor_atual - valor_anterior) / valor_anterior
    processed_df['delta'] = (processed_df['close'] - previous_close) / previous_close
    
    # O primeiro valor da coluna 'delta' será NaN (Not a Number), pois não há linha anterior.
    # Isso é esperado e correto. Podemos preencher com 0 se desejado.
    processed_df['delta'].fillna(0, inplace=True)
    
    print("\nColuna 'delta' calculada com sucesso.")

    # --- 5. Salvar o Resultado ---
    processed_df.to_csv(output_filename, index=False)

    print(f"\nArquivo final '{output_filename}' criado com {len(processed_df)} linhas.")
    print("Processo concluído!")
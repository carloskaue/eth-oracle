import pandas as pd
import numpy as np

# --- 1. Configuração ---
# Nome do arquivo de entrada que você criou no passo anterior
input_filename = 'ethereum_close_and_delta.csv'

# Nomes dos arquivos de saída
train_filename = 'dados_treino.csv'
validation_filename = 'dados_validacao.csv'
test_filename = 'dados_teste.csv'

# Proporções da divisão
train_split_ratio = 0.8
validation_split_ratio = 0.1
# O restante será para teste (0.1 ou 10%)

# --- 2. Ler o Arquivo de Dados ---
try:
    print(f"Lendo o arquivo de dados: '{input_filename}'...")
    df = pd.read_csv(input_filename)
except FileNotFoundError:
    print(f"Erro: O arquivo '{input_filename}' não foi encontrado. Execute o script anterior para criá-lo.")
    exit()

# Garante que os dados estejam ordenados por tempo antes de dividir
# O script anterior já fez isso, mas é uma boa prática garantir
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values(by='timestamp', inplace=True)
df.reset_index(drop=True, inplace=True) # Reseta o índice após a ordenação

print(f"Total de {len(df)} linhas de dados carregadas e ordenadas.")

# --- 3. Calcular os Pontos de Divisão ---
# Calcula o índice onde os dados de treino terminam
train_end_index = int(len(df) * train_split_ratio)

# Calcula o índice onde os dados de validação terminam
validation_end_index = train_end_index + int(len(df) * validation_split_ratio)

print(f"\nDividindo os dados:")
print(f" - Dados de Treino:      linhas 0 até {train_end_index}")
print(f" - Dados de Validação:   linhas {train_end_index} até {validation_end_index}")
print(f" - Dados de Teste:       linhas {validation_end_index} até {len(df)}")

# --- 4. Realizar a Divisão ---
# Pega todas as linhas do início até o 'train_end_index'
df_train = df.iloc[0:train_end_index]

# Pega as linhas entre o fim do treino e o fim da validação
df_validation = df.iloc[train_end_index:validation_end_index]

# Pega as linhas do fim da validação até o final do DataFrame
df_test = df.iloc[validation_end_index:]

# --- 5. Salvar os Arquivos CSV ---
try:
    print("\nSalvando os arquivos...")

    # Salva o arquivo de treino
    df_train.to_csv(train_filename, index=False)
    print(f" -> '{train_filename}' salvo com {len(df_train)} linhas.")

    # Salva o arquivo de validação
    df_validation.to_csv(validation_filename, index=False)
    print(f" -> '{validation_filename}' salvo com {len(df_validation)} linhas.")

    # Salva o arquivo de teste
    df_test.to_csv(test_filename, index=False)
    print(f" -> '{test_filename}' salvo com {len(df_test)} linhas.")

    print("\nProcesso de divisão de dados concluído com sucesso!")

except Exception as e:
    print(f"\nOcorreu um erro ao salvar os arquivos: {e}")
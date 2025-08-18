# [Oraculo Cripto] - Previsão de Preço para Ethereum

![Status do Projeto](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![Licença](https://img.shields.io/badge/licen%C3%A7a-MIT-blue)

Um projeto experimental para prever o preço do par ETH/USDT em 5 minutos utilizando Machine Learning, dados em tempo real e uma arquitetura de MLOps de ponta a ponta.

---

### ⚠️ Aviso Importante

Este projeto é desenvolvido para **fins puramente educacionais e de estudo**. As previsões geradas são resultado de um modelo estatístico e não constituem, de forma alguma, conselho financeiro. O mercado de criptoativos é extremamente volátil e arriscado. **Não utilize esta aplicação para tomar decisões reais de investimento.**

---

### ✨ Features Principais

* **Coleta de Dados em Tempo Real:** Ingestão de dados de mercado (OHLCV) da Binance a cada minuto.
* **Modelo Preditivo:** Utilização de uma Rede Neural Recorrente (LSTM) para prever o preço em um horizonte de 5 minutos.
* **API de Inferência:** Uma API RESTful construída com FastAPI para servir as previsões do modelo.
* **Dashboard Visual:** Uma interface web simples (React/Vue) para visualizar o preço atual e as previsões.
* **Pronto para Deploy:** Estrutura conteinerizada com Docker para fácil deployment na nuvem.

### 🛠️ Stack Tecnológico

* **Backend:** Python, FastAPI, Pandas, TensorFlow/Keras
* **Dados:** CCXT, InfluxDB (planejado)
* **Frontend:** React (planejado), Chart.js
* **DevOps:** Docker, Google Cloud Run (planejado), GitHub Actions

### 🚀 Começando

Estas são as instruções para configurar o projeto localmente.

#### Pré-requisitos

* Python 3.9+
* Git
* Docker (opcional, para rodar em contêiner)

#### Instalação

1.  Clone o repositório:
    ```sh
    git clone [https://github.com/](https://github.com/)[SEU-USUARIO-GIT]/[NOME-DO-REPOSITORIO].git
    ```
2.  Navegue até a pasta do projeto:
    ```sh
    cd [NOME-DO-REPOSITORIO]
    ```
3.  Crie um ambiente virtual e instale as dependências:
    ```sh
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

### ▶️ Uso

*(Esta seção será preenchida conforme o projeto avança)*

**Para rodar a API de previsão:**
```sh
# (Instruções futuras aqui)
````

#### 🗺️ Roadmap
[ ] Fase 0: Prova de Conceito (PoC) - Validar a coleta de dados e o treinamento de um modelo base em um Jupyter Notebook.

[ ] Fase 1: MVP do Backend - Construir a API de dados e o endpoint de previsão.

[ ] Fase 2: MVP do Frontend e Deploy - Desenvolver o dashboard e implantar a primeira versão.

[ ] Fase 3: Melhorias - Implementar re-treinamento automático, otimizar o modelo.

#### ⚖️ Licença
Distribuído sob a licença MIT. Veja LICENSE.txt para mais informações.

#### 📧 Contato
Carlos K P Gomes - ckauegomes@gmail.com

Link do Projeto: https://github.com/carloskaue/eth-oracle
# (Instruções futuras aqui)

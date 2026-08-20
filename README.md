# Bitcoin (BTC) Quantitative Trading Bot

Este repositório contém um sistema de trading quantitativo automatizado projetado exclusivamente para operações no par BTC/USDT. O sistema baseia-se em uma arquitetura de Deep Learning (Transformer) que analisa dados de mercado na granularidade de 15 minutos para inferir direcionalidade e atuar em tempo real.

A arquitetura original (híbrida) com redes focadas em microestrutura foi descontinuada em favor de um modelo único e mais robusto aplicado diretamente sobre a movimentação principal do Bitcoin.

## Arquitetura do Sistema

O projeto é estruturado em módulos independentes que atuam de forma sequencial na tomada de decisão:

- **Modelo Preditivo (TimeSeriesTransformer)**: Implementado em PyTorch (`model.py`), o modelo recebe sequências de indicadores técnicos (RSI, MACD, Bollinger Bands, ATR, distâncias de Médias Móveis) e dados transacionais para gerar uma probabilidade de compra ou venda.
- **Feature Engineering e Pipeline de Dados**: Coleta os dados de OHLCV em janelas de 15 minutos e padroniza as features utilizando um StandardScaler pré-treinado (`data_collection.py`). 
- **Módulo de Execução e Backtest**: Lógica de entrada e saída, simulando a performance com taxas realistas da exchange e definindo um capital fixo alocado por trade. (Exemplificado pelo `run_last_month_backtest.py` e iterado via `bot.py`).

## Como Funciona

1. **Captura de Dados**: O sistema consulta continuamente os candles de 15 minutos mais recentes via integração com a exchange (por exemplo, Bybit).
2. **Processamento**: É criado um dataframe histórico com os últimos 30 dias de dados. Em seguida, os indicadores técnicos de momento, volatilidade e tendência são calculados e normalizados.
3. **Inferência Neural**: Uma janela de observação temporal (ex: 60 períodos) é passada pelo Transformer. A saída final é submetida a uma função Softmax, que isola as probabilidades da próxima direção (Buy/Sell).
4. **Alocação de Risco**: Quando a probabilidade atinge o limiar estipulado pelo operador, a ordem é gerada com tamanho de posição controlado, permitindo um backtest realista. O registro das transações é salvo localmente.

## Estrutura do Projeto

O código principal para este workflow encontra-se atualmente centralizado e em iteração contínua (ex: pasta `Transformer_15_min/`), contemplando scripts modulares de treinamento, visualização por desktop app, além de rotinas locais de validação mensal (last month backtest) do modelo Transformer.

## Tecnologias e Bibliotecas

- Python 3.x
- Machine Learning: PyTorch, Scikit-learn (Joblib)
- Processamento de Dados: Pandas, Numpy
- Integração e Execução: CCXT
- Indicadores Técnicos: Biblioteca TA (Technical Analysis)

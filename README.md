# Hybrid Crypto Trading Bot (Macro-Micro Architecture)

Este repositório contém um sistema de trading quantitativo avançado utilizando uma abordagem híbrida (Macro + Micro), projetado para o mercado de criptomoedas, focando principalmente nos ativos `ETHUSDT` e `SOLUSDT`.

## Arquitetura do Sistema

O bot utiliza uma arquitetura em duas camadas para maximizar a precisão da previsão e a robustez da estratégia:
- **Modelo Macro (Autoformer)**: Analisa o panorama geral do mercado para determinar a "certeza direcional" (tendência principal e probabilidade de movimento).
- **Modelo Micro**: Analisa a microestrutura e os dados em baixa granularidade (volume bars, tick-a-tick) para encontrar os pontos exatos de entrada e saída.

Os módulos centrais incluem gerenciamento de risco rigoroso (`core/risk_manager.py`), pipelines de dados detalhados (`pipelines/`) e treinamento para várias granularidades (`train_micro.py`, `train_macro.py`).

## Resultados de Destaque (Backtest OOS - 60 Dias)

Durante o período de testes Out-of-Sample de 60 dias, rodando o *Modelo Macro* acoplado com execução em barras de volume, obtivemos resultados excepcionais no ativo **SOLUSDT**:

### Configuração: Limiar de Certeza Macro (Compra 75% / Venda 80%)
Ao configurar o bot para agir apenas quando a certeza do modelo Macro for alta (>=75% para compras e >=80% para vendas), o sistema filtrou os ruídos e obteve a seguinte performance:

- **Lucro Líquido (ROI)**: **+11.44%**
- **Win Rate (Ciclos de Lucro)**: **64.49%**
- **Max Drawdown**: -16.33%
- **Total de Ordens**: 1156 (409 Compras / 747 Vendas)
- **Saldo Final**: $1114.44 (Capital Inicial: $1000)

*Este cenário demonstra que ao exigir uma confiança maior do Autoformer, o bot reduz o número de trades, mas alcança uma expressiva taxa de assertividade, protegendo o capital e gerando lucros consistentes.*

## Como Funciona
1. O modelo Macro coleta dados (ex: janelas de 15m/1h) para projetar os próximos movimentos.
2. Quando a confiança atinge o limite pré-configurado, o sistema prepara ordens de Compra ou Venda.
3. As operações ocorrem em granularidade fina, gerando resiliência e adaptação em tempo real aos fluxos do order book.
4. O `risk_manager` intervém dinamicamente em caso de drawdown, garantindo o limite de perdas e acionando lucros em pontos otimizados.

## Tecnologias e Bibliotecas
- Python 3.x
- Machine Learning / Deep Learning Frameworks (PyTorch/Tensorflow) - Para arquitetura Autoformer.
- Bibliotecas para processamento e normalização (Pandas, Numpy).
- Backtesting e logging iterativo detalhado para visualização dos setups.

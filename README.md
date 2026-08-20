# Controlador Bybit V3.0 (Hybrid Execution App)

Este repositório contém o **Controlador Bybit V3.0**, um aplicativo Desktop desenvolvido com `customtkinter` para operar o mercado de Bitcoin (BTC/USDT e BTC/BRL) de forma totalmente algorítmica.

O projeto utiliza um modelo Macro baseado em **Autoformer** (`models/autoformer.py`) que processa indicadores técnicos do preço do Bitcoin (como ATR, MACD, Log Returns e distâncias de médias móveis) para definir a "certeza direcional" do ativo. Apenas as ordens que superam os limiares de segurança (ex: 60% de probabilidade) são acionadas na Bybit.

O sistema possui uma arquitetura centralizada na execução de ordens via Interface Gráfica, sem os módulos passados focados em micro-movimentos que foram descontinuados em favor de um framework unificado.

## Arquitetura do Sistema

O projeto é focado em três pilares principais que rodam ativamente no aplicativo:

1. **Dashboard & Interface (`hybrid_desktop_app.py`)**: Interface rica contendo painéis em tempo real mostrando o patrimônio, alocação dinâmica do portfólio (caixa vs. cripto), gráficos dinâmicos do spread/breakeven, e acompanhamento visual das perdas/ganhos das estratégias.
2. **Modelo Preditivo Macro (Autoformer)**: Modelo Deep Learning de séries temporais. É responsável por prever as flutuações primárias do mercado a partir da leitura de candles de 1 hora. Utiliza a biblioteca PyTorch.
3. **Risk Manager (`core/risk_manager.py`)**: Gestão avançada do capital através de controle de limite de *Drawdown*, critérios Kelly e *Stop-loss* móvel. Define exatamente com quanto de alocação de banca a ordem será enviada à corretora baseando-se no limite pré-estabelecido do usuário.

## Como Executar

Para iniciar a Interface, certifique-se de que todas as dependências do `requirements.txt` estão instaladas e as chaves de API estão definidas.

```bash
python hybrid_desktop_app.py
```

Na Interface você pode:
- Alternar entre modos de **Paper Trading Seguros** ou execução de ordens reais na conta.
- Configurar dinamicamente o *limiar de confiança* que ativa as entradas de Compra/Venda do modelo.
- Visualizar os retornos no Gráfico Interativo.
- Acessar os logs de trade salvos pelo bot (`history_log.json`).

## Tecnologias e Bibliotecas

- **Interface UI**: CustomTkinter, Matplotlib (com backend TkAgg)
- **Deep Learning**: PyTorch
- **Conexão de Exchange**: CCXT (Bybit)
- **Data & Features**: Pandas, Numpy, Joblib, biblioteca TA
- **Módulos Core Locais**: Database SQLite, Watchdogs e Executores modulares.

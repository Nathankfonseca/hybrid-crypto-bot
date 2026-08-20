import asyncio
import random
from datetime import datetime

from pipelines.micro_data import VolumeBarAggregator
from core.watchdog import Watchdog
from core.risk_manager import RiskManager
from core.execution import ExecutionEngine

async def run_orchestrator():
    print(f"[{datetime.now()}] 🚀 Iniciando Orquestrador V2 (Autoformer + CNN-Transformer)")
    
    # Inicializando Módulos Core
    risk_manager = RiskManager(max_risk_per_trade_pct=0.01)
    watchdog = Watchdog(drop_threshold_pct=0.02, time_window_seconds=60)
    execution = ExecutionEngine(slippage_budget_pct=0.0005)
    
    # Inicializando Agregador de Micro Dados (WebSocket)
    micro_aggregator = VolumeBarAggregator(target_volume=50.0) # Ajuste de volume alvo (50 BTC)
    
    # Task 1: O Watchdog fica monitorando crashes em background
    watchdog_task = asyncio.create_task(watchdog.monitor(risk_manager))
    
    # Task 2: O Agregador de Micro Dados (Volume Bars)
    micro_task = asyncio.create_task(micro_aggregator.watch_trades())
    
    # Task 3: O Loop de Inferência Híbrida
    inference_task = asyncio.create_task(inference_loop(risk_manager, execution))
    
    try:
        await asyncio.gather(watchdog_task, micro_task, inference_task)
    except KeyboardInterrupt:
        print("\nDesligando Orquestrador...")
    finally:
        await execution.close_connections()

async def inference_loop(risk_manager, execution):
    """
    Loop assíncrono que simula as passagens pelas IAs assim que os dados estão prontos.
    Em produção, ele leria a última Volume Bar do DuckDB quando fechada, passaria na CNN,
    verificaria o viés do Autoformer da hora atual e bateria o martelo.
    """
    print(f"[{datetime.now()}] 🧠 Motor de Inferência (AI Loop) Ativo.")
    
    balance_usdt = 10000.0 # Saldo Simulado de Teste
    
    while True:
        # Simula a espera por uma nova Volume Bar (substituir por polling no DuckDB ou Event)
        await asyncio.sleep(10) 
        
        # Como os modelos V2 (Autoformer e CNN-Transformer) acabaram de ser estruturados
        # e ainda precisam ser treinados através dos scripts train_macro/train_micro,
        # vamos usar valores de inferência simulados para demonstrar a passagem pela Lógica de Risco:
        
        # Simula a saída do Autoformer (1H)
        macro_bias = random.choice([0, 1, 2]) # 0: Bearish, 1: Neutral, 2: Bullish
        macro_confidence = random.uniform(0.4, 0.9)
        
        # Simula a saída da CNN-Transformer (Volume Bar recém fechada)
        micro_trigger = random.choice([-1, 0, 1]) # -1: Hold, 0: Sell, 1: Buy
        
        # Simula um pico de volatilidade do ATR da vela
        atr_volatility = random.uniform(0.5, 3.0) 
        
        # Preço simulado atual
        current_price = 60000.0 + random.uniform(-100, 100)
        
        # 1. Pede permissão para o Risk Manager (Controlador Lógico)
        action, alloc_pct, reason = risk_manager.evaluate_signal(
            macro_bias=macro_bias,
            micro_trigger=micro_trigger,
            macro_confidence=macro_confidence,
            atr_volatility=atr_volatility
        )
        
        if action != "Hold":
            print(f"\n[{datetime.now()}] 🟢 SINAL CONFIRMADO: {action} (Confiança Macro: {macro_confidence:.2f}, Z-Vol: {atr_volatility:.2f})")
            print(f"Razão: {reason}")
            print(f"Alocação Permitida (Kelly): {alloc_pct*100:.2f}% do capital")
            
            # 2. Roteia a ordem para a API
            await execution.execute_order(action, current_price, alloc_pct, balance_usdt)
        else:
            # Maioria das vezes será hold pela restrição rígida
            pass
            
if __name__ == "__main__":
    asyncio.run(run_orchestrator())

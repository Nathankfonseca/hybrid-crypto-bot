import asyncio
import ccxt.pro as ccxt
import time
from datetime import datetime

class Watchdog:
    """
    O "Cão de Guarda" Assíncrono (Circuit Breaker).
    Fica observando anomalias no mercado independentemente dos modelos de IA.
    Se detectar uma queda/subida bizarra (Flash Crash) em curtíssimo prazo, ele
    desarma o Risk Manager.
    """
    def __init__(self, symbol='BTC/USDT', drop_threshold_pct=0.02, time_window_seconds=60):
        self.symbol = symbol
        self.drop_threshold = drop_threshold_pct # Ex: 2% de queda
        self.time_window = time_window_seconds # Ex: 60 segundos
        self.exchange = ccxt.bybit({'enableRateLimit': True})
        
        self.price_history = [] # Lista de (timestamp, price)
        self.is_tripped = False
        
    async def monitor(self, risk_manager_ref):
        print(f"[{datetime.now()}] Watchdog iniciado. Monitorando flash crashes (> {self.drop_threshold*100}% em {self.time_window}s)")
        
        while True:
            try:
                # O ideal num HFT é ligar direto no WebSocket de Ticker
                ticker = await self.exchange.watch_ticker(self.symbol)
                current_price = ticker['last']
                current_time = time.time()
                
                # Guarda no histórico
                self.price_history.append((current_time, current_price))
                
                # Limpa preços mais antigos que a janela de tempo
                self.price_history = [p for p in self.price_history if current_time - p[0] <= self.time_window]
                
                if len(self.price_history) > 0:
                    max_price = max(p[1] for p in self.price_history)
                    min_price = min(p[1] for p in self.price_history)
                    
                    # Calcula o range em % 
                    # Se caiu subitamente do max para o atual
                    drop_pct = (max_price - current_price) / max_price
                    surge_pct = (current_price - min_price) / min_price
                    
                    if drop_pct >= self.drop_threshold or surge_pct >= self.drop_threshold:
                        if not self.is_tripped:
                            print(f"\n🚨🚨 [CIRCUIT BREAKER ACIONADO] Variação Anômala de {max(drop_pct, surge_pct)*100:.2f}% detectada! 🚨🚨")
                            self.is_tripped = True
                            risk_manager_ref.circuit_breaker_active = True
                            
                            # Mantém desarmado por 5 minutos após o caos
                            await asyncio.sleep(300) 
                            
                            print(f"\n✅ [CIRCUIT BREAKER DESLIGADO] Retornando ao normal.")
                            self.is_tripped = False
                            risk_manager_ref.circuit_breaker_active = False
                            self.price_history = []
                            
            except Exception as e:
                print(f"Watchdog erro: {e}. Reconectando...")
                await asyncio.sleep(5)

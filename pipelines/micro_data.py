import asyncio
import ccxt.pro as ccxt
import time
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database import DatabaseManager

class VolumeBarAggregator:
    def __init__(self, symbol='BTC/USDT', target_volume=100.0):
        self.symbol = symbol
        self.target_volume = target_volume
        self.exchange = ccxt.bybit({'enableRateLimit': True})
        self.db = DatabaseManager()
        
        self.reset_bar()
        
    def reset_bar(self):
        self.current_volume = 0.0
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.start_time = None
        self.tick_count = 0
        self.buy_volume = 0.0
        self.sell_volume = 0.0

    async def watch_trades(self):
        print(f"[{datetime.now()}] Iniciando a coleta de Ticks (Trades) via WebSocket para {self.symbol}...")
        print(f"Meta por Volume Bar: {self.target_volume} BTC")
        
        while True:
            try:
                trades = await self.exchange.watch_trades(self.symbol)
                for trade in trades:
                    self.process_trade(trade)
            except Exception as e:
                print(f"Erro no WebSocket: {e}. Reconectando em 5s...")
                await asyncio.sleep(5)

    def process_trade(self, trade):
        price = trade['price']
        amount = trade['amount']
        side = trade['side'] # 'buy' or 'sell'
        timestamp = pd.to_datetime(trade['timestamp'], unit='ms') if 'timestamp' in trade else datetime.now()
        
        # Initialize bar
        if self.open is None:
            self.open = price
            self.high = price
            self.low = price
            self.start_time = timestamp
            
        # Update bar
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.current_volume += amount
        self.tick_count += 1
        
        if side == 'buy':
            self.buy_volume += amount
        elif side == 'sell':
            self.sell_volume += amount
            
        # Check if volume target reached
        if self.current_volume >= self.target_volume:
            self.close_bar(timestamp)

    def close_bar(self, end_time):
        ofi = self.buy_volume - self.sell_volume
        
        # Save to DuckDB
        self.db.insert_volume_bar(
            start_time=self.start_time,
            end_time=end_time,
            open_p=self.open,
            high_p=self.high,
            low_p=self.low,
            close_p=self.close,
            volume=self.current_volume,
            tick_count=self.tick_count,
            ofi=ofi
        )
        
        print(f"[{end_time}] Volume Bar Fechada | C: {self.close} | Vol: {self.current_volume:.2f} | Ticks: {self.tick_count} | OFI: {ofi:.2f}")
        
        # Reset for next bar
        self.reset_bar()

if __name__ == "__main__":
    import pandas as pd # Needed for datetime parsing in process_trade
    aggregator = VolumeBarAggregator(target_volume=50.0) # 50 BTC per bar to test faster
    asyncio.run(aggregator.watch_trades())

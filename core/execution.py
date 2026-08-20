import ccxt.async_support as ccxt_async
import os
import asyncio
from datetime import datetime

class ExecutionEngine:
    def __init__(self, symbol='BTC/USDT', slippage_budget_pct=0.0005): # 0.05% slippage
        self.symbol = symbol
        self.slippage_budget = slippage_budget_pct
        
        # In a real environment, load keys from .env
        api_key = os.getenv('BYBIT_API_KEY', '')
        api_secret = os.getenv('BYBIT_API_SECRET', '')
        
        self.exchange = ccxt_async.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot' # can be 'future' if trading derivatives
            }
        })
        
        # Testnet (sandbox) mode by default for safety
        self.exchange.set_sandbox_mode(True)
        
    async def execute_order(self, action, current_price, alloc_pct, total_balance_usdt):
        if action not in ["Buy", "Sell"]:
            return None
            
        try:
            # 1. Size calculation
            usdt_to_spend = total_balance_usdt * alloc_pct
            amount_coin = usdt_to_spend / current_price
            
            # 2. Limit-IOC Logic with Slippage Budget
            # If buying, we are willing to pay up to current_price + slippage
            # If selling, we are willing to accept down to current_price - slippage
            if action == "Buy":
                limit_price = current_price * (1.0 + self.slippage_budget)
                side = 'buy'
            else:
                limit_price = current_price * (1.0 - self.slippage_budget)
                side = 'sell'
                
            # Formatting precision based on exchange rules (simplified here)
            limit_price = round(limit_price, 2)
            amount_coin = round(amount_coin, 5)
            
            print(f"[{datetime.now()}] ⚡ EXECUÇÃO: Roteando ordem {side.upper()} de {amount_coin} {self.symbol.split('/')[0]} @ {limit_price} (IOC)")
            
            # 3. Create Order
            if self.exchange.apiKey:
                order = await self.exchange.create_order(
                    symbol=self.symbol,
                    type='limit',
                    side=side,
                    amount=amount_coin,
                    price=limit_price,
                    params={
                        'timeInForce': 'IOC' # Immediate Or Cancel
                    }
                )
                print(f"✅ Ordem executada com sucesso! ID: {order['id']}")
                return order
            else:
                print("⚠️ API Keys ausentes. Ordem simulada localmente com sucesso (Dry-run).")
                return {"status": "simulated", "side": side, "amount": amount_coin, "price": limit_price}
                
        except Exception as e:
            print(f"❌ Erro na execução da ordem: {e}")
            return None

    async def close_connections(self):
        await self.exchange.close()

import sys
import traceback
from hybrid_desktop_app import HybridBotApp

def run_headless():
    app = HybridBotApp()
    # Mock some UI elements so it doesn't wait for the window
    app.var_live_trade.set(False)
    app.entry_sim_bal.delete(0, 'end')
    app.entry_sim_bal.insert(0, "1000.0")
    
    app.is_running = True
    app.balance = 0.0
    app.initial_balance = 0.0
    app.btc_held = 0.0
    app.current_price = 0.0
    app.initial_btc_price = 0.0
    
    app.threshold = float(app.entry_thresh.get())
    app.risk_manager.max_risk_pct = float(app.entry_risk.get()) / 100.0
    app.risk_manager.micro_threshold = app.threshold
    
    app.history_time = []
    app.history_portfolio = []
    app.history_btc = []
    app.history_cash = []
    app.history_alloc_pct = []
    
    app.peak_equity = 0.0
    app.max_dd = 0.0
    app.trades_buy = 0
    app.trades_sell = 0
    
    # We override the after method to execute synchronously for debugging
    def mock_after(delay, func, *args):
        try:
            func(*args)
        except Exception as e:
            print("ERROR IN GUI METHOD:")
            traceback.print_exc()
            sys.exit(1)
            
    app.after = mock_after
    
    # Run the loop exactly once
    def mock_sleep(seconds):
        print(f"Loop finished successfully. Sleeping {seconds}s.")
        sys.exit(0)
        
    import time
    time.sleep = mock_sleep
    
    print("Starting loop...")
    try:
        app.live_trading_loop()
    except Exception as e:
        print("ERROR IN LOOP:")
        traceback.print_exc()

if __name__ == "__main__":
    run_headless()

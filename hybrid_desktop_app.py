import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import ccxt
import joblib
import ta
import os
import traceback
import json
import matplotlib.dates as mdates
from dotenv import load_dotenv

# Nossos modelos
from models.autoformer import Autoformer
from core.risk_manager import RiskManager

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class HybridBotApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🤖 Controlador Bybit V3.0 - Hybrid Execution")
        self.geometry("1400x850")
        
        self.is_fullscreen = False
        self.bind("<Escape>", self.exit_fullscreen)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- Sidebar --
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=320, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Controlador V3.0 (Bybit)", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(15, 10))

        self.lbl_pair = ctk.CTkLabel(self.sidebar_frame, text="Par de Negociação:")
        self.lbl_pair.grid(row=1, column=0, padx=20, pady=(2, 0), sticky="w")
        self.combo_pair = ctk.CTkComboBox(self.sidebar_frame, values=["BTC/USDT", "BTC/BRL"], command=self.on_pair_change)
        self.combo_pair.set("BTC/USDT")
        self.combo_pair.grid(row=2, column=0, padx=20, pady=(2, 5), sticky="ew")

        # Configs de Risco
        self.lbl_thresh = ctk.CTkLabel(self.sidebar_frame, text="Limiar de Entrada OOS (Ex: 0.60):")
        self.lbl_thresh.grid(row=3, column=0, padx=20, pady=(2, 0), sticky="w")
        self.entry_thresh = ctk.CTkEntry(self.sidebar_frame)
        self.entry_thresh.insert(0, "0.60")
        self.entry_thresh.grid(row=4, column=0, padx=20, pady=(2, 5), sticky="ew")
        
        self.lbl_risk = ctk.CTkLabel(self.sidebar_frame, text="Alocação por Ordem (%):")
        self.lbl_risk.grid(row=5, column=0, padx=20, pady=(2, 0), sticky="w")
        self.entry_risk = ctk.CTkEntry(self.sidebar_frame)
        self.entry_risk.insert(0, "3.0") 
        self.entry_risk.grid(row=6, column=0, padx=20, pady=(2, 5), sticky="ew")
        
        self.lbl_sim_bal = ctk.CTkLabel(self.sidebar_frame, text="Saldo Simulado ($/R$ Paper):")
        self.lbl_sim_bal.grid(row=7, column=0, padx=20, pady=(2, 0), sticky="w")
        self.entry_sim_bal = ctk.CTkEntry(self.sidebar_frame)
        self.entry_sim_bal.insert(0, "1000.0") 
        self.entry_sim_bal.grid(row=8, column=0, padx=20, pady=(2, 5), sticky="ew")

        self.lbl_kelly = ctk.CTkLabel(self.sidebar_frame, text="Multiplicador Kelly (Ex: 0.5):")
        self.lbl_kelly.grid(row=9, column=0, padx=20, pady=(2, 0), sticky="w")
        self.entry_kelly = ctk.CTkEntry(self.sidebar_frame)
        self.entry_kelly.insert(0, "0.5")
        self.entry_kelly.grid(row=10, column=0, padx=20, pady=(2, 5), sticky="ew")

        self.chart_opts_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="#1a1a1a")
        self.chart_opts_frame.grid(row=11, column=0, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(self.chart_opts_frame, text="Variáveis do Gráfico", font=ctk.CTkFont(weight="bold")).pack(pady=2)
        
        self.timeframe_var = ctk.StringVar(value="1 Hora")
        self.seg_timeframe = ctk.CTkSegmentedButton(self.chart_opts_frame, 
                                                    variable=self.timeframe_var, 
                                                    values=["1 Hora", "1 Dia", "1 Semana", "1 Mês", "Tudo"])
        self.seg_timeframe.pack(fill="x", padx=10, pady=5)
        
        self.var_plot_port = ctk.BooleanVar(value=True)
        self.chk_port = ctk.CTkCheckBox(self.chart_opts_frame, text="Patrimônio Total", variable=self.var_plot_port, fg_color="#00ff88", text_color="#00ff88")
        self.chk_port.pack(anchor="w", padx=10, pady=2)
        
        self.var_plot_btc = ctk.BooleanVar(value=True)
        self.chk_btc = ctk.CTkCheckBox(self.chart_opts_frame, text="Bitcoin HODL", variable=self.var_plot_btc, fg_color="gray", text_color="gray")
        self.chk_btc.pack(anchor="w", padx=10, pady=2)
        
        self.var_plot_cash = ctk.BooleanVar(value=False)
        self.chk_cash = ctk.CTkCheckBox(self.chart_opts_frame, text="Caixa Livre", variable=self.var_plot_cash, fg_color="#ffcc99", text_color="#ffcc99")
        self.chk_cash.pack(anchor="w", padx=10, pady=2)
        
        self.var_plot_alloc = ctk.BooleanVar(value=False)
        self.chk_alloc = ctk.CTkCheckBox(self.chart_opts_frame, text="Balanço Alocado (%)", variable=self.var_plot_alloc, fg_color="#e066ff", text_color="#e066ff")
        self.chk_alloc.pack(anchor="w", padx=10, pady=2)
        
        self.var_plot_in_btc = ctk.BooleanVar(value=False)
        self.chk_in_btc = ctk.CTkCheckBox(self.chart_opts_frame, text="Valores em BTC", variable=self.var_plot_in_btc, fg_color="#4da6ff", text_color="#4da6ff")
        self.chk_in_btc.pack(anchor="w", padx=10, pady=5)

        # MODO DE EXECUCAO
        self.var_live_trade = ctk.BooleanVar(value=False)
        self.chk_live_trade = ctk.CTkCheckBox(self.sidebar_frame, text="⚠️ Execução Real (Usar Saldo Real)\n(Desmarque para Paper Trading)", 
                                              variable=self.var_live_trade, fg_color="red", hover_color="darkred")
        self.chk_live_trade.grid(row=12, column=0, padx=20, pady=(10, 5), sticky="w")

        self.btn_start = ctk.CTkButton(self.sidebar_frame, text="▶ INICIAR CONEXÃO", command=self.start_simulation, fg_color="darkred", hover_color="red")
        self.btn_start.grid(row=13, column=0, padx=20, pady=5, sticky="ew")

        self.btn_stop = ctk.CTkButton(self.sidebar_frame, text="⏹ Parar e Desconectar", command=self.stop_simulation, state="disabled", fg_color="gray")
        self.btn_stop.grid(row=14, column=0, padx=20, pady=5, sticky="ew")
        
        self.btn_history = ctk.CTkButton(self.sidebar_frame, text="📜 Ver Histórico", command=self.open_history, fg_color="#336699", hover_color="#224466")
        self.btn_history.grid(row=15, column=0, padx=20, pady=(15, 5), sticky="ew")
        
        self.btn_fs = ctk.CTkButton(self.sidebar_frame, text="⛶ Tela Cheia", command=self.toggle_fullscreen, fg_color="#444")
        self.btn_fs.grid(row=16, column=0, padx=20, pady=5, sticky="ew")
        
        # Quant Metrics Box
        self.metrics_box = ctk.CTkFrame(self.sidebar_frame, fg_color="#1a1a1a")
        self.metrics_box.grid(row=17, column=0, padx=10, pady=5, sticky="nsew")
        
        ctk.CTkLabel(self.metrics_box, text="Status da Execução", font=ctk.CTkFont(weight="bold")).pack(pady=2)
        
        self.lbl_trades = ctk.CTkLabel(self.metrics_box, text="Ordens: 0 (0 C / 0 V)")
        self.lbl_trades.pack(anchor="w", padx=10, pady=1)
        
        self.lbl_mdd = ctk.CTkLabel(self.metrics_box, text="Max Drawdown: 0.00%", text_color="orange")
        self.lbl_mdd.pack(anchor="w", padx=10, pady=1)
        
        self.lbl_api = ctk.CTkLabel(self.metrics_box, text="API: Desconectada", text_color="gray")
        self.lbl_api.pack(anchor="w", padx=10, pady=(1, 5))
        
        self.lbl_sync = ctk.CTkLabel(self.metrics_box, text="Sync de Saldo: Off", text_color="gray")
        self.lbl_sync.pack(anchor="w", padx=10, pady=(1, 5))

        # -- Main Area --
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.main_frame.grid_rowconfigure(3, weight=10)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Top Dashboards
        self.dash_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.dash_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        self.dash_frame.grid_columnconfigure((0,1,2), weight=1)
        
        self.macro_box = ctk.CTkFrame(self.dash_frame, fg_color="#222")
        self.macro_box.grid(row=0, column=0, padx=5, sticky="ew")
        ctk.CTkLabel(self.macro_box, text="Autoformer (Macro 1H)", font=ctk.CTkFont(size=12)).pack(pady=(2,0))
        self.lbl_macro_state = ctk.CTkLabel(self.macro_box, text="Standby", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_macro_state.pack(pady=(2,5))
        
        self.breakeven_box = ctk.CTkFrame(self.dash_frame, fg_color="#222")
        self.breakeven_box.grid(row=0, column=1, padx=5, sticky="ew")
        ctk.CTkLabel(self.breakeven_box, text="Spread Breakeven (Taxas 0.1%)", font=ctk.CTkFont(size=12)).pack(pady=(2,0))
        self.lbl_breakeven_state = ctk.CTkLabel(self.breakeven_box, text="0.00", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_breakeven_state.pack(pady=(2,5))
        
        self.risk_box = ctk.CTkFrame(self.dash_frame, fg_color="#222")
        self.risk_box.grid(row=0, column=2, padx=5, sticky="ew")
        ctk.CTkLabel(self.risk_box, text="Risk Manager (Execute)", font=ctk.CTkFont(size=12)).pack(pady=(2,0))
        self.lbl_risk_state = ctk.CTkLabel(self.risk_box, text="Standby", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_risk_state.pack(pady=(2,5))

        # Balance Info 
        self.lbl_portfolio = ctk.CTkLabel(self.main_frame, text="Patrimônio (Bybit): 0.00000 BTC | 0.00 USDT | 0.00 BRL", font=ctk.CTkFont(size=13, weight="bold"), text_color="#ffcc00")
        self.lbl_dynamic_balance = ctk.CTkLabel(self.main_frame, text="Alocação (BTC/USDT): Caixa = $0.00 (100.0%) | Moedas = 0.00000 BTC (0.0%)", font=ctk.CTkFont(size=12), text_color="#aaaaaa")
        
        self.lbl_portfolio.grid(row=1, column=0, pady=(10, 2))
        self.lbl_dynamic_balance.grid(row=2, column=0, pady=(0, 10))

        # Chart
        self.fig, self.ax = plt.subplots(facecolor='#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_color('#2b2b2b')
        self.ax.spines['right'].set_color('#2b2b2b')
        self.ax2 = None 
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        
        self.annot = self.ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                                      bbox=dict(boxstyle="round4", fc="#333333", ec="white", lw=1),
                                      arrowprops=dict(arrowstyle="->", color="white"), color="white")
        self.annot.set_visible(False)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

        self.is_running = False
        
        # Caches para thread safety
        self._cached_risk = 0.03
        self._cached_thresh = 0.60
        self._cached_sim_bal = 1000.0
        self._cached_kelly = 0.5
        self._cached_live_trade = False
        self.poll_ui_variables()
        
    def on_pair_change(self, value):
        if not self.is_running:
            quote = value.split('/')[1]
            fiat = "R$" if quote == "BRL" else "$"
            self.lbl_dynamic_balance.configure(text=f"[ Caixa Livre: {fiat}0.00 {quote} | Moedas: 0.00000 BTC ] -> Cotação BTC: {fiat}0.00")
        

    def update_lbl(self, lbl, txt, col):
        lbl.configure(text=txt, text_color=col)

    def poll_ui_variables(self):
        try:
            self._cached_risk = float(self.entry_risk.get()) / 100.0
            self._cached_thresh = float(self.entry_thresh.get())
            self._cached_sim_bal = float(self.entry_sim_bal.get())
            self._cached_kelly = float(self.entry_kelly.get())
            self._cached_live_trade = self.var_live_trade.get()
        except:
            pass
        self.after(1000, self.poll_ui_variables)
        
    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.attributes("-fullscreen", self.is_fullscreen)
        if self.is_fullscreen:
            self.btn_fs.configure(text="🗗 Sair da Tela Cheia")
        else:
            self.btn_fs.configure(text="⛶ Tela Cheia")
            
    def exit_fullscreen(self, event=None):
        self.is_fullscreen = False
        self.attributes("-fullscreen", False)
        self.btn_fs.configure(text="⛶ Tela Cheia")

    def open_history(self):
        history_win = ctk.CTkToplevel(self)
        history_win.title("Histórico de Operações")
        history_win.geometry("800x400")
        
        txt_box = ctk.CTkTextbox(history_win, font=ctk.CTkFont(family="Courier", size=12))
        txt_box.pack(fill="both", expand=True, padx=10, pady=10)
        
        if hasattr(self, 'trade_ledger') and self.trade_ledger:
            txt_box.insert("0.0", "\n".join(self.trade_ledger))
        else:
            txt_box.insert("0.0", "Nenhuma operação registrada ainda.")
            
        txt_box.configure(state="disabled")

    def on_hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax or event.inaxes == self.ax2:
            found = False
            axes = [self.ax]
            if self.ax2: axes.append(self.ax2)
            
            for ax in axes:
                for line in ax.lines:
                    cont, ind = line.contains(event)
                    if cont:
                        x, y = line.get_data()
                        idx = ind["ind"][0] 
                        self.annot.xy = (event.xdata, y[idx])
                        text = f"{line.get_label()}:\n {y[idx]:.4f}"
                        self.annot.set_text(text)
                        self.annot.set_visible(True)
                        self.fig.canvas.draw_idle()
                        found = True
                        break
                if found: break
                
            if not found and vis:
                self.annot.set_visible(False)
                self.fig.canvas.draw_idle()
        else:
            if vis:
                self.annot.set_visible(False)
                self.fig.canvas.draw_idle()
        
    def start_simulation(self):
        self.is_running = True
        self.balance = 0.0
        self.btc_held = 0.0
        self.usdt_held = 0.0
        self.brl_held = 0.0
        self.initial_balance = 0.0
        self.btc_held = 0.0
        self.current_price = 0.0
        self.initial_btc_price = 0.0
        self.acc_buy_fiat = 0.0
        self.acc_sell_btc = 0.0
        
        self.exec_pair = self.combo_pair.get()
        self.quote_currency = self.exec_pair.split('/')[1]
        self.fiat_sym = "R$" if self.quote_currency == "BRL" else "$"
        
        self.threshold = float(self.entry_thresh.get())
        risk_pct = float(self.entry_risk.get()) / 100.0
        kelly_frac = float(self.entry_kelly.get())
        
        self.risk_manager = RiskManager(max_risk_per_trade_pct=risk_pct, kelly_fraction=kelly_frac)
        self.risk_manager.micro_threshold = self.threshold
        
        self.history_time = []
        self.history_portfolio = []
        self.history_btc = []
        self.history_cash = []
        self.history_alloc_pct = []
        self.history_price = []
        self.trade_ledger = []
        
        self.peak_equity = 0.0
        self.max_dd = 0.0
        self.trades_buy = 0
        self.trades_sell = 0
        self.last_trade_candle = None
        
        if self.var_live_trade.get():
            try:
                with open("history_log.json", "r") as f:
                    data = json.load(f)
                    self.history_time = data.get("time", [])
                    self.history_portfolio = data.get("portfolio", [])
                    self.history_cash = data.get("cash", [])
                    self.history_alloc_pct = data.get("alloc", [])
                    self.history_price = data.get("price", [])
                    self.history_btc = data.get("btc", [])
                    self.initial_btc_price = data.get("initial_btc_price", 0.0)
                    self.initial_balance = data.get("initial_balance", 0.0)
                    self.peak_equity = data.get("peak_equity", 0.0)
                    self.max_dd = data.get("max_dd", 0.0)
                    self.trades_buy = data.get("trades_buy", 0)
                    self.trades_sell = data.get("trades_sell", 0)
                    if "last_saved_balance" in data:
                        self.last_saved_balance = data["last_saved_balance"]
                        self.last_saved_btc = data["last_saved_btc"]
                    if "baseline_btc" in data:
                        self.baseline_btc = data["baseline_btc"]
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            
        import time
        self.run_id = time.time()
        
        self.ax.clear()
        if self.ax2 is not None:
            self.ax2.remove()
            self.ax2 = None
            
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal", fg_color="red")
        
        # Bloquear edição de configs enquanto roda para evitar pulos de saldo
        self.chk_live_trade.configure(state="disabled")
        self.entry_sim_bal.configure(state="disabled")
        self.entry_kelly.configure(state="disabled")
        
        self.thread = threading.Thread(target=self.live_trading_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop_simulation(self, is_crash=False):
        self.is_running = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled", fg_color="gray")
        
        # Desbloquear configs
        self.chk_live_trade.configure(state="normal")
        self.entry_sim_bal.configure(state="normal")
        self.entry_kelly.configure(state="normal")
        if not is_crash:
            self.lbl_api.configure(text="API: Desconectada", text_color="gray")
            self.lbl_sync.configure(text="Sync de Saldo: Off", text_color="gray")

    def build_volume_bars(self, df_1m, target_volume=5.0):
        vol_bars = []
        current_vol = 0
        if len(df_1m) == 0: return pd.DataFrame()
            
        bar_open = df_1m['open'].iloc[0]
        bar_high = df_1m['high'].iloc[0]
        bar_low = df_1m['low'].iloc[0]
        
        for idx, row in df_1m.iterrows():
            current_vol += row['volume']
            bar_high = max(bar_high, row['high'])
            bar_low = min(bar_low, row['low'])
            
            if current_vol >= target_volume:
                vol_bars.append({
                    'timestamp': row['timestamp'],
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': row['close'],
                    'volume': current_vol
                })
                current_vol = 0
                bar_open = row['close']
                bar_high = row['close']
                bar_low = row['close']
                
        if current_vol > 0:
            vol_bars.append({
                'timestamp': df_1m['timestamp'].iloc[-1],
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': df_1m['close'].iloc[-1],
                'volume': current_vol
            })
                
        df_vol = pd.DataFrame(vol_bars)
        return df_vol

    def live_trading_loop(self):
        my_run_id = getattr(self, 'run_id', None)
        # CARREGANDO DO ARQUIVO arquivo.env
        load_dotenv("arquivo.env")
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_SECRET")
        
        real_execution = self._cached_live_trade
        
        if not api_key:
            self.after(0, self.update_lbl, self.lbl_api, "Aviso: Sem Chaves no arquivo.env. Usando API Pública.", "orange")
            bybit = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}})
        else:
            self.after(0, self.update_lbl, self.lbl_api, "API: Conectando Bybit (Auth)...", "yellow")
            bybit = ccxt.bybit({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
            })
            
            try:
                bybit.load_time_difference()
            except Exception as e:
                pass
        
        try:
            if real_execution and api_key:
                self.after(0, self.update_lbl, self.lbl_sync, "Sync: Lendo saldo real (Trade API)...", "yellow")
                real_balance = bybit.fetch_balance()
                self.balance = real_balance['total'].get(self.quote_currency, 0.0)
                self.btc_held = real_balance['total'].get('BTC', 0.0)
                self.free_balance = real_balance['free'].get(self.quote_currency, 0.0)
                self.free_btc = real_balance['free'].get('BTC', 0.0)
                
                if hasattr(self, 'last_saved_balance') and hasattr(self, 'last_saved_btc'):
                    self.offline_delta_fiat = self.balance - self.last_saved_balance
                    self.offline_delta_btc = self.btc_held - self.last_saved_btc
                    
                self.usdt_held = real_balance['total'].get('USDT', 0.0)
                self.brl_held = real_balance['total'].get('BRL', 0.0)
                self.after(0, self.update_lbl, self.lbl_sync, "Sync: OK (Saldo Real Obtido)", "green")
                
                # RECONSTRUÇÃO DE HISTÓRICO
                try:
                    self.after(0, self.update_lbl, self.lbl_api, "API: Reconstruindo histórico...", "yellow")
                    
                    if len(self.history_time) > 0:
                        last_t_str = self.history_time[-1]
                        last_t_obj = datetime.strptime(last_t_str, "%Y-%m-%d %H:%M:%S")
                        since_ms = int(last_t_obj.timestamp() * 1000)
                        is_gap = True
                    else:
                        since_ms = int((time.time() - 30 * 24 * 60 * 60) * 1000)
                        is_gap = False
                        
                    ohlcv = bybit.fetch_ohlcv(self.exec_pair, '1h', since=since_ms, limit=750)
                    trades = bybit.fetch_my_trades(self.exec_pair, since=since_ms, limit=1000)
                    
                    if ohlcv:
                        df_candles = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        temp_fiat = self.balance
                        temp_btc = self.btc_held
                        
                        for t in reversed(trades):
                            side = t['side']
                            amount = t['amount']
                            cost = t['cost']
                            if side == 'buy':
                                temp_btc -= amount
                                temp_fiat += cost
                            elif side == 'sell':
                                temp_btc += amount
                                temp_fiat -= cost
                                
                        run_fiat = max(0, temp_fiat)
                        run_btc = max(0, temp_btc)
                        
                        hist_time, hist_port, hist_cash, hist_alloc, hist_price = [], [], [], [], []
                        trade_idx = 0
                        n_trades = len(trades)
                        
                        for _, row in df_candles.iterrows():
                            ts = row['timestamp']
                            close_p = row['close']
                            
                            while trade_idx < n_trades and trades[trade_idx]['timestamp'] <= ts:
                                t = trades[trade_idx]
                                if t['side'] == 'buy':
                                    run_btc += t['amount']
                                    run_fiat -= t['cost']
                                elif t['side'] == 'sell':
                                    run_btc -= t['amount']
                                    run_fiat += t['cost']
                                trade_idx += 1
                                
                            port_val = run_fiat + (run_btc * close_p)
                            btc_fiat_val = run_btc * close_p
                            alloc_pct = (btc_fiat_val / port_val) * 100 if port_val > 0 else 0
                            
                            dt_str = datetime.fromtimestamp(ts/1000).strftime("%Y-%m-%d %H:%M:%S")
                            if is_gap and len(self.history_time) > 0 and dt_str == self.history_time[-1]:
                                continue
                                
                            hist_time.append(dt_str)
                            hist_port.append(port_val)
                            hist_cash.append(run_fiat)
                            hist_alloc.append(alloc_pct)
                            hist_price.append(close_p)
                            
                        if len(hist_time) > 0:
                            if is_gap:
                                self.history_time.extend(hist_time)
                                self.history_portfolio.extend(hist_port)
                                self.history_cash.extend(hist_cash)
                                self.history_alloc_pct.extend(hist_alloc)
                                self.history_price.extend(hist_price)
                                
                                if getattr(self, 'baseline_btc', 0) == 0 and len(self.history_portfolio) > 0 and self.history_price[0] > 0:
                                    self.baseline_btc = self.history_portfolio[0] / self.history_price[0]
                                new_hist_btc = [getattr(self, 'baseline_btc', 0) * p for p in hist_price]
                                self.history_btc.extend(new_hist_btc)
                                
                                self.peak_equity = max(self.peak_equity, max(hist_port))
                                self.trades_buy += sum(1 for t in trades if t['side'] == 'buy')
                                self.trades_sell += sum(1 for t in trades if t['side'] == 'sell')
                            else:
                                self.history_time = hist_time
                                self.history_portfolio = hist_port
                                self.history_cash = hist_cash
                                self.history_alloc_pct = hist_alloc
                                self.history_price = hist_price
                                initial_price = hist_price[0]
                                self.initial_btc_price = initial_price
                                self.initial_balance = hist_port[0]
                                self.peak_equity = max(hist_port)
                                self.baseline_btc = self.initial_balance / initial_price if initial_price > 0 else 0
                                self.history_btc = [self.baseline_btc * p for p in hist_price]
                                self.trades_buy = sum(1 for t in trades if t['side'] == 'buy')
                                self.trades_sell = sum(1 for t in trades if t['side'] == 'sell')
                            
                    self.after(0, self.update_lbl, self.lbl_api, "API: Histórico OK", "green")
                except Exception as e:
                    print(f"Erro na reconstrução de histórico: {e}")
                    
            else:
                # MODO PAPER TRADING (SIMULA O SALDO COM BASE NO INPUT CACHED)
                sim_bal = self._cached_sim_bal
                self.after(0, self.update_lbl, self.lbl_sync, f"Sync: Paper Money ({self.fiat_sym}{sim_bal} Fake)", "cyan")
                self.balance = sim_bal
                self.btc_held = 0.0
            
            # Carregar Modelos e Pesos
            self.after(0, self.update_lbl, self.lbl_api, "API: Carregando I.A...", "yellow")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            macro_scaler = joblib.load('models/macro_scaler.pkl')
            
            # Features configuration
            macro_features = ['log_return', 'atr_14', 'atr_50', 'macd', 'macd_signal', 'dist_sma_50', 'dist_sma_200']
            
            # Model Initialization
            macro_model = Autoformer(num_features=len(macro_features), num_classes=2).to(device)
            macro_model.load_state_dict(torch.load('models/macro_autoformer.pt', map_location=device))
            macro_model.eval()
            
            macro_seq_len = 96
            micro_seq_len = 10
            
            if real_execution:
                self.after(0, self.update_lbl, self.lbl_api, "API: ONLINE (TENTARÁ ORDENAR NA BYBIT)", "red")
            else:
                self.after(0, self.update_lbl, self.lbl_api, "API: ONLINE (PAPER TRADING SEGURO)", "cyan")
            
        except Exception as e:
            print("CRITICAL CRASH IN LIVE TRADING LOOP:")
            traceback.print_exc()
            self.after(0, self.update_lbl, self.lbl_api, f"API Falhou: {str(e)[:30]}", "red")
            self.is_running = False
            self.after(0, lambda: self.stop_simulation(is_crash=True))
            return
            
        sync_counter = 0
        fee_rate = 0.001
        slippage = 0.0005 
        
        while self.is_running and getattr(self, 'run_id', None) == my_run_id:
            try:
                if getattr(self, 'in_circuit_breaker', False):
                    self.in_circuit_breaker = False
                    if real_execution:
                        self.after(0, self.update_lbl, self.lbl_api, "API: ONLINE (RECONECTADO)", "green")
                    else:
                        self.after(0, self.update_lbl, self.lbl_api, "API: ONLINE (PAPER TRADING)", "cyan")
                
                # Polling Inteligente
                ohlcv_1h = bybit.fetch_ohlcv('BTC/USDT', '1h', limit=1000)
                df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                ohlcv_1m = bybit.fetch_ohlcv('BTC/USDT', '1m', limit=1000)
                df_1m = pd.DataFrame(ohlcv_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                current_candle_time = df_1m['timestamp'].iloc[-1]
                
                ticker = bybit.fetch_ticker(self.exec_pair)
                
                # Para pares de baixa liquidez (como BTC/BRL), o 'last' pode ficar congelado por minutos.
                # Usamos o 'mid-price' (média entre bid e ask) para refletir o mercado global em tempo real.
                bid = ticker.get('bid')
                ask = ticker.get('ask')
                if bid is not None and ask is not None and bid > 0 and ask > 0:
                    current_price = (bid + ask) / 2.0
                else:
                    current_price = ticker['last']
                
                if self.initial_btc_price == 0.0:
                    self.initial_btc_price = current_price
                    self.initial_balance = self.balance + (self.btc_held * current_price)
                    self.peak_equity = self.initial_balance
                self.current_price = current_price
                
                if not getattr(self, 'baseline_btc', 0.0):
                    self.baseline_btc = (self.balance + (self.btc_held * current_price)) / current_price if current_price > 0 else 0.0
                if hasattr(self, 'offline_delta_fiat'):
                    injected = self.offline_delta_fiat + (self.offline_delta_btc * current_price)
                    if abs(injected) > 1.0:
                        self.baseline_btc += (injected / current_price)
                    delattr(self, 'offline_delta_fiat')
                    delattr(self, 'offline_delta_btc')
                
                # Engenharia Macro
                df_1h['log_return'] = np.log(df_1h['close'] / df_1h['close'].shift(1))
                df_1h['atr_14'] = ta.volatility.average_true_range(df_1h['high'], df_1h['low'], df_1h['close'], window=14)
                df_1h['atr_50'] = ta.volatility.average_true_range(df_1h['high'], df_1h['low'], df_1h['close'], window=50)
                macd = ta.trend.MACD(df_1h['close'])
                df_1h['macd'] = macd.macd()
                df_1h['macd_signal'] = macd.macd_signal()
                sma_50 = ta.trend.sma_indicator(df_1h['close'], window=50)
                sma_200 = ta.trend.sma_indicator(df_1h['close'], window=200)
                df_1h['dist_sma_50'] = (df_1h['close'] - sma_50) / sma_50
                df_1h['dist_sma_200'] = (df_1h['close'] - sma_200) / sma_200
                df_1h.dropna(inplace=True)
                
                df_vol = self.build_volume_bars(df_1m, target_volume=50.0)
                df_vol['log_return'] = np.log(df_vol['close'] / df_vol['close'].shift(1))
                df_vol['momentum'] = df_vol['close'] - df_vol['close'].shift(1)
                df_vol.dropna(inplace=True)
                
                # Sync de Carteira Real (Apenas se o modo Real Execution estiver ligado)
                # Sempre verificamos o cached flag para atualizar
                real_execution = self._cached_live_trade
                
                if real_execution and api_key:
                    sync_counter += 1
                    if sync_counter >= 10:
                        try:
                            real_balance = bybit.fetch_balance()
                            new_bal = real_balance['total'].get(self.quote_currency, 0.0)
                            new_btc = real_balance['total'].get('BTC', 0.0)
                            new_usdt = real_balance['total'].get('USDT', 0.0)
                            new_brl = real_balance['total'].get('BRL', 0.0)
                            self.free_balance = real_balance['free'].get(self.quote_currency, 0.0)
                            self.free_btc = real_balance['free'].get('BTC', 0.0)
                            
                            delta_fiat = new_bal - self.balance
                            delta_btc = new_btc - self.btc_held
                            injected = delta_fiat + (delta_btc * self.current_price)
                            
                            if abs(injected) > 1.0:
                                self.baseline_btc += (injected / self.current_price)
                                
                            self.balance = new_bal
                            self.btc_held = new_btc
                            self.usdt_held = new_usdt
                            self.brl_held = new_brl
                            sync_counter = 0
                        except Exception:
                            pass
                
                print(f"[DEBUG] len(df_1h): {len(df_1h)}, len(df_vol): {len(df_vol)}")
                
                # Inferência da IA
                if len(df_1h) >= macro_seq_len and len(df_vol) >= 1:
                    print(f"[DEBUG] Entrou na inferencia!")
                    try:
                        self.risk_manager.max_risk_pct = self._cached_risk
                        self.risk_manager.kelly_fraction = self._cached_kelly
                        self.risk_manager.micro_threshold = self._cached_thresh
                        self.risk_manager.macro_intensity = 3.0 # Fixo o campeão absoluto!
                    except: pass
                    
                    macro_input = df_1h.tail(macro_seq_len).copy()
                    macro_input[macro_features] = macro_scaler.transform(macro_input[macro_features])
                    
                    x_macro = torch.tensor(macro_input[macro_features].values, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        macro_out = macro_model(x_macro)
                        macro_probs = torch.softmax(macro_out, dim=1).squeeze().cpu().numpy().tolist()
                        
                        # O Macro Vencedor opera sozinho (Pass-through do Micro)
                        micro_probs = [0.49, 0.51]
                        
                    atr_volatility = 1.0
                    mean_atr = df_1h['atr_14'].tail(14).mean()
                    std_atr = df_1h['atr_14'].tail(14).std()
                    if std_atr > 0:
                        atr_volatility = (df_1h['atr_14'].iloc[-1] - mean_atr) / std_atr
                        
                    action, alloc, reason = self.risk_manager.evaluate_signal(macro_probs, micro_probs, atr_volatility)
                    
                    # Força o respeito absoluto ao limiar de certeza:
                    if action == "Buy" and macro_probs[1] < self._cached_thresh:
                        action = "Hold"
                        reason = f"Certeza Buy ({macro_probs[1]:.2f}) < Limiar ({self._cached_thresh})"
                        alloc = 0.0
                    elif action == "Sell" and macro_probs[0] < self._cached_thresh:
                        action = "Hold"
                        reason = f"Certeza Sell ({macro_probs[0]:.2f}) < Limiar ({self._cached_thresh})"
                        alloc = 0.0

                    if action != "Hold" and current_candle_time == self.last_trade_candle:
                        action = "Hold"
                        reason = "Cooldown (Uma ordem por minuto)"
                        alloc = 0.0
                    
                    color_risk = "gray"
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # EXECUÇÃO FINANCEIRA
                    if action == "Buy":
                        if not hasattr(self, 'free_balance'): self.free_balance = self.balance
                        
                        if self.free_balance < 1.0:
                            if self.brl_held > 10.0 and self.quote_currency != "BRL":
                                action = "Hold (Mude o par para BTC/BRL!)"
                                self.after(0, self.update_lbl, self.lbl_api, "ATENÇÃO: Você tem BRL mas escolheu USDT!", "orange")
                            elif self.usdt_held > 2.0 and self.quote_currency != "USDT":
                                action = "Hold (Mude o par para BTC/USDT!)"
                                self.after(0, self.update_lbl, self.lbl_api, "ATENÇÃO: Você tem USDT mas escolheu BRL!", "orange")
                            else:
                                action = "Hold (Saldo Insuficiente)"
                            color_risk = "red"
                            fiat_spend = 0.0
                        else:
                            current_alloc = self.free_balance * alloc
                            total_fiat = current_alloc + getattr(self, 'acc_buy_fiat', 0.0)
                            min_order = 6.0 if self.quote_currency == "BRL" else 5.0
                            
                            if total_fiat < min_order:
                                self.acc_buy_fiat = total_fiat
                                action = f"Buy Acumulando... ({self.fiat_sym}{total_fiat:.2f})"
                                color_risk = "orange"
                                fiat_spend = 0.0
                            else:
                                self.acc_buy_fiat = 0.0
                                # Respeitar o buffer da Bybit para Market Orders
                                safe_max = self.free_balance * 0.95
                                fiat_spend = min(total_fiat, safe_max)
                        
                        if fiat_spend > 0:
                            if real_execution:
                                btc_amount = (fiat_spend / self.current_price) * (1 - fee_rate)
                                try:
                                    self.after(0, self.update_lbl, self.lbl_api, "Executando COMPRA na API...", "yellow")
                                    bybit.create_market_buy_order(self.exec_pair, btc_amount)
                                    self.trades_buy += 1
                                    self.last_trade_candle = current_candle_time
                                    self.last_buy_price = self.current_price
                                    time.sleep(1)
                                    r_bal = bybit.fetch_balance()
                                    self.balance = r_bal['total'].get(self.quote_currency, 0.0)
                                    self.btc_held = r_bal['total'].get('BTC', 0.0)
                                    self.free_balance = r_bal['free'].get(self.quote_currency, 0.0)
                                    self.free_btc = r_bal['free'].get('BTC', 0.0)
                                    
                                    log_msg = f"[{timestamp}] 🟢 COMPRA (Real): {btc_amount:.5f} BTC a {self.fiat_sym}{self.current_price:,.2f} | Gasto: {self.fiat_sym}{fiat_spend:,.2f}"
                                    self.trade_ledger.append(log_msg)
                                    
                                    self.after(0, self.update_lbl, self.lbl_api, "API: Ordem Executada", "green")
                                except Exception as ex:
                                    self.after(0, self.update_lbl, self.lbl_api, f"Falha (BUY): {str(ex)[:25]}", "red")
                                    self.after(0, lambda: self.var_live_trade.set(False))
                            else:
                                # Paper Trading Local
                                limit_price = self.current_price * (1 + slippage) 
                                btc_bought = (fiat_spend / limit_price) * (1 - fee_rate)
                                self.balance -= fiat_spend
                                self.btc_held += btc_bought
                                self.free_balance -= fiat_spend
                                self.free_btc += btc_bought
                                self.trades_buy += 1
                                self.last_trade_candle = current_candle_time
                                self.last_buy_price = limit_price
                                
                                log_msg = f"[{timestamp}] 🟩 COMPRA (Paper): {btc_bought:.5f} BTC a {self.fiat_sym}{limit_price:,.2f} | Gasto: {self.fiat_sym}{fiat_spend:,.2f}"
                                self.trade_ledger.append(log_msg)

                    elif action == "Sell":
                        if not hasattr(self, 'free_btc'): self.free_btc = self.btc_held
                        
                        if self.free_btc < 0.00001:
                            action = "Hold (Sem Moedas)"
                            color_risk = "orange"
                            btc_to_sell = 0.0
                        else:
                            min_order = 6.0 if self.quote_currency == "BRL" else 5.0
                            current_alloc_btc = self.free_btc * alloc
                            fiat_value = current_alloc_btc * self.current_price
                            total_btc = current_alloc_btc + getattr(self, 'acc_sell_btc', 0.0)
                            
                            if fiat_value < min_order:
                                self.acc_sell_btc = total_btc
                                action = f"Sell Acumulando... ({self.fiat_sym}{fiat_value:.2f})"
                                color_risk = "orange"
                                btc_to_sell = 0.0
                            else:
                                self.acc_sell_btc = 0.0
                                safe_max_btc = self.free_btc * 0.99
                                btc_to_sell = min(total_btc, safe_max_btc)
                                
                        btc_to_sell = round(btc_to_sell, 6)
                        
                        if btc_to_sell > 0.00001:
                            if real_execution:
                                try:
                                    self.after(0, self.update_lbl, self.lbl_api, "Executando VENDA na API...", "yellow")
                                    bybit.create_market_sell_order(self.exec_pair, btc_to_sell)
                                    self.trades_sell += 1
                                    self.last_trade_candle = current_candle_time
                                    time.sleep(1)
                                    r_bal = bybit.fetch_balance()
                                    self.balance = r_bal['total'].get(self.quote_currency, 0.0)
                                    self.btc_held = r_bal['total'].get('BTC', 0.0)
                                    self.free_balance = r_bal['free'].get(self.quote_currency, 0.0)
                                    self.free_btc = r_bal['free'].get('BTC', 0.0)
                                    
                                    fiat_gained = btc_to_sell * self.current_price
                                    log_msg = f"[{timestamp}] 🔴 VENDA (Real): {btc_to_sell:.5f} BTC a {self.fiat_sym}{self.current_price:,.2f} | Recebido: {self.fiat_sym}{fiat_gained:,.2f}"
                                    self.trade_ledger.append(log_msg)
                                    
                                    self.after(0, self.update_lbl, self.lbl_api, "API: Ordem Executada", "green")
                                except Exception as ex:
                                    self.after(0, self.update_lbl, self.lbl_api, f"Permissão Negada ou Falha (SELL): {str(ex)[:25]}", "red")
                                    self.after(0, lambda: self.var_live_trade.set(False))
                            else:
                                # Paper Trading Local
                                limit_price = self.current_price * (1 - slippage) 
                                usdt_gained = (btc_to_sell * limit_price) * (1 - fee_rate)
                                self.btc_held -= btc_to_sell
                                self.balance += usdt_gained
                                self.free_btc -= btc_to_sell
                                self.free_balance += usdt_gained
                                self.trades_sell += 1
                                self.last_trade_candle = current_candle_time
                                
                                log_msg = f"[{timestamp}] 🟥 VENDA (Paper): {btc_to_sell:.5f} BTC a {self.fiat_sym}{limit_price:,.2f} | Recebido: {self.fiat_sym}{usdt_gained:,.2f}"
                                self.trade_ledger.append(log_msg)
                    
                    # Update Visual
                    btc_fiat_value = self.btc_held * self.current_price
                    port_val = self.balance + btc_fiat_value
                    
                    if port_val > self.peak_equity and self.peak_equity > 0:
                        self.peak_equity = port_val
                        
                    if self.peak_equity > 0:
                        dd = (port_val / self.peak_equity) - 1.0
                        if dd < self.max_dd:
                            self.max_dd = dd
                        
                    alloc_pct = (btc_fiat_value / port_val) * 100 if port_val > 0 else 0
                    cash_pct = (self.balance / port_val) * 100 if port_val > 0 else 100
                        
                    macro_text = f"BUY {macro_probs[1]*100:.2f}%" if macro_probs[1] > 0.5 else f"SELL {macro_probs[0]*100:.2f}%"
                    macro_color = "green" if macro_probs[1] > 0.5 else "red"
                    
                    fee_rate = 0.001
                    breakeven_diff = self.current_price * ((1 / ((1 - fee_rate) ** 2)) - 1)
                    breakeven_text = f"+{self.fiat_sym}{breakeven_diff:,.2f}"
                    if hasattr(self, 'last_buy_price'):
                        breakeven_text += f"\nÚltima Compra: {self.fiat_sym}{self.last_buy_price:,.2f}"
                    else:
                        breakeven_text += "\nÚltima Compra: ---"
                    breakeven_color = "cyan"
                    
                    self.after(0, self.update_gui, timestamp, port_val, self.current_price, macro_text, macro_color, breakeven_text, breakeven_color, action, color_risk, alloc_pct, cash_pct)
                    print(f"[DEBUG] update_gui agendado para {timestamp}")

            except Exception as e:
                print(f"[DEBUG] EXCEPTION: {e}")
                self.in_circuit_breaker = True
                self.after(0, self.update_lbl, self.lbl_api, f"Circuit Breaker (Rede): {str(e)[:30]}", "orange")
                time.sleep(5) 
                
            # Polling suave
            time.sleep(3) 

    def update_gui(self, timestamp, port_val, btc_price, mac_txt, mac_c, breakeven_txt, breakeven_c, act, act_c, alloc_pct, cash_pct):
        self.lbl_macro_state.configure(text=mac_txt, text_color=mac_c)
        self.lbl_breakeven_state.configure(text=breakeven_txt, text_color=breakeven_c)
        self.lbl_risk_state.configure(text=act, text_color=act_c)
        
        mode = "Reais" if self.var_live_trade.get() else "Simuladas"
        roi = ((port_val / self.initial_balance) - 1.0) * 100 if self.initial_balance > 0 else 0.0
        
        if mode == "Simuladas":
            profit = port_val - self.initial_balance
            profit_str = f"+{self.fiat_sym}{profit:,.2f}" if profit >= 0 else f"-{self.fiat_sym}{abs(profit):,.2f}"
            self.lbl_portfolio.configure(text=f"Patrimônio Simulado: {self.fiat_sym}{port_val:,.2f} (Lucro: {profit_str} | {roi:+.2f}%)")
        else:
            profit = port_val - self.initial_balance
            profit_str = f"+{self.fiat_sym}{profit:,.2f}" if profit >= 0 else f"-{self.fiat_sym}{abs(profit):,.2f}"
            self.lbl_portfolio.configure(text=f"Patrimônio (Real): {self.fiat_sym}{port_val:,.2f} (Lucro: {profit_str} | {roi:+.2f}%)\nBalanços: {self.btc_held:.5f} BTC | $ {self.usdt_held:.2f} USDT | R$ {self.brl_held:.2f} BRL")
            
        self.lbl_dynamic_balance.configure(text=f"Alocação no Par {self.exec_pair}: Caixa = {self.fiat_sym}{self.balance:,.2f} ({cash_pct:.1f}%) | Moedas = {self.btc_held:.5f} BTC ({alloc_pct:.1f}%) | Cotação = {self.fiat_sym}{btc_price:,.2f}")
        
        self.lbl_trades.configure(text=f"Ordens {mode}: {self.trades_buy + self.trades_sell} ({self.trades_buy} C / {self.trades_sell} V)")
        self.lbl_mdd.configure(text=f"Max Drawdown: {self.max_dd*100:.2f}%")
        
        self.history_time.append(timestamp)
        self.history_portfolio.append(port_val)
        self.history_cash.append(self.balance)
        self.history_alloc_pct.append(alloc_pct)
        self.history_price.append(btc_price)
        
        norm_btc = getattr(self, 'baseline_btc', 0.0) * btc_price if getattr(self, 'baseline_btc', 0.0) > 0 else port_val
        self.history_btc.append(norm_btc)
        
        if getattr(self, '_cached_live_trade', False):
            try:
                with open("history_log.json", "w") as f:
                    json.dump({
                        "time": self.history_time,
                        "portfolio": self.history_portfolio,
                        "cash": self.history_cash,
                        "alloc": self.history_alloc_pct,
                        "price": self.history_price,
                        "btc": self.history_btc,
                        "initial_btc_price": self.initial_btc_price,
                        "initial_balance": self.initial_balance,
                        "peak_equity": self.peak_equity,
                        "max_dd": self.max_dd,
                        "trades_buy": self.trades_buy,
                        "trades_sell": self.trades_sell,
                        "last_saved_balance": self.balance,
                        "last_saved_btc": self.btc_held,
                        "baseline_btc": getattr(self, 'baseline_btc', 0.0)
                    }, f)
            except Exception:
                pass
            
        tf_mode = self.timeframe_var.get()
        
        from datetime import datetime, timedelta
        try:
            full_time_dt = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in self.history_time]
        except Exception:
            full_time_dt = self.history_time
            
        now = datetime.now()
        cutoff = None
        if tf_mode == "1 Hora":
            cutoff = now - timedelta(hours=1)
        elif tf_mode == "1 Dia":
            cutoff = now - timedelta(days=1)
        elif tf_mode == "1 Semana":
            cutoff = now - timedelta(weeks=1)
        elif tf_mode == "1 Mês":
            cutoff = now - timedelta(days=30)
            
        if cutoff and isinstance(full_time_dt[0], datetime) if len(full_time_dt) > 0 else False:
            idx = 0
            for i, t in enumerate(full_time_dt):
                if t >= cutoff:
                    idx = i
                    break
                idx = len(full_time_dt) # if none found, empty or just the last point
                
            plot_time_dt = full_time_dt[idx:]
            plot_port = self.history_portfolio[idx:]
            plot_btc = self.history_btc[idx:]
            plot_cash = self.history_cash[idx:]
            plot_alloc = self.history_alloc_pct[idx:]
            plot_price = self.history_price[idx:]
        else:
            plot_time_dt = full_time_dt
            plot_port = self.history_portfolio
            plot_btc = self.history_btc
            plot_cash = self.history_cash
            plot_alloc = self.history_alloc_pct
            plot_price = self.history_price
            
        if self.var_plot_in_btc.get():
            plot_port = [p / pr if pr > 0 else 0 for p, pr in zip(plot_port, plot_price)]
            plot_btc = [b / pr if pr > 0 else 0 for b, pr in zip(plot_btc, plot_price)]
            plot_cash = [c / pr if pr > 0 else 0 for c, pr in zip(plot_cash, plot_price)]
            
        self.ax.clear()
        if self.ax2 is not None:
            self.ax2.remove()
            self.ax2 = None
            
        self.annot = self.ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                                      bbox=dict(boxstyle="round4", fc="#333333", ec="white", lw=1),
                                      arrowprops=dict(arrowstyle="->", color="white"), color="white")
        self.annot.set_visible(False)
            
        if self.var_plot_port.get():
            self.ax.plot(plot_time_dt, plot_port, color='#00ff88', label='Patrimônio (IA)', linewidth=2)
            
        if self.var_plot_btc.get():
            self.ax.plot(plot_time_dt, plot_btc, color='gray', linestyle='--', label='BTC (HODL)', picker=True)
            
        if self.var_plot_cash.get():
            self.ax.plot(plot_time_dt, plot_cash, color='#ffcc99', linestyle='-', label='Caixa Livre')
            
        if self.var_plot_alloc.get():
            self.ax2 = self.ax.twinx()
            self.ax2.plot(plot_time_dt, plot_alloc, color='#e066ff', linestyle='-', label='Alocação em BTC (%)', alpha=0.6)
            self.ax2.set_ylim(-5, 105)
            self.ax2.tick_params(colors='#e066ff')
            
        self.ax.set_title(f"Ação Real do Mercado | BTC: {self.fiat_sym}{btc_price:,.2f}", color="white")
        self.ax.tick_params(colors='white')
        
        lines, labels = self.ax.get_legend_handles_labels()
        if self.ax2 is not None:
            lines2, labels2 = self.ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
            
        if lines:
            self.ax.legend(lines, labels, facecolor='#2b2b2b', edgecolor='white', labelcolor='white', loc='upper left')
        
        import matplotlib.ticker as ticker
        if len(plot_time_dt) > 0 and isinstance(plot_time_dt[0], datetime):
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            
        self.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)
        self.fig.tight_layout()
        
        self.canvas.draw()

if __name__ == "__main__":
    app = HybridBotApp()
    app.mainloop()

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import ccxt
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime
import os

from model import TimeSeriesTransformer
from data_collection import feature_engineering

# Configuração visual
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class CryptoBotApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🤖 Simulador Dry-Run - Bot Cripto Ao Vivo")
        self.geometry("900x600")

        # Layout da Interface
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- Sidebar (Esquerda) --
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Configurações", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.lbl_balance = ctk.CTkLabel(self.sidebar_frame, text="Saldo Inicial (USDT):")
        self.lbl_balance.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.entry_balance = ctk.CTkEntry(self.sidebar_frame, placeholder_text="1000")
        self.entry_balance.insert(0, "1000")
        self.entry_balance.grid(row=2, column=0, padx=20, pady=(5, 10), sticky="ew")

        self.lbl_trade = ctk.CTkLabel(self.sidebar_frame, text="Valor por Trade (USDT):")
        self.lbl_trade.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.entry_trade = ctk.CTkEntry(self.sidebar_frame, placeholder_text="100")
        self.entry_trade.insert(0, "100")
        self.entry_trade.grid(row=4, column=0, padx=20, pady=(5, 20), sticky="ew")
        
        self.lbl_fee = ctk.CTkLabel(self.sidebar_frame, text="Taxa (%):")
        self.lbl_fee.grid(row=5, column=0, padx=20, pady=(0, 0), sticky="w")
        self.entry_fee = ctk.CTkEntry(self.sidebar_frame, placeholder_text="0.1")
        self.entry_fee.insert(0, "0.1")
        self.entry_fee.grid(row=6, column=0, padx=20, pady=(5, 10), sticky="ew")

        self.lbl_conf = ctk.CTkLabel(self.sidebar_frame, text="Nível de Confiança (%):")
        self.lbl_conf.grid(row=7, column=0, padx=20, pady=(0, 0), sticky="w")
        self.entry_conf = ctk.CTkEntry(self.sidebar_frame, placeholder_text="60")
        self.entry_conf.insert(0, "60")
        self.entry_conf.grid(row=8, column=0, padx=20, pady=(5, 20), sticky="ew")

        self.btn_start = ctk.CTkButton(self.sidebar_frame, text="▶ Iniciar Dry-Run", command=self.start_simulation)
        self.btn_start.grid(row=9, column=0, padx=20, pady=10, sticky="ew")

        self.btn_stop = ctk.CTkButton(self.sidebar_frame, text="⏹ Parar", command=self.stop_simulation, state="disabled", fg_color="red", hover_color="darkred")
        self.btn_stop.grid(row=10, column=0, padx=20, pady=20, sticky="ew")

        # -- Main View (Direita) --
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Labels de Resumo
        self.summary_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.summary_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        
        self.lbl_current_balance = ctk.CTkLabel(self.summary_frame, text="Saldo Atual: $1000.00", font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_current_balance.pack(side="left", padx=20)
        
        self.lbl_status = ctk.CTkLabel(self.summary_frame, text="Status: Aguardando...", text_color="gray")
        self.lbl_status.pack(side="right", padx=20)

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4), facecolor='#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_color('#2b2b2b')
        self.ax.spines['right'].set_color('#2b2b2b')
        self.ax.set_title("Variação do Portfólio Ao Vivo", color="white")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Variáveis de Estado
        self.is_running = False
        self.history_time = []
        self.history_portfolio = []
        self.history_decision = []
        self.line = None

        self.annot = self.ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="#333333", ec="white", alpha=0.9),
                                      arrowprops=dict(arrowstyle="->", color="white"))
        self.annot.set_color("white")
        self.annot.set_visible(False)
        self.canvas.mpl_connect("motion_notify_event", self.hover)
        
        self.balance_usdt = 1000.0
        self.balance_btc = 0.0
        self.trade_amount = 100.0
        self.fee = 0.001
        
        # ML
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        
    def start_simulation(self):
        if not os.path.exists('models/best_model.pt') or not os.path.exists('models/scaler.pkl'):
            self.lbl_status.configure(text="Erro: Modelo não encontrado!", text_color="red")
            return
            
        try:
            self.balance_usdt = float(self.entry_balance.get())
            self.trade_amount = float(self.entry_trade.get())
            self.fee = float(self.entry_fee.get()) / 100.0
            self.confidence_threshold = float(self.entry_conf.get()) / 100.0
        except ValueError:
            self.lbl_status.configure(text="Erro: Valores inválidos!", text_color="red")
            return
            
        self.balance_btc = 0.0
        self.history_time = []
        self.history_portfolio = []
        self.history_decision = []
        
        self.model = TimeSeriesTransformer(num_features=12).to(self.device)
        self.model.load_state_dict(torch.load('models/best_model.pt', map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load('models/scaler.pkl')

        self.btn_start.configure(state="disabled")
        self.entry_balance.configure(state="disabled")
        self.entry_trade.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.lbl_status.configure(text="Status: Conectando...", text_color="yellow")

        self.is_running = True
        self.thread = threading.Thread(target=self.bot_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop_simulation(self):
        self.is_running = False
        self.btn_start.configure(state="normal")
        self.entry_balance.configure(state="normal")
        self.entry_trade.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.lbl_status.configure(text="Status: Parado", text_color="gray")

    def update_chart(self, timestamp, portfolio_value, prediction_text):
        self.history_time.append(timestamp)
        self.history_portfolio.append(portfolio_value)
        self.history_decision.append(prediction_text)
        
        self.lbl_current_balance.configure(text=f"Saldo Atual: ${portfolio_value:,.2f}")
        self.lbl_status.configure(text=f"Última Decisão: {prediction_text} | Aguardando...", text_color="green")

        self.ax.clear()
        self.line, = self.ax.plot(self.history_time, self.history_portfolio, color='#00ff88', marker='o')
        self.ax.set_title("Variação do Portfólio Ao Vivo", color="white")
        self.ax.tick_params(colors='white')
        self.fig.autofmt_xdate()

        self.annot = self.ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points", 
                                      bbox=dict(boxstyle="round", fc="#333333", ec="white", alpha=0.9),
                                      arrowprops=dict(arrowstyle="->", color="white"))
        self.annot.set_color("white")
        self.annot.set_visible(False)

        self.canvas.draw()

    def update_annot(self, ind):
        x_index = ind["ind"][0]
        x, y = self.line.get_data()
        self.annot.xy = (x[x_index], y[x_index])
        
        dec = self.history_decision[x_index]
        val = self.history_portfolio[x_index]
        text = f"Decisão: {dec}\nSaldo: ${val:.2f}"
        
        if dec == "Buy":
            self.annot.get_bbox_patch().set_edgecolor("#00ff88")
        elif dec == "Sell":
            self.annot.get_bbox_patch().set_edgecolor("#ff4444")
        else:
            self.annot.get_bbox_patch().set_edgecolor("gray")
            
        self.annot.set_text(text)

    def hover(self, event):
        if self.line is None: return
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.line.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()

    def bot_loop(self):
        exchange = ccxt.bybit({'enableRateLimit': True})
        symbol = 'BTC/USDT'
        timeframe = '15m'
        seq_len = 60
        features_list = [
            'log_return', 'rsi', 'macd', 'macd_signal', 'macd_diff',
            'dist_sma_20', 'dist_sma_50', 'dist_sma_200',
            'bb_width', 'bb_percent', 'atr', 'volume_ratio'
        ]
        
        last_processed_candle = None
        
        while self.is_running:
            try:
                # Busca as últimas 300 velas para ter histórico suficiente pros indicadores
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=300)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Descartamos a última vela que ainda não fechou
                df = df.iloc[:-1]
                latest_candle_time = df.index[-1]
                
                if last_processed_candle == latest_candle_time:
                    # Nenhuma vela nova. Dorme 30 segundos.
                    time.sleep(30)
                    continue
                    
                # Nova vela detectada! Processar:
                self.lbl_status.configure(text="Processando nova vela...", text_color="cyan")
                
                df_features = feature_engineering(df.copy())
                df_features.dropna(inplace=True)
                
                if len(df_features) >= seq_len:
                    df_scaled = df_features.copy()
                    df_scaled[features_list] = self.scaler.transform(df_features[features_list])
                    
                    seq_data = df_scaled[features_list].values[-seq_len:]
                    x_tensor = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(x_tensor)
                        probs = torch.softmax(outputs, dim=1).squeeze()
                        
                        prob_sell = probs[0].item()
                        prob_buy = probs[1].item()
                        
                        if prob_buy >= self.confidence_threshold:
                            pred = 1 # Buy
                        elif prob_sell >= self.confidence_threshold:
                            pred = 0 # Sell
                        else:
                            pred = -1 # Hold
                        
                    current_price = df['close'].iloc[-1]
                    
                    # Logica de Trade
                    decision = "Hold"
                    if pred == 1 and self.balance_usdt > 0:
                        decision = "Buy"
                        amount = min(self.trade_amount, self.balance_usdt)
                        btc_bought = (amount * (1 - self.fee)) / current_price
                        self.balance_usdt -= amount
                        self.balance_btc += btc_bought
                    elif pred == 0 and self.balance_btc > 0:
                        decision = "Sell"
                        btc_val = self.balance_btc * current_price
                        amount = min(self.trade_amount, btc_val)
                        btc_to_sell = amount / current_price
                        
                        usdt_received = amount * (1 - self.fee)
                        self.balance_btc -= btc_to_sell
                        self.balance_usdt += usdt_received
                        
                    portfolio_value = self.balance_usdt + (self.balance_btc * current_price)
                    
                    # Atualiza GUI no main thread
                    self.after(0, self.update_chart, latest_candle_time.strftime("%H:%M"), portfolio_value, decision)
                else:
                    self.lbl_status.configure(text=f"Erro: Dados insuficientes ({len(df_features)} velas).", text_color="red")
                    
                last_processed_candle = latest_candle_time
                
            except Exception as e:
                print(f"Erro: {e}")
                time.sleep(10)

if __name__ == "__main__":
    app = CryptoBotApp()
    app.mainloop()

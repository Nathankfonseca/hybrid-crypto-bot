class RiskManager:
    def __init__(self, max_risk_per_trade_pct=0.01, kelly_fraction=0.5, macro_intensity=1.0):
        self.max_risk_pct = max_risk_per_trade_pct
        self.kelly_fraction = kelly_fraction
        self.circuit_breaker_active = False
        self.micro_threshold = 0.45 
        self.macro_intensity = macro_intensity
        
    def evaluate_signal(self, macro_probs, micro_probs, atr_volatility):
        """
        Avalia a interseção dos modelos com viés probabilístico multiplicativo.
        
        macro_probs: [Bearish, Neutral, Bullish] -> índices [0, 1, 2]
        micro_probs: [Sell, Buy, Hold] -> índices [0, 1, 2]
        atr_volatility: métrica Z-Score do ATR recente
        """
        
        # 1. Circuit Breaker Override
        if self.circuit_breaker_active:
            return "Hold", 0.0, "Circuit Breaker Ativo"
            
        macro_sell, macro_buy = macro_probs
        micro_sell, micro_buy = micro_probs
        
        # 2. Aplicação do Viés (Bias Factor)
        # O modelo Macro atua como um multiplicador nas probabilidades do Micro.
        
        buy_bias = 1.0 + (macro_buy * 0.5 * self.macro_intensity) - (macro_sell * 0.3 * self.macro_intensity)
        sell_bias = 1.0 + (macro_sell * 0.5 * self.macro_intensity) - (macro_buy * 0.3 * self.macro_intensity)
        
        adj_micro_buy = micro_buy * buy_bias
        adj_micro_sell = micro_sell * sell_bias
        
        action = "Hold"
        reason = "Aguardando Oportunidade"
        win_rate = 0.0
        
        # A probabilidade direcional enviesada deve ser superior ao micro_threshold dinâmico
        
        if adj_micro_buy > self.micro_threshold and adj_micro_buy > adj_micro_sell:
            action = "Buy"
            win_rate = micro_buy # Kelly será baseado na confiança direcional bruta
            reason = f"Micro Buy (Raw: {micro_buy:.2f} | Adj: {adj_micro_buy:.2f})"
            if macro_buy > 0.5:
                reason += " [Macro Aliado]"
            else:
                reason += " [Macro Divergente]"
                
        elif adj_micro_sell > self.micro_threshold and adj_micro_sell > adj_micro_buy:
            action = "Sell"
            win_rate = micro_sell
            reason = f"Micro Sell (Raw: {micro_sell:.2f} | Adj: {adj_micro_sell:.2f})"
            if macro_sell > 0.5:
                reason += " [Macro Aliado]"
            else:
                reason += " [Macro Divergente]"
                
        if action == "Hold":
            return action, 0.0, reason
            
        # 3. Position Sizing Dinâmico (Fractional Kelly)
        r_r_ratio = 1.5 
        
        if win_rate > 0.5:
            # Formula de Kelly: K = W - ((1 - W) / R)
            kelly_pct = win_rate - ((1.0 - win_rate) / r_r_ratio)
            alloc_pct = kelly_pct * self.kelly_fraction
        else:
            alloc_pct = 0.01 
            
        # Ajuste de Volatilidade (Regime Switching)
        if atr_volatility > 2.0:
            alloc_pct = alloc_pct * 0.5
            reason += " | Volatilidade Crítica (Posição Cortada)"
            
        # Limit to max risk defined globally
        final_alloc = min(max(alloc_pct, 0), self.max_risk_pct)
        
        return action, final_alloc, reason

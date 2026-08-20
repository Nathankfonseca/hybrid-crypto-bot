import re
import os

with open("hybrid_desktop_app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix self.after configure bugs
if "def update_lbl(self, lbl, txt, col):" not in content:
    helper = """
    def update_lbl(self, lbl, txt, col):
        lbl.configure(text=txt, text_color=col)
"""
    # Insert it right before def poll_ui_variables
    content = content.replace("    def poll_ui_variables(self):", helper + "\n    def poll_ui_variables(self):")

# Now replace the after calls to use update_lbl
def replace_after(match):
    lbl = match.group(1)
    txt = match.group(2)
    col = match.group(3)
    return f'self.after(0, self.update_lbl, {lbl}, {txt}, {col})'

pattern = r'self\.after\(0, (self\.lbl_[a-z_]+)\.configure, \{"text": (.*?), "text_color": (.*?)\}\)'
content = re.sub(pattern, replace_after, content)

# Fix var_live_trade.set
content = content.replace("self.after(0, self.var_live_trade.set, False)", "self.after(0, lambda: self.var_live_trade.set(False))")
content = content.replace("self.after(0, self.stop_simulation)", "self.after(0, lambda: self.stop_simulation())")


# Fix variables and logic limits for execution
content = content.replace("usdt_spend", "fiat_spend")

content = content.replace("if action == \"Buy\" and self.balance > 10:", "if action == \"Buy\" and self.balance > 1.0:")
content = content.replace("if fiat_spend > 10:", "if fiat_spend > 1.0:")

content = content.replace("elif action == \"Sell\" and (self.btc_held * self.current_price) > 10:", "elif action == \"Sell\" and (self.btc_held * self.current_price) > 1.0:")
content = content.replace("if btc_to_sell > 0.0001:", "if btc_to_sell > 0.00001:")
content = content.replace("round(btc_to_sell, 5)", "round(btc_to_sell, 6)")

with open("hybrid_desktop_app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed!")

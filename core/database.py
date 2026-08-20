import duckdb
import os
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path='v2_data.duckdb'):
        self.db_path = db_path
        self.conn = duckdb.connect(self.db_path)
        self.init_db()

    def init_db(self):
        # Create table for Volume Bars
        self.conn.execute('''
            CREATE SEQUENCE IF NOT EXISTS seq_volume_bar_id;
            CREATE TABLE IF NOT EXISTS volume_bars (
                id INTEGER DEFAULT nextval('seq_volume_bar_id') PRIMARY KEY,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                tick_count INTEGER,
                ofi DOUBLE -- Order Flow Imbalance
            )
        ''')
        
    def insert_volume_bar(self, start_time, end_time, open_p, high_p, low_p, close_p, volume, tick_count, ofi):
        self.conn.execute('''
            INSERT INTO volume_bars (start_time, end_time, open, high, low, close, volume, tick_count, ofi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (start_time, end_time, open_p, high_p, low_p, close_p, volume, tick_count, ofi))
        
    def get_latest_bars(self, limit=1000):
        return self.conn.execute('''
            SELECT * FROM volume_bars ORDER BY id DESC LIMIT ?
        ''', (limit,)).df()

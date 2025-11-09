# src/features.py
"""
Feature Engineering: 20+ indicadores técnicos categorizados
"""
import pandas as pd
import numpy as np
import yaml
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Clase para crear features técnicos"""
    
    def __init__(self, config):
        self.config = config
        
    def add_momentum_features(self, df):
        """
        Features de momentum: RSI, ROC, Stochastic, etc.
        """
        # RSI (Relative Strength Index)
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        df['rsi_30'] = self.calculate_rsi(df['close'], 30)
        
        # ROC (Rate of Change)
        df['roc_10'] = df['close'].pct_change(10) * 100
        df['roc_20'] = df['close'].pct_change(20) * 100
        
        # Stochastic Oscillator
        df['stoch_k'] = self.calculate_stochastic(df, 14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # CCI (Commodity Channel Index)
        df['cci'] = self.calculate_cci(df, 20)
        
        return df
    
    def add_volatility_features(self, df):
        """
        Features de volatilidad: ATR, Bollinger, volatilidad rolling
        """
        # ATR (Average True Range)
        df['atr_14'] = self.calculate_atr(df, 14)
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (2 * std_20)
        df['bb_lower'] = sma_20 - (2 * std_20)
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
        
        # Volatilidad rolling
        df['volatility_10'] = df['close'].pct_change().rolling(10).std()
        df['volatility_30'] = df['close'].pct_change().rolling(30).std()
        
        # ADX (Average Directional Index)
        df['adx'] = self.calculate_adx(df, 14)
        
        return df
    
    def add_volume_features(self, df):
        """
        Features de volumen: OBV, MFI, z-score volumen
        """
        # OBV (On Balance Volume)
        df['obv'] = self.calculate_obv(df)
        
        # MFI (Money Flow Index)
        df['mfi'] = self.calculate_mfi(df, 14)
        
        # Volume Z-score
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
        
        # Volume MA ratio
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        return df
    
    def add_price_features(self, df):
        """
        Features adicionales de precio: SMA, EMA, returns
        """
        # Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
        # Price ratios
        df['price_to_sma20'] = df['close'] / df['sma_20']
        df['price_to_sma50'] = df['close'] / df['sma_50']
        
        # Returns
        df['return_1d'] = df['close'].pct_change()
        df['return_5d'] = df['close'].pct_change(5)
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stochastic(self, df, period=14):
        """Calcula Stochastic %K"""
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        return stoch_k
    
    def calculate_atr(self, df, period=14):
        """Calcula Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def calculate_cci(self, df, period=20):
        """Calcula Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (typical_price - sma) / (0.015 * mad)
        return cci
    
    def calculate_adx(self, df, period=14):
        """Calcula Average Directional Index (simplificado)"""
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self.calculate_atr(df, 1)  # True Range
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return adx
    
    def calculate_obv(self, df):
        """Calcula On Balance Volume"""
        obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        return obv
    
    def calculate_mfi(self, df, period=14):
        """Calcula Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(), 0)
        
        positive_flow_sum = positive_flow.rolling(period).sum()
        negative_flow_sum = negative_flow.rolling(period).sum()
        
        money_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi
    
    def normalize_features(self, df, train_idx, test_idx, val_idx):
        """
        Normalización: fit en train, transform en test/val
        """
        feature_cols = [col for col in df.columns if col not in 
                       ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']]
        
        # Calcular estadísticas solo en train
        train_start, train_end = train_idx
        train_data = df.iloc[train_start:train_end]
        
        # Guardar estadísticas
        stats = {}
        for col in feature_cols:
            stats[col] = {
                'mean': train_data[col].mean(),
                'std': train_data[col].std()
            }
        
        # Aplicar normalización
        for col in feature_cols:
            if stats[col]['std'] > 0:
                df[f'{col}_norm'] = (df[col] - stats[col]['mean']) / stats[col]['std']
            else:
                df[f'{col}_norm'] = 0
        
        # Guardar estadísticas para inferencia
        with open('data/processed/normalization_stats.pkl', 'wb') as f:
            pickle.dump(stats, f)
        
        return df
    
    def create_features(self):
        """Pipeline principal de creación de features"""
        # Cargar datos
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        df = pd.read_csv(f"data/raw/{config['asset']}.csv")
        df['date'] = pd.to_datetime(df['date'])
        
        # Cargar índices de splits
        with open('data/processed/splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        
        logger.info("Creando features...")
        
        # Agregar todas las categorías de features
        df = self.add_price_features(df)
        df = self.add_momentum_features(df)
        df = self.add_volatility_features(df)
        df = self.add_volume_features(df)
        
        # Eliminar filas con NaN (debido a rolling windows)
        df = df.dropna()
        
        # Normalizar features
        df = self.normalize_features(
            df,
            splits['train_idx'],
            splits['test_idx'],
            splits['val_idx']
        )
        
        # Seleccionar solo features normalizados para el modelo
        feature_cols = [col for col in df.columns if col.endswith('_norm')]
        
        logger.info(f"Total de features creados: {len(feature_cols)}")
        logger.info(f"Features: {feature_cols[:10]}...")  # Mostrar primeros 10
        
        # Guardar dataset con features
        df.to_parquet('data/processed/features.parquet', index=False)
        
        # Guardar lista de features
        with open('data/processed/feature_columns.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        
        logger.info(f"Features guardados en data/processed/features.parquet")
        logger.info(f"Shape final: {df.shape}")
        
        return df

def main():
    """Ejecutar feature engineering"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    engineer = FeatureEngineer(config)
    engineer.create_features()

if __name__ == "__main__":
    main()
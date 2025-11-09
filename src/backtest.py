# src/backtest.py
"""
Backtest con costos realistas de trading
"""
import pandas as pd
import numpy as np
import yaml
import pickle
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Backtester:
    """Backtesting con costos de transacción"""
    
    def __init__(self, config):
        self.config = config
        self.backtest_config = config['backtest']
        self.initial_capital = self.backtest_config['initial_capital']
        self.n_shares = self.backtest_config['n_shares']
        self.stop_loss = self.backtest_config['stop_loss']
        self.take_profit = self.backtest_config['take_profit']
        self.commission = self.backtest_config['commission']
        self.borrow_rate = self.backtest_config['borrow_rate']
        
    def calculate_commission(self, price, shares):
        """Calcula comisión por transacción"""
        return price * shares * self.commission
    
    def calculate_borrow_cost(self, price, shares, days):
        """Calcula costo de préstamo para posiciones cortas"""
        annual_cost = price * shares * self.borrow_rate
        daily_cost = annual_cost / 365
        return daily_cost * days
    
    def execute_backtest(self, df, signals):
        """
        Ejecuta backtest con las señales generadas
        
        Args:
            df: DataFrame con precios
            signals: DataFrame con señales de trading
            
        Returns:
            Diccionario con resultados y métricas
        """
        # Preparar datos
        df = df.copy()
        df['signal'] = 'hold'
        
        # Mapear señales a fechas
        test_start_idx = len(df) - len(signals)
        df.iloc[test_start_idx:, df.columns.get_loc('signal')] = signals['signal'].values
        
        # Variables de estado
        cash = self.initial_capital
        position = 0  # 1: long, -1: short, 0: neutral
        shares = 0
        entry_price = 0
        entry_date = None
        
        # Tracking
        trades = []
        equity_curve = []
        
        for i in range(test_start_idx, len(df)):
            row = df.iloc[i]
            current_price = row['close']
            current_signal = row['signal']
            current_date = row['date']
            
            # Calcular equity actual
            if position == 1:  # Long
                current_equity = cash + (shares * current_price)
            elif position == -1:  # Short
                current_equity = cash - (shares * current_price)
            else:
                current_equity = cash
            
            equity_curve.append({
                'date': current_date,
                'equity': current_equity,
                'cash': cash,
                'position': position
            })
            
            # Check stop loss / take profit
            if position != 0 and entry_price > 0:
                if position == 1:  # Long position
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    if pnl_pct <= -self.stop_loss or pnl_pct >= self.take_profit:
                        # Cerrar posición
                        gross_pnl = shares * (current_price - entry_price)
                        commission = self.calculate_commission(current_price, shares)
                        net_pnl = gross_pnl - commission
                        
                        cash += shares * current_price - commission
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': current_date,
                            'type': 'long',
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'shares': shares,
                            'gross_pnl': gross_pnl,
                            'commission': commission * 2,  # Entrada + salida
                            'net_pnl': net_pnl,
                            'return_pct': pnl_pct
                        })
                        
                        position = 0
                        shares = 0
                        entry_price = 0
                        entry_date = None
                        
                elif position == -1:  # Short position
                    pnl_pct = (entry_price - current_price) / entry_price
                    
                    if pnl_pct <= -self.stop_loss or pnl_pct >= self.take_profit:
                        # Cerrar posición
                        gross_pnl = shares * (entry_price - current_price)
                        
                        # Calcular días en posición para borrow cost
                        days_held = (current_date - entry_date).days
                        borrow_cost = self.calculate_borrow_cost(
                            entry_price, shares, days_held
                        )
                        commission = self.calculate_commission(current_price, shares)
                        net_pnl = gross_pnl - commission - borrow_cost
                        
                        cash += shares * entry_price  # Devolver préstamo
                        cash -= shares * current_price  # Comprar para cubrir
                        cash -= commission
                        cash -= borrow_cost
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': current_date,
                            'type': 'short',
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'shares': shares,
                            'gross_pnl': gross_pnl,
                            'commission': commission * 2,
                            'borrow_cost': borrow_cost,
                            'net_pnl': net_pnl,
                            'return_pct': pnl_pct
                        })
                        
                        position = 0
                        shares = 0
                        entry_price = 0
                        entry_date = None
            
            # Procesar nuevas señales
            if position == 0:  # Sin posición
                if current_signal == 'long':
                    # Abrir long
                    shares = self.n_shares
                    commission = self.calculate_commission(current_price, shares)
                    
                    if cash >= shares * current_price + commission:
                        cash -= shares * current_price + commission
                        position = 1
                        entry_price = current_price
                        entry_date = current_date
                        
                elif current_signal == 'short':
                    # Abrir short
                    shares = self.n_shares
                    commission = self.calculate_commission(current_price, shares)
                    
                    cash += shares * current_price - commission  # Recibir préstamo
                    position = -1
                    entry_price = current_price
                    entry_date = current_date
        
        # Cerrar posición final si existe
        if position != 0:
            final_price = df.iloc[-1]['close']
            
            if position == 1:
                gross_pnl = shares * (final_price - entry_price)
                commission = self.calculate_commission(final_price, shares)
                net_pnl = gross_pnl - commission
                cash += shares * final_price - commission
                
            else:  # Short
                gross_pnl = shares * (entry_price - final_price)
                days_held = (df.iloc[-1]['date'] - entry_date).days
                borrow_cost = self.calculate_borrow_cost(entry_price, shares, days_held)
                commission = self.calculate_commission(final_price, shares)
                net_pnl = gross_pnl - commission - borrow_cost
                
                cash += shares * entry_price
                cash -= shares * final_price
                cash -= commission
                cash -= borrow_cost
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': df.iloc[-1]['date'],
                'type': 'long' if position == 1 else 'short',
                'entry_price': entry_price,
                'exit_price': final_price,
                'shares': shares,
                'gross_pnl': gross_pnl,
                'commission': commission * 2,
                'borrow_cost': borrow_cost if position == -1 else 0,
                'net_pnl': net_pnl,
                'return_pct': (final_price - entry_price) / entry_price if position == 1 
                             else (entry_price - final_price) / entry_price
            })
        
        # Calcular métricas
        metrics = self.calculate_metrics(trades, equity_curve)
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics
        }
    
    def calculate_metrics(self, trades, equity_curve):
        """Calcula métricas de performance"""
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'max_drawdown': 0
            }
        
        # Convertir a DataFrame
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        # Métricas básicas
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        win_rate = winning_trades / total_trades
        
        # Return total
        initial_equity = self.initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Returns diarios
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Sharpe Ratio (anualizado)
        daily_returns = equity_df['returns'].dropna()
        if len(daily_returns) > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio (solo downside volatility)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            sortino_ratio = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # Maximum Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # Calmar Ratio
        if max_drawdown != 0:
            annualized_return = total_return * (252 / len(equity_df))
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = 0
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'avg_trade_return': trades_df['return_pct'].mean() * 100,
            'total_commission': trades_df['commission'].sum(),
            'total_borrow_cost': trades_df['borrow_cost'].sum() if 'borrow_cost' in trades_df else 0
        }
        
        return metrics
    
    def plot_results(self, equity_curve, trades):
        """Genera gráficos de resultados"""
        equity_df = pd.DataFrame(equity_curve)
        trades_df = pd.DataFrame(trades) if len(trades) > 0 else pd.DataFrame()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Equity Curve
        axes[0, 0].plot(equity_df['date'], equity_df['equity'], label='Portfolio Value')
        axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        axes[0, 1].fill_between(equity_df['date'], equity_df['drawdown'] * 100, 0, color='red', alpha=0.3)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)
        
        # 3. Distribution of Returns
        if not trades_df.empty:
            axes[1, 0].hist(trades_df['return_pct'] * 100, bins=30, edgecolor='black')
            axes[1, 0].axvline(x=0, color='r', linestyle='--')
            axes[1, 0].set_title('Distribution of Trade Returns')
            axes[1, 0].set_xlabel('Return (%)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)
        
        # 4. Cumulative Trades
        if not trades_df.empty:
            trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
            axes[1, 1].plot(range(len(trades_df)), trades_df['cumulative_pnl'])
            axes[1, 1].set_title('Cumulative P&L')
            axes[1, 1].set_xlabel('Trade Number')
            axes[1, 1].set_ylabel('Cumulative P&L ($)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=100)
        plt.close()
        
        return 'backtest_results.png'
    
    def run(self):
        """Ejecuta backtest completo"""
        # Cargar datos
        df = pd.read_parquet('data/processed/labeled_features.parquet')
        signals = pd.read_csv('data/processed/test_signals.csv')
        
        logger.info("Ejecutando backtest...")
        
        # Ejecutar backtest
        results = self.execute_backtest(df, signals)
        
        # Generar gráficos
        plot_path = self.plot_results(results['equity_curve'], results['trades'])
        
        # Log métricas
        logger.info("\n=== Métricas de Backtest ===")
        for key, value in results['metrics'].items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        
        # Guardar resultados
        import mlflow
        with mlflow.start_run():
            # Log métricas
            for key, value in results['metrics'].items():
                mlflow.log_metric(f"backtest_{key}", value)
            
            # Log gráficos
            mlflow.log_artifact(plot_path)
            
            # Guardar trades
            if len(results['trades']) > 0:
                trades_df = pd.DataFrame(results['trades'])
                trades_df.to_csv('backtest_trades.csv', index=False)
                mlflow.log_artifact('backtest_trades.csv')
        
        return results

def main():
    """Ejecutar backtest"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    backtester = Backtester(config)
    backtester.run()

if __name__ == "__main__":
    main()
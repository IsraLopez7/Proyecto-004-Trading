# src/backtest.py
"""
Backtest realista con costos:
- Comisión: 0.125% por lado (compra/venta)
- Borrow cost: 0.25% anual en cortos (prorrateado por días)
- Stop Loss y Take Profit

Métricas: retorno total, Sharpe, Sortino, Calmar, Max Drawdown, Win-rate, etc.
"""

import numpy as np
import pandas as pd
import yaml
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_signals_and_prices():
    """
    Carga señales generadas y precios correspondientes.
    Asume que las señales están en results/signals_val.csv
    y que corresponden al set de validación.
    """
    # Cargar señales
    signals_df = pd.read_csv('results/signals_val.csv')
    
    # Cargar precios del set de validación
    # Necesitamos obtener las fechas correspondientes
    df_full = pd.read_parquet('data/processed/features.parquet')
    
    # Cargar splits info para obtener índice de validación
    splits_info = pd.read_csv('data/processed/splits_info.csv', index_col=0)
    train_size = int(splits_info.loc['train', 'size'])
    test_size = int(splits_info.loc['test', 'size'])
    
    # Ajustar por ventana W
    W = load_config()['windows']['W']
    
    # El set de validación comienza en train_size + test_size
    val_start_idx = train_size + test_size + W - 1
    val_prices = df_full.iloc[val_start_idx:val_start_idx + len(signals_df)]
    
    logger.info(f"Precios de validación: {len(val_prices)} filas")
    logger.info(f"Señales: {len(signals_df)} filas")
    
    return signals_df, val_prices


def calculate_trade_returns(signals, prices, config):
    """
    Calcula retornos por trade considerando costos.
    
    Args:
        signals: DataFrame con columna 'signal'
        prices: DataFrame con columna 'Close'
        config: configuración con parámetros de backtest
    
    Returns:
        DataFrame con trades y métricas
    """
    commission = config['backtest']['commission']
    borrow_rate_annual = config['backtest']['borrow_rate']
    sl_pct = config['backtest']['stop_loss_pct']
    tp_pct = config['backtest']['take_profit_pct']
    
    trades = []
    equity = [1.0]  # Capital inicial = 1.0 (normalizado)
    
    current_position = None  # {type: 'long'/'short', entry_price, entry_idx}
    
    signals_array = signals['signal'].values
    prices_array = prices['Close'].values
    
    for i in range(len(signals_array)):
        signal = signals_array[i]
        price = prices_array[i]
        
        # Si no hay posición, abrir según señal
        if current_position is None:
            if signal == 0:  # Long
                current_position = {'type': 'long', 'entry_price': price, 'entry_idx': i}
            elif signal == 2:  # Short
                current_position = {'type': 'short', 'entry_price': price, 'entry_idx': i}
            # Si es hold (1), no hacer nada
        
        else:
            # Verificar SL/TP o cambio de señal
            entry_price = current_position['entry_price']
            position_type = current_position['type']
            
            # Calcular retorno actual
            if position_type == 'long':
                raw_return = (price - entry_price) / entry_price
            else:  # short
                raw_return = (entry_price - price) / entry_price
            
            # Verificar SL/TP
            close_position = False
            exit_reason = None
            
            if raw_return <= -sl_pct:
                close_position = True
                exit_reason = 'stop_loss'
            elif raw_return >= tp_pct:
                close_position = True
                exit_reason = 'take_profit'
            elif (position_type == 'long' and signal != 0) or \
                 (position_type == 'short' and signal != 2):
                close_position = True
                exit_reason = 'signal_change'
            
            if close_position:
                # Calcular retorno neto después de costos
                days_held = i - current_position['entry_idx']
                
                # Comisión: 2 lados
                cost_commission = 2 * commission
                
                # Borrow cost (solo para shorts)
                if position_type == 'short':
                    cost_borrow = borrow_rate_annual * (days_held / 252)
                else:
                    cost_borrow = 0
                
                net_return = raw_return - cost_commission - cost_borrow
                
                # Actualizar equity
                equity.append(equity[-1] * (1 + net_return))
                
                # Registrar trade
                trades.append({
                    'entry_idx': current_position['entry_idx'],
                    'exit_idx': i,
                    'type': position_type,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'raw_return': raw_return,
                    'cost_commission': cost_commission,
                    'cost_borrow': cost_borrow,
                    'net_return': net_return,
                    'days_held': days_held,
                    'exit_reason': exit_reason
                })
                
                # Cerrar posición
                current_position = None
        
        # Equity tracking (si no hay posición, mantener equity constante)
        if current_position is None and len(equity) <= i:
            equity.append(equity[-1])
    
    # Cerrar posición abierta al final si existe
    if current_position is not None:
        i = len(signals_array) - 1
        price = prices_array[i]
        entry_price = current_position['entry_price']
        position_type = current_position['type']
        
        if position_type == 'long':
            raw_return = (price - entry_price) / entry_price
        else:
            raw_return = (entry_price - price) / entry_price
        
        days_held = i - current_position['entry_idx']
        cost_commission = 2 * commission
        cost_borrow = borrow_rate_annual * (days_held / 252) if position_type == 'short' else 0
        net_return = raw_return - cost_commission - cost_borrow
        
        equity.append(equity[-1] * (1 + net_return))
        
        trades.append({
            'entry_idx': current_position['entry_idx'],
            'exit_idx': i,
            'type': position_type,
            'entry_price': entry_price,
            'exit_price': price,
            'raw_return': raw_return,
            'cost_commission': cost_commission,
            'cost_borrow': cost_borrow,
            'net_return': net_return,
            'days_held': days_held,
            'exit_reason': 'end_of_period'
        })
    
    trades_df = pd.DataFrame(trades)
    equity_curve = np.array(equity[:len(signals_array)])
    
    return trades_df, equity_curve


def calculate_metrics(trades_df, equity_curve):
    """Calcula métricas de performance."""
    if len(trades_df) == 0:
        logger.warning("No se ejecutaron trades")
        return {}
    
    # Retorno total
    total_return = equity_curve[-1] - 1.0
    
    # Retornos diarios
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # Sharpe ratio (anualizado, asumiendo 252 días de trading)
    if np.std(daily_returns) > 0:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Sortino ratio (solo desviación de retornos negativos)
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) > 0 and np.std(negative_returns) > 0:
        sortino = (np.mean(daily_returns) / np.std(negative_returns)) * np.sqrt(252)
    else:
        sortino = 0
    
    # Max Drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - cummax) / cummax
    max_drawdown = np.min(drawdowns)
    
    # Calmar ratio
    if max_drawdown != 0:
        calmar = (total_return / abs(max_drawdown))
    else:
        calmar = 0
    
    # Win rate
    wins = (trades_df['net_return'] > 0).sum()
    win_rate = wins / len(trades_df) if len(trades_df) > 0 else 0
    
    # Promedio de retornos por trade
    avg_return_per_trade = trades_df['net_return'].mean()
    
    metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_trades': len(trades_df),
        'avg_return_per_trade': avg_return_per_trade,
        'total_commission_cost': trades_df['cost_commission'].sum(),
        'total_borrow_cost': trades_df['cost_borrow'].sum()
    }
    
    return metrics


def plot_results(equity_curve, trades_df, save_dir='results'):
    """Genera gráficos de resultados."""
    Path(save_dir).mkdir(exist_ok=True)
    
    # Equity curve
    plt.figure(figsize=(14, 6))
    plt.plot(equity_curve, linewidth=2)
    plt.xlabel('Días')
    plt.ylabel('Equity (normalizado)')
    plt.title('Equity Curve - Backtest')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Distribución de retornos por trade
    if len(trades_df) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(trades_df['net_return'] * 100, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Retorno neto por trade (%)')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Retornos por Trade')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/returns_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Gráficos guardados en {save_dir}/")


def main():
    config = load_config()
    
    logger.info("Iniciando backtest...")
    
    # Cargar señales y precios
    signals_df, prices_df = load_signals_and_prices()
    
    # Ejecutar backtest
    trades_df, equity_curve = calculate_trade_returns(signals_df, prices_df, config)
    
    # Calcular métricas
    metrics = calculate_metrics(trades_df, equity_curve)
    
    # Mostrar resultados
    logger.info("\n" + "="*60)
    logger.info("RESULTADOS DEL BACKTEST")
    logger.info("="*60)
    logger.info(f"Retorno Total:      {metrics['total_return']*100:.2f}%")
    logger.info(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.3f}")
    logger.info(f"Sortino Ratio:      {metrics['sortino_ratio']:.3f}")
    logger.info(f"Calmar Ratio:       {metrics['calmar_ratio']:.3f}")
    logger.info(f"Max Drawdown:       {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"Win Rate:           {metrics['win_rate']*100:.2f}%")
    logger.info(f"Número de Trades:   {metrics['n_trades']}")
    logger.info(f"Retorno Prom/Trade: {metrics['avg_return_per_trade']*100:.2f}%")
    logger.info(f"Costo Comisiones:   {metrics['total_commission_cost']*100:.2f}%")
    logger.info(f"Costo Borrow:       {metrics['total_borrow_cost']*100:.2f}%")
    logger.info("="*60 + "\n")
    
    # Guardar resultados
    with open('results/backtest_report.txt', 'w') as f:
        f.write("BACKTEST REPORT\n")
        f.write("="*60 + "\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    
    trades_df.to_csv('results/trades.csv', index=False)
    np.save('results/equity_curve.npy', equity_curve)
    
    # Gráficos
    plot_results(equity_curve, trades_df)
    
    # Loggear en MLflow (opcional, conectar a último run)
    try:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact('results/backtest_report.txt')
        mlflow.log_artifact('results/equity_curve.png')
        mlflow.log_artifact('results/returns_distribution.png')
        logger.info("Resultados loggeados en MLflow")
    except Exception as e:
        logger.warning(f"No se pudo loggear en MLflow: {e}")
    
    logger.info("✅ Backtest completado")


if __name__ == "__main__":
    main()
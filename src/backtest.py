"""
Backtesting Module
Realistic backtesting with commissions and borrow costs
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class Backtester:
    """Backtesting engine with realistic costs."""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.00125,
        borrow_rate: float = 0.0025,
        stop_loss: float = 0.02,
        take_profit: float = 0.04
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            commission: Commission rate per side (e.g., 0.00125 = 0.125%)
            borrow_rate: Annual borrow rate for shorts (e.g., 0.0025 = 0.25%)
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.borrow_rate = borrow_rate
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # Tracking
        self.cash = initial_capital
        self.position = None  # 'long', 'short', or None
        self.entry_price = None
        self.entry_date = None
        self.shares = 0

        self.equity_curve = []
        self.trades = []

    def enter_long(self, price: float, date):
        """Enter long position."""
        # Calculate shares (use all cash)
        self.shares = (self.cash / price) * (1 - self.commission)
        self.entry_price = price
        self.entry_date = date
        self.position = 'long'

        # Pay commission
        commission_cost = self.cash * self.commission
        self.cash = 0

        logger.debug(f"[{date}] ENTER LONG @ {price:.2f} | Shares: {self.shares:.2f} | Comm: {commission_cost:.2f}")

    def exit_long(self, price: float, date):
        """Exit long position."""
        if self.position != 'long':
            return

        # Calculate proceeds
        gross_proceeds = self.shares * price
        commission_cost = gross_proceeds * self.commission
        net_proceeds = gross_proceeds - commission_cost

        # Calculate return
        days_held = (date - self.entry_date).days
        pnl = net_proceeds - self.initial_capital

        # Update cash
        self.cash = net_proceeds

        # Record trade
        self.trades.append({
            'entry_date': self.entry_date,
            'exit_date': date,
            'direction': 'long',
            'entry_price': self.entry_price,
            'exit_price': price,
            'shares': self.shares,
            'days_held': days_held,
            'pnl': pnl,
            'return': (price - self.entry_price) / self.entry_price,
            'commission_cost': commission_cost * 2  # Entry + exit
        })

        logger.debug(f"[{date}] EXIT LONG @ {price:.2f} | PnL: {pnl:.2f} | Days: {days_held}")

        # Reset position
        self.position = None
        self.entry_price = None
        self.entry_date = None
        self.shares = 0

    def enter_short(self, price: float, date):
        """Enter short position."""
        # Borrow shares and sell
        self.shares = (self.cash / price) * (1 - self.commission)
        self.entry_price = price
        self.entry_date = date
        self.position = 'short'

        # Receive cash from short sale (minus commission)
        short_proceeds = self.shares * price
        commission_cost = short_proceeds * self.commission
        self.cash = short_proceeds - commission_cost

        logger.debug(f"[{date}] ENTER SHORT @ {price:.2f} | Shares: {self.shares:.2f} | Comm: {commission_cost:.2f}")

    def exit_short(self, price: float, date):
        """Exit short position."""
        if self.position != 'short':
            return

        # Buy back shares
        buyback_cost = self.shares * price
        commission_cost = buyback_cost * self.commission

        # Calculate borrow cost
        days_held = (date - self.entry_date).days
        borrow_cost = (self.shares * self.entry_price) * self.borrow_rate * (days_held / 365)

        # Total cost
        total_cost = buyback_cost + commission_cost + borrow_cost

        # Calculate PnL
        initial_proceeds = self.shares * self.entry_price
        pnl = initial_proceeds - total_cost

        # Update cash
        self.cash = self.cash - total_cost + initial_proceeds

        # Record trade
        self.trades.append({
            'entry_date': self.entry_date,
            'exit_date': date,
            'direction': 'short',
            'entry_price': self.entry_price,
            'exit_price': price,
            'shares': self.shares,
            'days_held': days_held,
            'pnl': pnl,
            'return': (self.entry_price - price) / self.entry_price,
            'commission_cost': commission_cost * 2,  # Entry + exit
            'borrow_cost': borrow_cost
        })

        logger.debug(f"[{date}] EXIT SHORT @ {price:.2f} | PnL: {pnl:.2f} | Borrow: {borrow_cost:.2f}")

        # Reset position
        self.position = None
        self.entry_price = None
        self.entry_date = None
        self.shares = 0

    def check_stop_loss_take_profit(self, current_price: float, date):
        """Check and execute SL/TP."""
        if self.position is None:
            return

        if self.position == 'long':
            pct_change = (current_price - self.entry_price) / self.entry_price

            if pct_change <= -self.stop_loss:
                logger.debug(f"[{date}] STOP LOSS triggered (long)")
                self.exit_long(current_price, date)
            elif pct_change >= self.take_profit:
                logger.debug(f"[{date}] TAKE PROFIT triggered (long)")
                self.exit_long(current_price, date)

        elif self.position == 'short':
            pct_change = (self.entry_price - current_price) / self.entry_price

            if pct_change <= -self.stop_loss:
                logger.debug(f"[{date}] STOP LOSS triggered (short)")
                self.exit_short(current_price, date)
            elif pct_change >= self.take_profit:
                logger.debug(f"[{date}] TAKE PROFIT triggered (short)")
                self.exit_short(current_price, date)

    def get_equity(self, current_price: float):
        """Calculate current equity."""
        if self.position is None:
            return self.cash
        elif self.position == 'long':
            return self.shares * current_price
        elif self.position == 'short':
            # Cash - (shares * current_price) + (shares * entry_price)
            unrealized_pnl = self.shares * (self.entry_price - current_price)
            return self.cash + unrealized_pnl

    def run(self, signals: pd.DataFrame, prices: pd.DataFrame):
        """
        Run backtest.

        Args:
            signals: DataFrame with 'signal' column {long, short, hold}
            prices: DataFrame with 'Close' column
        """
        logger.info("Running backtest...")

        # Align indices
        common_idx = signals.index.intersection(prices.index)
        signals = signals.loc[common_idx]
        prices = prices.loc[common_idx]

        for date, row in signals.iterrows():
            signal = row['signal']
            price = prices.loc[date, 'Close']

            # Check SL/TP first
            self.check_stop_loss_take_profit(price, date)

            # Execute signals
            if signal == 'long' and self.position is None:
                self.enter_long(price, date)
            elif signal == 'short' and self.position is None:
                self.enter_short(price, date)
            elif signal == 'hold' and self.position is not None:
                # Exit current position
                if self.position == 'long':
                    self.exit_long(price, date)
                elif self.position == 'short':
                    self.exit_short(price, date)

            # Track equity
            equity = self.get_equity(price)
            self.equity_curve.append({
                'date': date,
                'equity': equity,
                'position': self.position
            })

        # Close any open position at end
        if self.position is not None:
            final_date = signals.index[-1]
            final_price = prices.loc[final_date, 'Close']
            if self.position == 'long':
                self.exit_long(final_price, final_date)
            elif self.position == 'short':
                self.exit_short(final_price, final_date)

        logger.info(f"Backtest complete: {len(self.trades)} trades executed")

    def calculate_metrics(self) -> dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {}

        equity_df = pd.DataFrame(self.equity_curve).set_index('date')
        trades_df = pd.DataFrame(self.trades)

        # Returns
        equity_df['returns'] = equity_df['equity'].pct_change()

        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Sharpe Ratio (annualized)
        mean_return = equity_df['returns'].mean()
        std_return = equity_df['returns'].std()
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        # Sortino Ratio
        downside_returns = equity_df['returns'][equity_df['returns'] < 0]
        downside_std = downside_returns.std()
        sortino = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        # Max Drawdown
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade stats
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]

        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0

        metrics = {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_trade_pnl': trades_df['pnl'].mean(),
            'avg_winning_trade': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_losing_trade': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
        }

        return metrics


def plot_equity_curve(backtester: Backtester, save_path: str):
    """Plot and save equity curve."""
    equity_df = pd.DataFrame(backtester.equity_curve).set_index('date')

    plt.figure(figsize=(14, 6))
    plt.plot(equity_df.index, equity_df['equity'], linewidth=2)
    plt.axhline(backtester.initial_capital, color='gray', linestyle='--', label='Initial Capital')
    plt.title('Equity Curve', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Equity curve saved to {save_path}")


def plot_returns_distribution(backtester: Backtester, save_path: str):
    """Plot returns distribution."""
    if not backtester.trades:
        return

    trades_df = pd.DataFrame(backtester.trades)

    plt.figure(figsize=(10, 6))
    plt.hist(trades_df['return'], bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.title('Distribution of Trade Returns', fontsize=16)
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Returns distribution saved to {save_path}")


def main():
    """Main execution function."""
    logger.info("Starting backtesting pipeline...")

    # Load config
    config = load_config()

    # Load signals
    data_dir = Path("data/processed")
    signals = pd.read_parquet(data_dir / "signals_test.parquet")

    # Load prices for test period
    prices = pd.read_parquet(data_dir / "data_test.parquet")

    # Align on window size (signals are shorter due to windowing)
    # We need to get the dates corresponding to signals
    # The window generation drops the first W-1 samples
    W = config['model']['W']

    # Get the dates after windowing
    test_dates = prices.index[W-1:]
    signals.index = test_dates[:len(signals)]

    # Initialize backtester
    backtester = Backtester(
        initial_capital=config['backtest']['capital'],
        commission=config['backtest']['commission'],
        borrow_rate=config['backtest']['borrow_rate'],
        stop_loss=config['backtest']['stop_loss'],
        take_profit=config['backtest']['take_profit']
    )

    # Run backtest
    backtester.run(signals, prices)

    # Calculate metrics
    metrics = backtester.calculate_metrics()

    # Print results
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS:")
    logger.info("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'return' in key or 'rate' in key or 'ratio' in key or 'drawdown' in key:
                logger.info(f"{key:25s}: {value:10.2%}")
            else:
                logger.info(f"{key:25s}: {value:10.2f}")
        else:
            logger.info(f"{key:25s}: {value}")
    logger.info("=" * 60)

    # Save results
    results_path = data_dir / "backtest_results.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Plot equity curve
    equity_path = data_dir / "equity_curve.png"
    plot_equity_curve(backtester, equity_path)

    # Plot returns distribution
    returns_path = data_dir / "returns_distribution.png"
    plot_returns_distribution(backtester, returns_path)

    logger.info("Backtesting complete!")


if __name__ == "__main__":
    main()

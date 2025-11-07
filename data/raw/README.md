# Raw Data Directory

## Downloading QQQ Historical Data

After installing the dependencies from `requirements.txt`, run:

```bash
python scripts/download_data.py
```

This will download approximately 15 years of daily OHLCV data for QQQ and save it to `data/raw/QQQ.csv`.

## Manual Download

Alternatively, you can manually download data from:
- Yahoo Finance: https://finance.yahoo.com/quote/QQQ/history
- Download as CSV with columns: Date, Open, High, Low, Close, Volume
- Save as `QQQ.csv` in this directory

## Expected Format

The CSV should have the following columns:
- Date
- Open
- High
- Low
- Close
- Volume

Example:
```
Date,Open,High,Low,Close,Volume
2010-01-04,46.50,46.75,46.25,46.70,100000000
...
```

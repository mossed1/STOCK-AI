disclaimer = """
⚠️ DISCLAIMER ⚠️
This script is for educational, informational, and experimental purposes only and should not be considered financial advice.
Always do your own research or consult a professional before making financial decisions.
By typing 'yes', you acknowledge this disclaimer.
"""
user_ack = input(disclaimer + "\nType 'yes' to continue: ").strip().lower()
if user_ack != 'yes':
    print("Exiting script. Disclaimer not accepted.")
    exit()

# Data source selection
print("\nChoose data source:")
print("1. Alpha Vantage")
print("2. yFinance")
while True:
    data_source_choice = input("Enter 1 or 2: ").strip()
    if data_source_choice == '1':
        data_source = 'alphavantage'
        break
    elif data_source_choice == '2':
        data_source = 'yfinance'
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")

import time
import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volume import VolumeWeightedAveragePrice
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from alpha_vantage.timeseries import TimeSeries
from xgboost import XGBRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from rich import print
from rich.console import Console

console = Console()

class StockDataset(Dataset):
    def __init__(self, data: np.ndarray, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + self.block_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class StockAI(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layers=4, n_heads=4, d_model=128, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = idx.unsqueeze(-1)
        x = self.input_proj(x)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        pos = self.pos_emb(pos)
        x = x + pos
        x = self.transformer(x)
        x = self.ln_f(x)
        logits = self.head(x).squeeze(-1)
        return logits


# --- Transformer Training Function ---
def train_transformer_model(df, args):
    close_prices = df['Close'].tail(5000).values
    if len(close_prices) <= args.block_size:
        console.print(f"[red]Not enough data points ({len(close_prices)}) for block size {args.block_size}[/red]")
        return None

    dataset = StockDataset(close_prices, args.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockAI(vocab_size=dataset.data.shape[0], block_size=args.block_size, n_layers=args.n_layers, n_heads=args.n_heads, d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.mse_loss(logits[:, -1], yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            run_loss += loss.item()
        avg = run_loss / len(loader)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), "transformer_model.pth")
            console.log(f"[yellow]Epoch {epoch}  Loss {avg:.4f} (saved)[/yellow]")
        elif epoch % args.log_interval == 0:
            console.log(f"Epoch {epoch}  Loss {avg:.4f}")

    return model


# Fetch stock data from Alpha Vantage or yFinance with retry mechanism
def fetch_data(ticker="AAPL", retries=5, delay=10):
    if data_source == 'yfinance':
        import yfinance as yf
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="max")
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.dropna(inplace=True)
            return df
        except Exception as e:
            console.print(f"❌ yFinance error for {ticker}: {e}")
            return pd.DataFrame()
    else:
        key = "YOUR_API_KEY"
        ts = TimeSeries(key)
        for i in range(retries):
            try:
                data, meta_data = ts.get_daily_adjusted(symbol=ticker, outputsize="full")
                df = pd.DataFrame(data).T
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df = df.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. adjusted close': 'Close',
                    '6. volume': 'Volume'
                })
                df = df.astype({
                    'Open': float,
                    'High': float,
                    'Low': float,
                    'Close': float,
                    'Volume': float
                })
                df.dropna(inplace=True)
                return df
            except Exception as e:
                console.print(f"❌ Failed to fetch data for {ticker}: {e}")
                if "YOUR_API_KEY" in str(e) or "rate limit" in str(e).lower():
                    break
                if i < retries - 1:
                    console.print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    console.print("Max retries reached. Exiting.")
                    return pd.DataFrame()

# --- News Scraping Function ---
def fetch_recent_news(ticker):
    news_url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch"
    response = requests.get(news_url)
    soup = BeautifulSoup(response.text, "html.parser")
    headlines = soup.find_all("h3")
    return [h.get_text(strip=True) for h in headlines if h.get_text(strip=True)]


def add_indicators(df):
    from ta.trend import MACD, CCIIndicator, ADXIndicator, EMAIndicator
    from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
    from ta.volatility import AverageTrueRange, DonchianChannel, KeltnerChannel
    from ta.momentum import StochasticOscillator, WilliamsRIndicator
    from ta.trend import PSARIndicator
    from ta.trend import IchimokuIndicator, VortexIndicator
    from ta.volume import EaseOfMovementIndicator
    from ta.momentum import KAMAIndicator, AwesomeOscillatorIndicator

    df['SMA'] = SMAIndicator(df['Close'], window=14).sma_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    
    bb = BollingerBands(df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    df['BullPower'] = df['High'] - df['SMA']
    df['BearPower'] = df['Low'] - df['SMA']
    
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()

    df['MACD'] = MACD(df['Close']).macd()
    df['CCI'] = CCIIndicator(df['High'], df['Low'], df['Close']).cci()
    df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    df['EMA'] = EMAIndicator(df['Close'], window=14).ema_indicator()

    df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['CMF'] = ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume']).chaikin_money_flow()

    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['Donchian_High'] = DonchianChannel(df['High'], df['Low'], df['Close']).donchian_channel_hband()
    df['Donchian_Low'] = DonchianChannel(df['High'], df['Low'], df['Close']).donchian_channel_lband()
    df['Keltner_High'] = KeltnerChannel(df['High'], df['Low'], df['Close']).keltner_channel_hband()
    df['Keltner_Low'] = KeltnerChannel(df['High'], df['Low'], df['Close']).keltner_channel_lband()

    df['Stochastic'] = StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    df['WilliamsR'] = WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()

    psar = PSARIndicator(df['High'], df['Low'], df['Close'])
    df['PSAR'] = psar.psar()

    # New indicators
    df['Ichimoku_a'] = IchimokuIndicator(df['High'], df['Low']).ichimoku_a()
    df['Ichimoku_b'] = IchimokuIndicator(df['High'], df['Low']).ichimoku_b()
    df['Vortex_Pos'] = VortexIndicator(df['High'], df['Low'], df['Close']).vortex_indicator_pos()
    df['Vortex_Neg'] = VortexIndicator(df['High'], df['Low'], df['Close']).vortex_indicator_neg()
    df['EoM'] = EaseOfMovementIndicator(df['High'], df['Low'], df['Volume']).ease_of_movement()
    df['KAMA'] = KAMAIndicator(df['Close'], window=10).kama()
    df['Awesome_Osc'] = AwesomeOscillatorIndicator(df['High'], df['Low']).awesome_oscillator()
    
    return df.dropna()


# --- Quant Model Functions ---
def apply_pca(features):
    pca = PCA(n_components=10)
    return pca.fit_transform(features)

def arima_forecast(df, order=(5, 1, 0)):
    # ARIMA expects a 1D array; use 'Close' price
    try:
        model = ARIMA(df['Close'], order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast[0]
    except Exception as e:
        console.print(f"ARIMA error: {e}")
        return df['Close'].iloc[-1]

def sharpe_ratio(df):
    # Calculate log returns if not present
    if 'log_returns' not in df.columns:
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    returns = df['log_returns'].dropna()
    if returns.std() == 0:
        return 0.0
    return np.mean(returns) / np.std(returns)

def max_drawdown(df):
    # Calculate log returns if not present
    if 'log_returns' not in df.columns:
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    cumulative_returns = (1 + df['log_returns'].fillna(0)).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def predict_price_with_quant_model(df):
    df = add_indicators(df)
    # Compute log returns and volatility
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_returns'].rolling(window=14).std()
    df['roc'] = df['Close'].pct_change(periods=5)
    df = df.dropna()

    # Feature selection for PCA and models
    feature_cols = [
        'SMA', 'RSI', 'BB_High', 'BB_Low', 'BullPower', 'BearPower', 'Volume_MA',
        'MACD', 'CCI', 'ADX', 'EMA', 'OBV', 'CMF', 'ATR', 'log_returns', 'volatility',
        'roc'
    ]
    features = df[feature_cols].values
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # Apply PCA for dimensionality reduction
    features_pca = apply_pca(features_scaled)

    # ARIMA forecast for price direction
    arima_pred = arima_forecast(df)

    # Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(features_scaled[:-1], df['Close'].values[:-1])
    rf_pred = rf_model.predict(features_scaled[-1].reshape(1, -1))[0]

    # XGBoost Regressor
    xgb_model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_model.fit(features_scaled[:-1], df['Close'].values[:-1])
    xgb_pred = xgb_model.predict(features_scaled[-1].reshape(1, -1))[0]

    pca_mean = np.mean(features_pca[-1])
    
    # Combine RF, XGB, ARIMA, and PCA mean
    final_pred = (
        0.4 * rf_pred +
        0.4 * xgb_pred +
        0.15 * arima_pred +
        0.05 * pca_mean
    )

    # Calculate performance metrics: Sharpe Ratio and Max Drawdown
    sr = sharpe_ratio(df)
    mdd = max_drawdown(df)
    console.print(f"Sharpe Ratio: {sr:.2f}, Max Drawdown: {mdd:.2f}")

    return final_pred

def analyze(df, predicted_price):
    current_price = df['Close'].iloc[-1]
    signals = []

    if df['RSI'].iloc[-1] < 30:
        signals.append("RSI oversold")

    if df['Close'].iloc[-1] < df['BB_Low'].iloc[-1]:
        signals.append("Below Bollinger Band")

    if df['BullPower'].iloc[-1] > 0 and df['BearPower'].iloc[-1] > 0:
        signals.append("Bullish power")

    if df['Close'].iloc[-1] > df['SMA'].iloc[-1]:
        signals.append("Above SMA")

    if df['Volume'].iloc[-1] > df['Volume_MA'].iloc[-1]:
        signals.append("Volume spike")

    if df['MACD'].iloc[-1] > 0:
        signals.append("MACD bullish")

    if df['ADX'].iloc[-1] > 25:
        signals.append("Strong trend (ADX)")

    if df['CCI'].iloc[-1] < -100:
        signals.append("CCI oversold")

    if df['OBV'].iloc[-1] > df['OBV'].iloc[-2]:
        signals.append("OBV rising")

    if df['CMF'].iloc[-1] > 0:
        signals.append("CMF positive")

    if df['Stochastic'].iloc[-1] < 20:
        signals.append("Stochastic oversold")

    if df['WilliamsR'].iloc[-1] < -80:
        signals.append("Williams %R oversold")

    if df['Close'].iloc[-1] > df['PSAR'].iloc[-1]:
        signals.append("Above PSAR")

    # New indicator signals
    if df['Ichimoku_a'].iloc[-1] > df['Ichimoku_b'].iloc[-1]:
        signals.append("Ichimoku bullish")

    if df['Vortex_Pos'].iloc[-1] > df['Vortex_Neg'].iloc[-1]:
        signals.append("Vortex bullish")

    if df['EoM'].iloc[-1] > 0:
        signals.append("EoM positive")

    if df['KAMA'].iloc[-1] < df['Close'].iloc[-1]:
        signals.append("Price above KAMA")

    if df['Awesome_Osc'].iloc[-1] > 0:
        signals.append("Awesome Oscillator positive")

    price_diff = predicted_price - current_price
    direction = "UP 📈" if price_diff > 0 else "DOWN 📉"

    confidence = round(len(signals) / 18 * 100, 2)

    return direction, confidence, signals

def stock_bot(ticker):
    df = fetch_data(ticker)
    if df.empty:
        console.print(f"⚠️ No data available for {ticker}. Skipping analysis.")
        return
    import argparse
    args = argparse.Namespace(
        epochs=25,
        batch_size=32,
        block_size=64,
        n_layers=8,
        n_heads=8,
        d_model=256,
        dim_feedforward=1024,
        dropout=0.05,
        lr=3e-4,
        weight_decay=1e-2,
        grad_clip=1.0,
        log_interval=2
    )
    transformer_model = train_transformer_model(df, args)
    predicted_price = predict_price_with_quant_model(df)
    direction, confidence, signals = analyze(df, predicted_price)
    console.print(f"\n📊 {ticker} Stock Analysis")
    console.print(f"🔮 Predicted Direction: {direction}")
    console.print(f"📈 Confidence: {confidence}%")
    console.print(f"✅ Signals: {', '.join(signals)}")
    console.print(f"📍 Predicted Price: {predicted_price:.2f}, Current Price: {df['Close'].iloc[-1]:.2f}")
    console.print(f"⏱️ Timeframe: Daily")
    console.print(f"⏳ Suggested Holding Period: ~5 trading days (based on model horizon)")
    
if __name__ == "__main__":
    stock_code = input("Enter the stock ticker (e.g., AAPL): ").strip().upper()
    if not stock_code:
        stock_bot("AAPL")
    else:
        stock_bot(stock_code)

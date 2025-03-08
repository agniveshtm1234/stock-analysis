import yfinance as yf
import talib
import numpy as np
from nsepython import *
import pandas as pd
import warnings
import streamlit as st
from datetime import datetime
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

# Global variables
now = datetime.now()
date = now.strftime("%Y-%m-%d %H-%M-%S")
names = ['STOCK', 'RSI', 'PRICE VS MA50', 'MACD', 'VWAP', 'NIFTYTREND', 'BOLLINGERBANDS', 
         'ATR STOP-LOSS', 'BULLISH INDICATOR COUNT', 'RECOMMENDATION', 'PROFIT POTENTIAL']

# Moving Average Check
def check_moving_average_conditions(stock_ticker):
    try:
        data = yf.download(stock_ticker, period="59d", interval="15m")
        if data.empty or 'Close' not in data:
            return None, f"No data found for {stock_ticker}. Check the ticker symbol."
        close_prices = np.array(data['Close'].dropna().astype(float)).flatten()
        if len(close_prices) < 50:
            return None, f"Not enough data to calculate moving averages for {stock_ticker}."
        
        ma_10 = talib.SMA(close_prices, timeperiod=10)
        ma_20 = talib.SMA(close_prices, timeperiod=20)
        ma_50 = talib.SMA(close_prices, timeperiod=50)

        latest_price = close_prices[-1]
        latest_ma_10 = ma_10[-1]
        latest_ma_20 = ma_20[-1]
        latest_ma_50 = ma_50[-1]

        price_above_50 = latest_price > latest_ma_50
        near_10_20 = abs(latest_price - latest_ma_10) < (latest_price * 0.075) and abs(latest_price - latest_ma_20) < (latest_price * 0.075)
        L2 = [latest_price, latest_ma_10, latest_ma_20, latest_ma_50]

        if price_above_50 and near_10_20:
            return L2, f"{stock_ticker}: ✅ Conditions Met (Price: {latest_price:.2f}, MA10: {latest_ma_10:.2f}, MA20: {latest_ma_20:.2f}, MA50: {latest_ma_50:.2f})"
        else:
            return False, f"{stock_ticker}: ❌ Conditions NOT Met (Price: {latest_price:.2f}, MA10: {latest_ma_10:.2f}, MA20: {latest_ma_20:.2f}, MA50: {latest_ma_50:.2f})"
    except Exception as e:
        return None, f"Error in moving average conditions for {stock_ticker}: {e}"

# RSI Calculation
def calculate_rsi_manual(data, period=14):
    if len(data) < period + 1:
        return np.nan
    delta = np.diff(data)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    return rsi if np.isfinite(rsi) else 50

# MACD Crossover
def check_macd_crossover(data):
    close = data['Close'].to_numpy().astype(np.float64)
    if len(close) < 34:
        return "Insufficient data for MACD"
    try:
        macd, signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        return "Bullish Crossover" if macd[-1] > signal[-1] else "Bearish Crossover" if macd[-1] < signal[-1] else "Neutral"
    except Exception as e:
        return f"MACD failed: {e}"

# ATR Stop Loss and Take Profit
def calculate_atr_stop_loss(data, entry_price, atr_multiplier=1.5, atr_period=14):
    high = data['High'].to_numpy().astype(np.float64)
    low = data['Low'].to_numpy().astype(np.float64)
    close = data['Close'].to_numpy().astype(np.float64)
    if len(close) < atr_period:
        return entry_price - 5, entry_price + 5
    try:
        atr = talib.ATR(high, low, close, timeperiod=atr_period)[-1]
        return entry_price - (atr * atr_multiplier), entry_price + (atr * 2)
    except Exception as e:
        return entry_price - 5, entry_price + 5

# VWAP Check
def check_vwap(data, current_date):
    today_data = data[data.index.date == current_date].copy()
    if len(today_data) < 2:
        return "No intraday data available"
    today_data['TP'] = (today_data['High'] + today_data['Low'] + today_data['Close']) / 3
    today_data['VWAP'] = (today_data['TP'] * today_data['Volume']).cumsum() / today_data['Volume'].cumsum()
    return "Above VWAP (Bullish)" if today_data['Close'].iloc[-1] > today_data['VWAP'].iloc[-1] else "Below VWAP (Bearish)"

# Nifty Trend
def check_nifty_trend():
    try:
        nifty = yf.download("^NSEI", period="60d", interval="15m", auto_adjust=True)
        if nifty.empty:
            nifty = yf.download("^NIFTY50", period="60d", interval="15m", auto_adjust=True)
        if nifty.empty:
            return "No sufficient Nifty data"
        close = nifty['Close'].dropna().values.astype(np.float64).flatten()
        if len(close) < 50:
            return "Insufficient data for SMA calculation"
        ma_50 = talib.SMA(close, timeperiod=50)
        if np.isnan(ma_50[-1]):
            return "SMA calculation failed due to missing data"
        return "Nifty Above MA50 (Bullish)" if close[-1] > ma_50[-1] else "Nifty Below MA50 (Bearish)"
    except Exception as e:
        return f"Error fetching Nifty trend: {e}"

# Bollinger Bands
def check_bollinger_bands(data):
    close = data['Close'].to_numpy().astype(np.float64)
    if len(close) < 20:
        return "Insufficient data for Bollinger Bands"
    try:
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        latest_price = data['Close'].iloc[-1]
        if latest_price <= lower[-1]:
            return "Near Lower Band (Potential Buy)"
        elif latest_price >= upper[-1]:
            return "Near Upper Band (Potential Sell)"
        return "Inside Bands"
    except Exception as e:
        return f"Bollinger Bands failed: {e}"

# Stock Analysis
def analyze_stock(stock_ticker):
    try:
        ticker = stock_ticker if stock_ticker.endswith('.NS') else stock_ticker + '.NS'
        data = yf.download(ticker, period="60d", interval="15m", auto_adjust=False)
        if data.empty:
            return None, f"No data found for {ticker}"

        if data.columns.nlevels > 1:
            data.columns = [col[0] for col in data.columns]

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns) or len(data) < 50:
            return None, f"Insufficient or invalid data for {ticker} (Rows: {len(data)})"

        latest_price = data['Close'].iloc[-1]
        close_array = data['Close'].to_numpy().astype(np.float64)

        if not np.issubdtype(close_array.dtype, np.number):
            return None, "Error: Close array contains non-numeric data"
        if np.any(~np.isfinite(close_array)):
            close_array = pd.Series(close_array).fillna(method='ffill').fillna(method='bfill').to_numpy()
        close_array = close_array.astype(np.float64)

        rsi = calculate_rsi_manual(close_array, period=14)
        rsi_bullish = rsi > 52

        try:
            ma_50 = talib.SMA(close_array, timeperiod=50)[-1]
        except Exception as e:
            ma_50 = np.mean(close_array[-50:])
        price_above_ma50 = latest_price > ma_50

        macd_status = check_macd_crossover(data)
        vwap_status = check_vwap(data, data.index[-1].date())
        nifty_status = check_nifty_trend()
        bb_status = check_bollinger_bands(data)
        stop_loss, take_profit = calculate_atr_stop_loss(data, latest_price)

        macd_bullish = macd_status == "Bullish Crossover"
        vwap_bullish = vwap_status == "Above VWAP (Bullish)"
        nifty_bullish = nifty_status == "Nifty Above MA50 (Bullish)"
        bb_bullish = bb_status in ["Near Lower Band (Potential Buy)", "Inside Bands"]
        profit_potential = (take_profit - latest_price) >= 4
        profit = (take_profit - latest_price)

        bullish_count = sum([rsi_bullish, price_above_ma50, macd_bullish, vwap_bullish, nifty_bullish, bb_bullish])
        recommendation = f"Buy at {latest_price:.2f}, Stop-Loss: {stop_loss:.2f}, Take-Profit: {take_profit:.2f}" if bullish_count >= 4 and profit_potential else "Do not buy"

        stock_data = [
            stock_ticker,
            f"{rsi:.2f} ({'Bullish' if rsi_bullish else 'Not Bullish'})",
            'Above' if price_above_ma50 else 'Below',
            macd_status,
            vwap_status,
            nifty_status,
            bb_status,
            f"{stop_loss:.2f}, {take_profit:.2f}",
            f"{bullish_count}",
            recommendation,
            profit
        ]

        output = f"""
Stock: {stock_ticker}
RSI: {rsi:.2f} ({'Bullish' if rsi_bullish else 'Not Bullish'})
Price vs MA50: {'Above' if price_above_ma50 else 'Below'})
MACD: {macd_status}
VWAP: {vwap_status}
Nifty Trend: {nifty_status}
Bollinger Bands: {bb_status}
ATR Stop-Loss: {stop_loss:.2f}, Take-Profit: {take_profit:.2f}
Bullish Indicators Count: {bullish_count}/6
Recommendation: {recommendation}
Profit Potential: {profit}
"""
        return stock_data, output
    except Exception as e:
        return None, f"Error analyzing stock: {e}"

# Modified Email Sending Function
def send_email(subject, body, csv_data, recipient_email, sender_email, sender_password):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        part = MIMEBase('application', 'octet-stream')
        part.set_payload(csv_data)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="analysis.csv"')
        msg.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True, "Email sent successfully!"
    except Exception as e:
        return False, f"Failed to send email: {e}"

# Streamlit App
def main():
    st.title("Stock Analysis Dashboard")

    option = st.sidebar.selectbox("Choose an Option", ["Specific Stock", "Latest Gainers", "Intraday Stocks"])

    if option == "Specific Stock":
        st.header("Analyze a Specific Stock")
        ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE)")
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                stock_data, output = analyze_stock(ticker)
                st.write(output)
                if stock_data and not isinstance(stock_data, str):
                    df = pd.DataFrame([stock_data], columns=names)
                    st.dataframe(df)
                    csv_content = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv_content,
                        file_name="specific_stock_analysis.csv",
                        mime="text/csv"
                    )
                    if st.checkbox("Send Email"):
                        recipient = st.text_input("Recipient Email")
                        sender = st.text_input("Your Email")
                        password = st.text_input("App Password", type="password")
                        if st.button("Send"):
                            subject = f"Stock Analysis Report for {ticker} - {date}"
                            body = f"Attached is the analysis for {ticker} generated on {date}.\n\n{output}"
                            success, msg = send_email(subject, body, csv_content, recipient, sender, password)
                            st.write(msg)

    elif option == "Latest Gainers":
        st.header("Analyze Latest Gainers")
        uploaded_file = st.file_uploader("Upload CSV File", type="csv")
        if uploaded_file and st.button("Analyze"):
            with st.spinner("Analyzing..."):
                df = pd.read_csv(uploaded_file)
                headers = df.columns.tolist()
                if 'SYMBOL' not in headers or 'CHNG' not in headers:
                    st.error("CSV must contain 'SYMBOL' and 'CHNG' columns.")
                else:
                    stock_changes = []
                    for _, row in df.iterrows():
                        try:
                            change_value = float(row['CHNG'])
                            stock_changes.append((row['SYMBOL'], change_value))
                        except ValueError:
                            st.write(f"Skipping {row['SYMBOL']}: Invalid CHNG value '{row['CHNG']}'")
                    
                    filtered_stocks = [stock for stock, change in stock_changes if change >= 0]
                    total = len(filtered_stocks)
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, stock in enumerate(filtered_stocks, 1):
                        stock_ticker = stock + '.NS'
                        result, msg = check_moving_average_conditions(stock_ticker)
                        st.write(msg)
                        if result != False:
                            stock_data, output = analyze_stock(stock_ticker)
                            st.write(output)
                            if stock_data and not isinstance(stock_data, str):
                                results.append(stock_data)
                        progress_bar.progress(i / total)
                    
                    if results:
                        df_results = pd.DataFrame(results, columns=names)
                        st.dataframe(df_results)
                        csv_content = df_results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download CSV",
                            data=csv_content,
                            file_name="gainerstock.csv",
                            mime="text/csv"
                        )
                        if st.checkbox("Send Email"):
                            recipient = st.text_input("Recipient Email")
                            sender = st.text_input("Your Email")
                            password = st.text_input("App Password", type="password")
                            if st.button("Send"):
                                subject = f"Latest Gainers Analysis Report - {date}"
                                body = f"Attached is the gainers analysis report generated on {date}."
                                success, msg = send_email(subject, body, csv_content, recipient, sender, password)
                                st.write(msg)

    elif option == "Intraday Stocks":
        st.header("Analyze Intraday Stocks")
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                intraday_stocks = [
                    'ZEEL.NS', 'TATASTEEL.NS', 'ENGINERSIN.NS', 'MMTC.NS', 'REDTAPE.NS', 'SBFC.NS',
                    'USHAMART.NS', 'CAMPUS.NS', 'RHIM.NS', 'GMDCLTD.NS', 'BALRAMCHIN.NS', 'EMBDL.NS',
                    'GENUSPOWER.NS', 'INDIACEM.NS', 'ARVIND.NS', 'TIMETECHNO.NS', 'LATENTVIEW.NS',
                    'GRAPHITE.NS', 'ALOKINDS.NS', 'ASKAUTOLTD.NS', 'SYRMA.NS', 'LLOYDSENGG.NS',
                    'STARCEMENT.NS', 'BBOX.NS', 'JKTYRE.NS', 'TRIVENI.NS', 'PARADEEP.NS', 'GSFC.NS',
                    'EPL.NS', 'RELIGARE.NS', 'RCF.NS', 'GMRP&UI.NS', 'JINDWORLD.NS', 'GALLANTT.NS',
                    'MAXESTATES.NS', 'SCI.NS', 'NETWORK18.NS', 'DBREALTY.NS', 'HONASA.NS', 'KNRCON.NS',
                    'EQUITASBNK.NS', 'GABRIEL.NS', 'IIFLCAPS.NS', 'CMSINFO.NS', 'PNCINFRA.NS', 'ORIENTCEM.NS',
                    'KESORAMIND.NS', 'RENUKA.NS'
                ]
                total = len(intraday_stocks)
                progress_bar = st.progress(0)
                results = []
                
                for i, stock in enumerate(intraday_stocks, 1):
                    result, msg = check_moving_average_conditions(stock)
                    st.write(msg)
                    if result != False:
                        stock_data, output = analyze_stock(stock)
                        st.write(output)
                        if stock_data and not isinstance(stock_data, str):
                            results.append(stock_data)
                    progress_bar.progress(i / total)
                
                if results:
                    df_results = pd.DataFrame(results, columns=names)
                    st.dataframe(df_results)
                    csv_content = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv_content,
                        file_name="intraday_stock_analysis.csv",
                        mime="text/csv"
                    )
                    if st.checkbox("Send Email"):
                        recipient = st.text_input("Recipient Email")
                        sender = st.text_input("Your Email")
                        password = st.text_input("App Password", type="password")
                        if st.button("Send"):
                            subject = f"Intraday Stocks Analysis Report - {date}"
                            body = f"Attached is the intraday analysis report generated on {date}."
                            success, msg = send_email(subject, body, csv_content, recipient, sender, password)
                            st.write(msg)

if __name__ == "__main__":
    main()

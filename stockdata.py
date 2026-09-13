from config import TOKEN
from t_tech.invest import Client, CandleInterval, Quotation
import pandas as pd
from datetime import datetime, timedelta
import os

os.environ['SSL_TBANK_VERIFY'] = 'True'

def quotation_to_float(num):
    return float(num.units) + float(num.nano) / 1e9

def get_figi(ticker: str):
    with Client(TOKEN) as client:
        instruments = client.instruments.find_instrument(query=ticker)
        for instrument in instruments.instruments:
            if instrument.ticker.lower() == ticker.lower() and instrument.figi[0] == "B":
                return str(instrument.figi)
            
figi = get_figi("SBER")

def df(figi: str):
    data = []
    with Client(TOKEN) as client:
        candles = client.get_all_candles(
            figi=figi,
            from_=datetime.now() - timedelta(days=10000),
            to=datetime.now(),
            interval=CandleInterval.CANDLE_INTERVAL_DAY
        )
        for candle in candles:
            data.append({
                'date': candle.time,
                'open': quotation_to_float(candle.open),
                'high': quotation_to_float(candle.high),
                'low': quotation_to_float(candle.low),
                'close': quotation_to_float(candle.close),
                'volume': candle.volume
            })
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df['close']

def get_dates(figi: str):
    data = []
    try:
        with Client(TOKEN) as client:
            candles = client.get_all_candles(
                figi=figi,
                from_=datetime.now() - timedelta(days=10000),
                to=datetime.now(),
                interval=CandleInterval.CANDLE_INTERVAL_DAY
            )
            for candle in candles:
                data.append({
                    'date': candle.time
                })
            data = pd.DataFrame(data=data, columns=['date'])
            return data
    except Exception as e:
        pass

df = df(figi=figi)
dates = get_dates(figi=figi)

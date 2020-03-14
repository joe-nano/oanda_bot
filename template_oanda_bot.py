import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades
import pandas as pd
import matplotlib.pyplot as plt
import time
import talib
import numpy as np
import statsmodels.api as sm
from stocktrends import Renko

client = oandapyV20.API(access_token="api token",environment="practice or live")
account_id = "your-account-id"

#defining strategy paramters
pairs = ['EUR_USD','GBP_USD','AUD_USD', 'CAD_JPY', 'EUR_CHF', 'USD_CHF', 'USD_JPY', 'USD_CAD']
pos_size = 600

#calculate_renko_bricks and number of renko bars from DF
def renko_DF(DF):
    df = DF.copy()
    df2 = Renko(df)
    "Using ATR as renko brick size"
    real = talib.ATR(df["high"], df["low"], df["close"], timeperiod = 14)
    brick_sizing = round(real[len(real) - 1], 6)
    df2.brick_size = brick_sizing
    renko_df = df2.get_bricks()
    renko_df["bar_num"] = np\
        .where(renko_df["uptrend"]==True,1,np\
            .where(renko_df["uptrend"]==False,-1,0))
    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
    return renko_df

#simple moving average across 10 data entries
def SMA10(df):
    df['sma_10']=df['close'].rolling(10).mean()
    return df

#on balance volume
def OBV(DF):
    df = DF.copy()
    df['daily_ret'] = df['close'].pct_change()
    df['direction'] = np.where(df['daily_ret']>=0,1,-1)
    df['direction'][0] = 0
    df['vol_adj'] = df['volume'] * df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df['obv']

def slope(ser,n):
    slopes = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

#returns a pd_df in the form [date, open, high, low, close, volumn]
def candles(instrument):
    params = {"count": 800,"granularity": "M5"} #granularity can be in seconds S5 - S30, minutes M1 - M30, hours H1 - H12, days D, weeks W or months M
    candles = instruments.InstrumentsCandles(instrument=instrument,params=params)
    client.request(candles)
    ohlc_dict = candles.response["candles"]
    ohlc = pd.DataFrame(ohlc_dict)
    ohlc_df = ohlc.mid.dropna().apply(pd.Series)
    ohlc_df["volume"] = ohlc["volume"]
    ohlc_df.index = ohlc["time"]
    ohlc_df = ohlc_df.apply(pd.to_numeric)
    ohlc_df.reset_index(inplace=True)
    ohlc_df = ohlc_df.iloc[:,[0,1,2,3,4,5]]
    ohlc_df.columns = ["date","open","high","low","close","volume"]
    ohlc_df['date'] = ohlc_df['date'].apply(lambda x: str(x).split('T'))
    ohlc_df['date'] = ohlc_df['date'].apply(lambda x: x[0] + " " + x[1].split('.')[0])
    return ohlc_df

def market_order(instrument,units,account_id):  
    params = {"instruments": instrument}
    r = pricing.PricingInfo(accountID=account_id, params=params)
    rv = client.request(r)
    #sl = round(sl, 4) # optimize here
    data = {
            "order": {
                "price": "",
                # 'trailingStopLossOnFill': {
                #     'distance': str(sl), #in pips
                # },
                "timeInForce": "FOK",
                "instrument": str(instrument),
                "units": str(units),
                "type": "MARKET",
                "positionFill": "DEFAULT"
                }
            }
    r = orders.OrderCreate(accountID=account_id, data=data)
    client.request(r)

def signalIntoPosition(df):
    signal = ""
    bar_num = df['bar_num'].iloc[-1]
    obv_slope = df['obv_slope'].iloc[-1]
    sma_10 = df['sma_10'].iloc[-1]
    sma_slope = df['sma_slope'].iloc[-1]
    close = df['close'].iloc[-1]

    print("BarNum: ", bar_num)
    print("OBVSlope: ", obv_slope)
    print("SMA10: ", sma_10)
    print("Close: ", close)
    print("SMASlope: ", sma_slope)

    if bar_num >= 2 and obv_slope >= 30 and close > sma_10 and sma_slope > 0:
         signal = "Buy"
    if bar_num <= -2 and obv_slope <= -30 and close < sma_10 and sma_slope < 0:
         signal = "Sell"

    print("Signal ",signal)
    return signal

def signalOutOfPosition(df, isLong):
    signal = ""
    bar_num = df['bar_num'].iloc[-1]
    sma_10 = df['sma_10'].iloc[-1]
    close = df['close'].iloc[-1]

    print("BarNum: ", bar_num)
    print("SMA10: ", sma_10)
    print("Close: ", close)
    print("isLong: ", isLong)

    if isLong and sma_10 > close:
        signal = "Sell"
    if not isLong and sma_10 < close:
        signal = "Buy"

    print("Signal ",signal)
    return signal

def main():
    global pairs
    try:
        r = trades.OpenTrades(accountID=account_id)
        open_trades = client.request(r)['trades']
        
        curr_ls = []
        for i in range(len(open_trades)):
            curr_ls.append(open_trades[i]['instrument'])
        pairsNotInPosition = [i for i in pairs if i not in curr_ls]

        for currency in pairsNotInPosition:
            print("analyzing currency not in position: ",currency)
            data = candles(currency)
            renkobars = renko_DF(data)
            renkobars = data.merge(renkobars.loc[:,["date","bar_num"]],how="outer",on="date")
            renkobars["bar_num"].fillna(method='ffill',inplace=True)
            renkobars["obv"]= OBV(renkobars)
            renkobars["obv_slope"]= slope(renkobars["obv"],5)
            renkobars = SMA10(renkobars)
            renkobars["sma_slope"] = slope(renkobars["sma_10"], 5)

            #LONG: 
            # 1.When green renko Above 10sma
            # 2.Confirm bias with increase in OBV in correlation with price 
            # 3.(Optional) Stop loss 2 bars below entry point, min take profit 3 renko bars
            # 4.Exit position when Renko goes belows SMA
            signal = signalIntoPosition(renkobars)
            if signal == "Buy":
                market_order(currency,pos_size, account_id)
                print("New long position initiated for ", currency)
            elif signal == "Sell":
                market_order(currency,-1*pos_size, account_id)
                print("New short position initi#ated for ", currency)

        for positions in open_trades:
            currency = positions['instrument']
            print("analyzing currency already in position: ", currency)
            units = int(positions['currentUnits'])
            data = candles(currency)
            renkobars = renko_DF(data)
            renkobars = data.merge(renkobars.loc[:,["date","bar_num"]],how="outer",on="date")
            renkobars["bar_num"].fillna(method='ffill',inplace=True)
            renkobars = SMA10(renkobars)
            renkobars["sma_slope"] = slope(renkobars["sma_10"], 5)
            #print(renkobars)

            signal = signalOutOfPosition(renkobars, isLong = units > 0)
            if signal == "Buy":
                market_order(currency,pos_size, account_id)
                print("New long position initiated for ", currency)
            elif signal == "Sell":
                market_order(currency,-1*pos_size, account_id)
                print("New short position initi#ated for ", currency)
    except ValueError as error:
        print(format(error))
    except:
        print("error")


#script duration
starttime=time.time()
timeout = time.time() + 60*600
while time.time() <= timeout:
    try:
        print("passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep(240 - ((time.time() - starttime) % 240.0))
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()
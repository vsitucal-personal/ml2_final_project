import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def concatenate_strings(row, params_list):
    filtered_strings = [str(row[col]) for col in params_list if pd.notnull(row[col])]
    return '_'.join(filtered_strings)


def get_stock_price(symbol, date, price_dict, stock):
    if price_dict.get(symbol) == None:
        price_dict[symbol] = {}

    if price_dict[symbol].get(date) == None:
        history = stock.history(start=date)
        if not history.empty:
            price = history['Close'].values[0]
            price_dict[symbol][date] = round(price, 2)
            return round(price, 2)
        else:
            price = stock.history(period="1d")['Close'].values[0]
            price_dict[symbol][date] = round(price, 2)
            return round(price, 2)
    else:
        return round(price_dict[symbol][date], 2)


def combine_dfs(ticker_list, price_dict):
    ticker_df_list = []
    for ticker in ticker_list:
        print(ticker)
        df = ticker_option(ticker, price_dict)
        if len(df) > 0:
            ticker_df_list.append(df)
        else:
            print(f"No options found for {ticker}")

    df = pd.concat(ticker_df_list).sort_values(by='lastTradeDate').reset_index(drop=True)
    return df


def ticker_option(ticker, price_dict):
    stock = yf.Ticker(ticker)
    option_df_list = []
    if len(stock.options) == 0:
        return pd.DataFrame()
    for i in stock.options:
        opt_chain = stock.option_chain(i)
        # calls
        calls = opt_chain.calls
        if len(calls) > 0:
            calls = calls[calls['contractSymbol'].str.startswith(ticker, na=False)]
            calls['type'] = 'call'
            calls['exp'] = calls['contractSymbol'].str.lstrip(ticker).str.slice(0, 6)
            calls['exp'] = pd.to_datetime(calls['exp'], format='%y%m%d')
            calls['lastTradeDate'] = pd.to_datetime(calls['lastTradeDate'].astype(str).str[:-6])
            calls['days_to_exp'] = round((calls['exp'] - calls['lastTradeDate']).dt.total_seconds() / (24 * 3600), 2)
        # puts
        puts = opt_chain.puts
        if len(puts) > 0:
            puts = puts[puts['contractSymbol'].str.startswith(ticker, na=False)]
            puts['type'] = 'put'
            puts['exp'] = puts['contractSymbol'].str.lstrip(ticker).str.slice(0, 6)
            puts['exp'] = pd.to_datetime(puts['exp'], format='%y%m%d')
            puts['lastTradeDate'] = pd.to_datetime(puts['lastTradeDate'].astype(str).str[:-6])
            puts['days_to_exp'] = round((puts['exp'] - puts['lastTradeDate']).dt.total_seconds() / (24 * 3600), 2)
            # concat and store to list
        option_df_list.append(pd.concat([calls, puts]))

    df = pd.concat(option_df_list).reset_index(drop=True)
    df['lastTradeDate'] = df['lastTradeDate'].astype(str).str[:10]
    df['stockPrice'] = df.apply(lambda row: get_stock_price(ticker, row['lastTradeDate'], price_dict, stock), axis=1)
    df['option_ticker'] = ticker
    return df


def prepare_ml_dataset(df):
    # Binary and Categorical
    df['inTheMoney'] = df['inTheMoney'].astype(int)
    cat_cols = [
        'type',
        # 'option_ticker'
    ]
    df['exp'] = df['exp'].astype(str)
    encoder = OneHotEncoder(
        dtype=int,
        handle_unknown="infrequent_if_exist",
        min_frequency=10
    )
    encoded_cols = pd.DataFrame(encoder.fit_transform(df[cat_cols]).toarray())
    encoded_cols.columns = encoder.get_feature_names_out(cat_cols)
    df = pd.concat([df, encoded_cols], axis=1)

    # remove NaN volume
    df = df.dropna(subset=['volume', 'openInterest', 'percentChange'])
    df_orig = df
    df = df.drop(['contractSymbol',
                  'option_ticker',
                  'bid', 'ask',
                  # 'change',
                  'contractSize', 'currency', 'type', 'exp', 'lastTradeDate'], axis=1)

    end_col = ['lastPrice']
    start_col = []
    new_order = start_col + [item for item in df.columns if item not in end_col + start_col] + end_col
    df = df[new_order]
    df = df[df['lastPrice'] > 10].reset_index(drop=True)
    return df
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    make_scorer, r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import train_test_split
import shap
from IPython.display import display
import warnings
import numpy as np
import scipy.stats as si
from itertools import zip_longest

warnings.filterwarnings("ignore")

def train_model(df, ax_shap_sum=None, ticker=None):
    # train test split
    X, y = df.iloc[:, :-1], df.iloc[:, -1:].to_numpy().ravel()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, random_state=1337, test_size=0.25
    )

    # Gridsearch CV
    max_depth_range = [2, 3, 4]
    n_estimators_range = [200, 250]

    steps = [
        ('scaler', StandardScaler()),
        ('clf', None),
    ]
    pipe = Pipeline(steps)
    param_grid = [
        {
            'clf': [LinearRegression()],
        },
        {
            'clf': [DecisionTreeRegressor()],
            'clf__max_depth': max_depth_range
        },
        {
            'clf': [
                RandomForestRegressor(),
                GradientBoostingRegressor(),
            ],
            'clf__max_depth': max_depth_range,
            'clf__n_estimators': n_estimators_range,
        }
    ]
    cv = 5
    met_grid = [
        'r2',
        'neg_mean_absolute_error',
        'neg_mean_squared_error',
        'neg_mean_absolute_percentage_error'
    ]
    grid_search = GridSearchCV(
        pipe, param_grid, scoring=met_grid, refit='r2',
        cv=cv, n_jobs=-1, return_train_score=True
    )
    grid_search.fit(X_trainval, y_trainval)

    cv_results = pd.DataFrame(grid_search.__dict__['cv_results_'])
    params = [i for i in cv_results.columns if "param_clf" in i]
    cv_results['model'] = cv_results.apply(lambda row: concatenate_strings(row, params), axis=1)

    cv_columns_to_show = [
        'model',
        'mean_train_r2',
        'mean_test_r2',
        'mean_train_neg_mean_absolute_error',
        'mean_test_neg_mean_absolute_error',
        'mean_train_neg_mean_squared_error',
        'mean_test_neg_mean_squared_error',
        'mean_train_neg_mean_absolute_percentage_error',
        'mean_test_neg_mean_absolute_percentage_error',
    ]

    best_model_df = cv_results[(cv_results['params'] == grid_search.best_params_)][
            cv_columns_to_show].reset_index(drop=True)

    rename_dict = {
        'mean_test_neg_mean_absolute_error': 'mean_test_mae',
        'mean_test_neg_mean_squared_error': 'mean_test_mse',
        'mean_train_neg_mean_absolute_error': 'mean_train_mae',
        'mean_train_neg_mean_squared_error': 'mean_test_mse',
        'mean_train_neg_mean_absolute_percentage_error': 'mean_train_mape',
        'mean_test_neg_mean_absolute_percentage_error': 'mean_test_mape',
    }
    best_model_df = best_model_df.rename(columns=rename_dict)
    for cols_ in best_model_df._get_numeric_data().columns:
        best_model_df[cols_] = best_model_df[cols_].abs()

    new_columns = pd.MultiIndex.from_tuples([
        ('model', ''),
        ('r2', 'train'),
        ('r2', 'test'),
        ('mae', 'train'),
        ('mae', 'test'),
        ('mse', 'train'),
        ('mse', 'test'),
        ('mape', 'train'),
        ('mape', 'test'),
    ])
    best_model_df.columns = new_columns
    # display(best_model_df.round(3))

    bs_predict = X_test.apply(black_scholes_apply, axis=1)

    # print("bsm performance:")
    bsm = pd.DataFrame(
        data=[[
            'Black-Scholes-Merton',
            r2_score(y_test, bs_predict),
            mean_absolute_error(y_test, bs_predict),
            mean_squared_error(y_test, bs_predict),
            mean_absolute_percentage_error(y_test, bs_predict),
          ]],
        columns=['model', 'r2', 'mae', 'mse', 'mape'])
    # display(bsm)

    best_estimator = grid_search.best_estimator_
    best_estimator.fit(X_trainval, y_trainval)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ("GradientBoostingClassifier", best_estimator.named_steps['clf']),
        # ("GradientBoostingClassifier", GradientBoostingRegressor(max_depth=4, n_estimators=200))
    ])
    pipeline.fit(X_trainval, y_trainval)
    holdout_preds = pipeline.predict(X_test)
    bm = pipeline.named_steps['GradientBoostingClassifier']
    bm.fit(X_trainval, y_trainval)

    shap_explainer = shap.Explainer(
        pipeline.named_steps['GradientBoostingClassifier'], X_trainval, feature_names=X.columns
    )
    shap_values = shap_explainer(
        X_test,
        check_additivity=False
    )

    if ax_shap_sum != None:
        plt.sca(ax_shap_sum)
    shap.summary_plot(shap_values, plot_type='bar', feature_names=X.columns, max_display=5, show=False)
    if ax_shap_sum != None:
        ax_shap_sum.set_xlabel('mean shap value', fontsize=7)
        ax_shap_sum.tick_params(axis='y', labelsize=7)
        ax_shap_sum.tick_params(axis='x', labelsize=7)
    if ticker != None and ax_shap_sum != None:
        ax_shap_sum.set_title(f"{ticker}", fontsize=7)

    feature_names = X.columns
    rf_resultX = pd.DataFrame(
        shap_values.values,
        columns=feature_names
    )
    vals = np.abs(rf_resultX.values).mean(0)
    shap_importance = pd.DataFrame(
        list(zip(feature_names, vals)),
        columns=['col_name', 'feature_importance_vals']
    )
    shap_importance.sort_values(
        by=['feature_importance_vals'],
        ascending=False,
        inplace=True
    )

    return X_test, y_test, shap_values, X, shap_explainer, bm, (best_model_df, bsm)


def train_model_regularization(df, ax_shap_sum=None, ticker=None):
    # train test split
    X, y = df.iloc[:, :-1], df.iloc[:, -1:].to_numpy().ravel()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, random_state=1337, test_size=0.25
    )

    # Gridsearch CV
    max_depth_range = [3, 4]
    n_estimators_range = [100]
    max_leaf_nodes_range =[10, 20]
    min_samples_split_range=[4, 10]
    min_samples_leaf_range= [3, 5]
    steps = [
        ('scaler', StandardScaler()),
        ('clf', None),
    ]
    pipe = Pipeline(steps)
    param_grid = [
        {
            'clf': [
                GradientBoostingRegressor(max_features="sqrt"),
            ],
            'clf__max_depth': max_depth_range,
            'clf__n_estimators': n_estimators_range,
            'clf__max_leaf_nodes': max_leaf_nodes_range,
            'clf__min_samples_split': min_samples_split_range,
            'clf__min_samples_leaf': min_samples_leaf_range
        }
    ]
    cv = 5
    met_grid = [
        'r2',
        'neg_mean_absolute_error',
        'neg_mean_squared_error',
        'neg_mean_absolute_percentage_error'
    ]
    grid_search = GridSearchCV(
        pipe, param_grid, scoring=met_grid, refit='r2',
        cv=cv, n_jobs=-1, return_train_score=True
    )
    grid_search.fit(X_trainval, y_trainval)

    cv_results = pd.DataFrame(grid_search.__dict__['cv_results_'])
    params = [i for i in cv_results.columns if "param_clf" in i]
    cv_results['model'] = cv_results.apply(lambda row: concatenate_strings(row, params), axis=1)

    cv_columns_to_show = [
        'model',
        'mean_train_r2',
        'mean_test_r2',
        'mean_train_neg_mean_absolute_error',
        'mean_test_neg_mean_absolute_error',
        'mean_train_neg_mean_squared_error',
        'mean_test_neg_mean_squared_error',
        'mean_train_neg_mean_absolute_percentage_error',
        'mean_test_neg_mean_absolute_percentage_error',
    ]

    best_model_df = cv_results[(cv_results['params'] == grid_search.best_params_)][
            cv_columns_to_show].reset_index(drop=True)

    rename_dict = {
        'mean_test_neg_mean_absolute_error': 'mean_test_mae',
        'mean_test_neg_mean_squared_error': 'mean_test_mse',
        'mean_train_neg_mean_absolute_error': 'mean_train_mae',
        'mean_train_neg_mean_squared_error': 'mean_test_mse',
        'mean_train_neg_mean_absolute_percentage_error': 'mean_train_mape',
        'mean_test_neg_mean_absolute_percentage_error': 'mean_test_mape',
    }
    best_model_df = best_model_df.rename(columns=rename_dict)
    for cols_ in best_model_df._get_numeric_data().columns:
        best_model_df[cols_] = best_model_df[cols_].abs()

    new_columns = pd.MultiIndex.from_tuples([
        ('model', ''),
        ('r2', 'train'),
        ('r2', 'test'),
        ('mae', 'train'),
        ('mae', 'test'),
        ('mse', 'train'),
        ('mse', 'test'),
        ('mape', 'train'),
        ('mape', 'test'),
    ])
    best_model_df.columns = new_columns
    # display(best_model_df.round(3))

    bs_predict = X_test.apply(black_scholes_apply, axis=1)

    # print("bsm performance:")
    bsm = pd.DataFrame(
        data=[[
            'Black-Scholes-Merton',
            r2_score(y_test, bs_predict),
            mean_absolute_error(y_test, bs_predict),
            mean_squared_error(y_test, bs_predict),
            mean_absolute_percentage_error(y_test, bs_predict),
          ]],
        columns=['model', 'r2', 'mae', 'mse', 'mape'])
    # display(bsm)

    best_estimator = grid_search.best_estimator_
    best_estimator.fit(X_trainval, y_trainval)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ("GradientBoostingClassifier", best_estimator.named_steps['clf']),
        # ("GradientBoostingClassifier", GradientBoostingRegressor(max_depth=4, n_estimators=200))
    ])
    pipeline.fit(X_trainval, y_trainval)
    holdout_preds = pipeline.predict(X_test)
    bm = pipeline.named_steps['GradientBoostingClassifier']
    bm.fit(X_trainval, y_trainval)

    shap_explainer = shap.Explainer(
        pipeline.named_steps['GradientBoostingClassifier'], X_trainval, feature_names=X.columns
    )
    shap_values = shap_explainer(
        X_test,
        check_additivity=False
    )

    if ax_shap_sum != None:
        plt.sca(ax_shap_sum)
        shap.summary_plot(shap_values, plot_type='bar', feature_names=X.columns, max_display=5, show=False)

    if ax_shap_sum != None:
        ax_shap_sum.set_xlabel('mean shap value', fontsize=7)
        ax_shap_sum.tick_params(axis='y', labelsize=7)
        ax_shap_sum.tick_params(axis='x', labelsize=7)
    if ticker != None and ax_shap_sum != None:
        ax_shap_sum.set_title(f"{ticker}", fontsize=7)

    feature_names = X.columns
    rf_resultX = pd.DataFrame(
        shap_values.values,
        columns=feature_names
    )
    vals = np.abs(rf_resultX.values).mean(0)
    shap_importance = pd.DataFrame(
        list(zip(feature_names, vals)),
        columns=['col_name', 'feature_importance_vals']
    )
    shap_importance.sort_values(
        by=['feature_importance_vals'],
        ascending=False,
        inplace=True
    )

    return X_test, y_test, shap_values, X, shap_explainer, bm, (best_model_df, bsm)


def view_shap_value_for_instance(
    X_test, y_test, shap_values, X, shap_explainer, bm, instance_index, ax_shap_sum2, ticker
):
    shap_value_instance = shap_values[instance_index]
    instance_data = X_test.iloc[instance_index]

    # Create the SHAP waterfall plot for the chosen instance
    if ax_shap_sum2 != None:
        plt.sca(ax_shap_sum2)
        shap.waterfall_plot(shap.Explanation(values=shap_value_instance,
                                             base_values=shap_explainer.expected_value,
                                             data=instance_data,
                                             feature_names=X.columns), max_display=6, show=False)
        font_size = 7
        ax_shap_sum2.tick_params(axis='y', labelsize=font_size)
        ax_shap_sum2.tick_params(axis='x', labelsize=font_size)
        ax_shap_sum2.set_title(f"{ticker}", fontsize=font_size)
        ax_shap_sum2.tick_params(axis='y', which='major', pad=27.5)

        shared_axes = ax_shap_sum2.get_shared_y_axes()
        for ax in shared_axes.get_siblings(ax_shap_sum2):
            for annotation in ax_shap_sum2.texts:
                # print(annotation)
                annotation.set_fontsize(font_size)

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(font_size)
                # item.set_color('black')
                try:
                    if len(item) > 1:
                        for more_items in item:
                            more_items.set_fontsize(font_size)
                            # more_items.set_color('black')
                except:
                    pass


def concatenate_strings(row, params_list):
    filtered_strings = [str(row[col]) for col in params_list if pd.notnull(row[col])]
    filtered_strings = [s.strip("()") for s in filtered_strings]
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
        # print(ticker)
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
    df['risk_free_rate'] = df.apply(lambda row: tnx_apply(row, price_dict), axis=1)
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
                  'percentChange',
                  'inTheMoney',
                  # 'change',
                  'contractSize', 'currency', 'type', 'exp', 'lastTradeDate'], axis=1)

    # df["moneyness"] = df["strike"] / df["stockPrice"]
    end_col = ['lastPrice']
    start_col = []
    new_order = start_col + [item for item in df.columns if item not in end_col + start_col] + end_col
    df = df[new_order]
    df = df[df['lastPrice'] > 10].reset_index(drop=True)
    df = df.drop(['type_put'], axis=1)
    df = df.reset_index(drop=True)

    return df


class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma, type_):
        self.S = S  # Underlying asset price
        self.K = K  # Option strike price
        self.T = T  # Time to expiration in years
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying asset
        self.type_ = type_

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_option_price(self):
        return (self.S * si.norm.cdf(self.d1(), 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(),
                                                                                                            0.0, 1.0))

    def put_option_price(self):
        return (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0.0, 1.0) - self.S * si.norm.cdf(-self.d1(),
                                                                                                             0.0, 1.0))
    def price(self):
        if self.type_ == "call":
            return self.call_option_price()
        else:
            return self.put_option_price()


def tnx_apply(value, price_dict):
    rf_rate_ticker = "^TNX"
    if price_dict.get("^TNX") == None:
        price_dict["^TNX"] = {}

    date = value['lastTradeDate']  # Change this to your desired date
    # print(date)
    # print(price_dict["^TNX"].get(date))

    if price_dict["^TNX"].get(date) == None:
        # Fetch historical data for the specified date
        tnx = yf.Ticker(rf_rate_ticker)
        val = round(tnx.history(start=date).iloc[0:1, :]['Close'].values[0], 2)
        price_dict["^TNX"][date] = val
        return val
    else:
        return round(price_dict["^TNX"][date], 2)


def black_scholes_apply(value):
    return BlackScholesModel(
        S=value['stockPrice'],
        K=value['strike'],
        T=value['days_to_exp']/365,
        r=value['risk_free_rate']/100,
        sigma=value['impliedVolatility'],
        type_='call' if value['type_call'] == 1 else 'put'
    ).price()

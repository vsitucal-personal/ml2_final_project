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
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import shap
from IPython.display import display
import warnings

warnings.filterwarnings("ignore")


def train_model(df):
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
    scoring = make_scorer(r2_score)
    grid_search = GridSearchCV(
        pipe, param_grid, scoring=scoring,
        cv=cv, n_jobs=-1, return_train_score=True
    )
    grid_search.fit(X_trainval, y_trainval)

    print("bests:")
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    cv_results = pd.DataFrame(grid_search.__dict__['cv_results_'])
    params = [i for i in cv_results.columns if "param_clf" in i]
    cv_results['idn'] = cv_results.apply(lambda row: concatenate_strings(row, params), axis=1)

    display(
        cv_results[['idn', 'params', 'mean_test_score']].sort_values(by=["mean_test_score"],
                                                                     ascending=False).reset_index(drop=True)
    )
    display(
        cv_results[(cv_results['params'] == grid_search.best_params_)][
            ['idn', 'params', 'mean_test_score']].reset_index(drop=True)
    )

    best_estimator = grid_search.best_estimator_
    best_estimator.fit(X_trainval, y_trainval)

    holdout_preds = best_estimator.predict(X_test)
    print("model performance:")
    print(r2_score(y_test, holdout_preds))
    print(mean_absolute_error(y_test, holdout_preds))
    print(mean_squared_error(y_test, holdout_preds))

    display(
        pd.DataFrame({'Predict': holdout_preds, 'Actual': y_test})
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ("GradientBoostingClassifier", GradientBoostingRegressor(max_depth=4, n_estimators=200))
    ])
    pipeline.fit(X_trainval, y_trainval)
    holdout_preds = pipeline.predict(X_test)
    # print(r2_score(y_test, holdout_preds))
    # print(mean_absolute_error(y_test, holdout_preds))
    # print(mean_squared_error(y_test, holdout_preds))
    bm = pipeline.named_steps['GradientBoostingClassifier']
    bm.fit(X_trainval, y_trainval)

    shap_explainer = shap.Explainer(
        pipeline.named_steps['GradientBoostingClassifier'], X_trainval, feature_names=X.columns
    )
    shap_values = shap_explainer(
        X_test,
        check_additivity=False
    )

    print("shap summary plots")
    shap.summary_plot(shap_values, X_test, feature_names=X.columns)
    shap.summary_plot(shap_values, plot_type='bar', feature_names=X.columns)

    shap_explanation = shap.Explanation(shap_values.values[:, :],
                                        shap_values.base_values[0],
                                        shap_values.data,
                                        feature_names=X.columns)

    print("shap values vs. feature values")
    for i in X.columns:
        shap.plots.scatter(shap_explanation[:, i])

    return X_test, y_test, shap_values, X, shap_explainer, bm


def view_shap_value_for_instance(X_test, y_test, shap_values, X, shap_explainer, bm, instance_index):
    # Choose an instance to explain (e.g., the first instance in the test set)
    display(X_test.iloc[[instance_index]])
    shap_value_instance = shap_values[instance_index]
    instance_data = X_test.iloc[instance_index]

    # Create the SHAP waterfall plot for the chosen instance
    ax = shap.waterfall_plot(shap.Explanation(values=shap_value_instance,
                                              base_values=shap_explainer.expected_value,
                                              data=instance_data,
                                              feature_names=X.columns), max_display=100, show=False)
    ax.set_position([1, 1, 1.2, 1])
    plt.show()

    print('predict')
    print(bm.predict(X_test.iloc[[instance_index]])[0])
    print('actual')
    print(y_test[instance_index])

    # shap.force_plot(shap.Explanation(values=shap_value_instance,
    #                                  base_values=shap_explainer.expected_value,
    #                                  data=instance_data,
    #                                  feature_names=X.columns), matplotlib=True, figsize=(60, 5))

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
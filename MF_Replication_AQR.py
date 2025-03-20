import numpy as np
import pandas as pd


def vol_scaling(strategy_returns, volatility=0.10):
    portfolio_volatility = strategy_returns.rolling(window=12).std() * np.sqrt(4)
    scaling_factor = volatility / portfolio_volatility
    adjusted_together_strategy = strategy_returns * scaling_factor.fillna(1)
    return adjusted_together_strategy


def price_trend(data, target_volatility=0.10):
    """
    Calculates the returns for a price trend-following strategy across different asset classes.
    For equities and bonds, long and short positions are based on expanding mean returns compared
    to 4-quarter returns, while for currencies and commodities, positions are based on the sign of the
    4-quarter return. Applies volatility targeting to each position based on 12-month rolling volatility.
    Parameters:
        data (pd.DataFrame): Log-transformed returns data for the asset class.
        target_volatility (float, optional): Target annual volatility level for the strategy, used to scale
                                             returns. Defaults to 0.10.
    Returns:
        tuple:
            adjusted_together_strategy (pd.Series): Combined returns of the strategy with volatility targeting applied.
            strategy_returns (pd.DataFrame): DataFrame of individual security returns based on trend signals.
    """
    # Calculate 4-quarter rolling and expanding mean returns
    data_4q = data.rolling(window=4).sum().dropna(how='all')

    # Calculate 12-month rolling volatility for volatility targeting
    rolling_volatility = data.rolling(window=12).std().dropna(how='all')
    data = data.loc[rolling_volatility.index]
    data_4q = data_4q.loc[rolling_volatility.index]

    # Initialize DataFrame to store individual security returns
    strategy_returns = pd.DataFrame(index=data.index, columns=data.columns)
    for i in data.index:
        # Determine long and short positions
        long_positions = data_4q.loc[i][data_4q.loc[i] >= 0].index.tolist()
        short_positions = data_4q.loc[i][data_4q.loc[i] < 0].index.tolist()

        current_index = data_4q.index.get_loc(i)
        if current_index + 1 < len(data_4q.index):
            next_date = data_4q.index[current_index + 1]
            vol_weights = ((target_volatility / np.sqrt(4)) / rolling_volatility.loc[i]).dropna()

            # Apply equal risk weights for long and short positions
            for security in long_positions:
                if security in vol_weights.index and security in data.columns:
                    strategy_returns.loc[next_date, security] = vol_weights[security] * data.loc[next_date, security]

            for security in short_positions:
                if security in vol_weights.index and security in data.columns:
                    strategy_returns.loc[next_date, security] = -vol_weights[security] * data.loc[next_date, security]

    strategy_returns = strategy_returns.apply(pd.to_numeric, errors='coerce')

    # Calculate combined strategy (together strategy) with volatility targeting
    together_strategy = (np.exp(strategy_returns) - 1).mean(axis=1)
    adjusted_together_strategy = vol_scaling(np.log(1 + together_strategy), target_volatility)

    return adjusted_together_strategy, strategy_returns


def calculate_summary_stats(strategy_series, individual_returns_df):
    """
    Calculates annual return, annual standard deviation, maximum drawdown,
    last quarter return, and last year return.

    Parameters:
        strategy_series (pd.Series): Combined strategy returns (together_strategy).
        individual_returns_df (pd.DataFrame): Individual security returns (strategy_returns).

    Returns:
        pd.DataFrame: Summary statistics for both combined and individual returns.
    """
    stats = {}

    def annualized_return(returns):
        return returns.mean() * 4

    def annualized_std(returns):
        return returns.std() * np.sqrt(4)

    def sharpe_ratio(returns):
        return (returns.mean() * 4) / (returns.std() * np.sqrt(4))

    def max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        drawdown = cumulative.div(cumulative.cummax()) - 1
        return drawdown.min()

    def last_period_return(returns):
        return returns.sum()

    # Calculate stats for the combined strategy
    stats['Combined'] = {
        'Annual Return': annualized_return(strategy_series),
        'Annual Std Dev': annualized_std(strategy_series),
        'Sharpe Ratio': sharpe_ratio(strategy_series),
        'Max Drawdown': max_drawdown(strategy_series),
        'Last Quarter Return': last_period_return(strategy_series.tail(1)),  # Last quarter return
        'Last Year Return': last_period_return(strategy_series.tail(4))  # Last year return
    }

    # Calculate stats for each individual security
    for col in individual_returns_df.columns:
        stats[col] = {
            'Annual Return': annualized_return(individual_returns_df[col]),
            'Annual Std Dev': annualized_std(individual_returns_df[col]),
            'Sharpe Ratio': sharpe_ratio(individual_returns_df[col]),
            'Max Drawdown': max_drawdown(individual_returns_df[col]),
            'Last Quarter Return': last_period_return(individual_returns_df[col].tail(1)),  # Last quarter return
            'Last Year Return': last_period_return(individual_returns_df[col].tail(4))  # Last year return
        }

    return pd.DataFrame(stats)


def equal_weighted_strategy(together_strategies, weights=None):
    """
    Computes the weighted strategy returns from a collection of combined strategies across asset classes,
    and applies volatility targeting to achieve the specified target volatility level.
    If returns are missing for a column, the weight is distributed equally over the remaining columns.

    Parameters:
        together_strategies (dict): Dictionary containing asset class names as keys and their respective
                                    strategy return Series as values.
        weights (dict, optional): Dictionary containing the weights for each asset class.
        If None, equal weights are assumed.

    Returns:
        pd.Series: Adjusted weighted strategy returns with applied volatility targeting.
    """
    if weights is None:
        combined_strategy = pd.concat(together_strategies, axis=1).dropna(how='all')
        weighted_returns = (np.exp(combined_strategy) - 1).mean(axis=1)
        adjusted_weighted_returns = vol_scaling(np.log(weighted_returns + 1))
    else:
        combined_strategy = pd.concat(together_strategies, axis=1).dropna(how='all')
        # Adjust weights for missing data and calculate weights (if missing return then put weight equally to others)
        weight_df = pd.DataFrame(weights, index=combined_strategy.index, columns=combined_strategy.columns)
        valid_data = combined_strategy.notna()
        adjusted_weights = weight_df * valid_data  # Zero out weights where data is missing
        adjusted_weights = adjusted_weights.div(adjusted_weights.sum(axis=1), axis=0)  # Normalize weights to sum to 1
        # Compute weighted returns
        weighted_returns = ((np.exp(combined_strategy) - 1) * adjusted_weights).sum(axis=1)
        # Apply volatility scaling
        adjusted_weighted_returns = vol_scaling(np.log(weighted_returns + 1))
    return adjusted_weighted_returns


def process_trend(asset_classes, trend_data, trend_function, trend_name, asset_class_w, file_path):
    summary_tables = {}
    together_strategies = {}
    individual_securities = {}

    for asset in asset_classes:
        # Read in the data for each asset class
        rets = pd.read_excel(file_path, f'{asset} Returns', index_col=0)

        if trend_name == 'Price':
            together_strategy, individual_security_returns = trend_function(np.log(1 + rets))
        else:
            together_strategy, individual_security_returns = trend_function(
                trend_data, np.log(1 + rets), asset_class=asset)

        # Calculate summary statistics
        summary_stats = calculate_summary_stats(together_strategy, individual_security_returns)
        summary_tables[asset] = summary_stats
        together_strategies[asset] = together_strategy  # Store for equal weighting
        individual_securities[asset] = individual_security_returns

    # Calculate the equal-weighted strategy, asset class weights applied to only economic strategy
    if trend_name == 'Price':
        equal_weighted_returns = equal_weighted_strategy(together_strategies)
    else:
        equal_weighted_returns = equal_weighted_strategy(together_strategies, asset_class_w)

    return together_strategies, summary_tables, equal_weighted_returns, individual_securities


# RISK AVERSION ECONOMIC TREND
def risk_aversion_economic_trend(equity_returns, asset_returns, target_vol=0.10, asset_class=None):
    """
    Strategy based on the risk aversion signal inferred from each country's 12-month equity returns. If a country’s
    equity return is positive, the strategy goes long on equities, currencies, and commodities, and short on bonds.
    If negative, it goes short on equities, currencies, and commodities, and long on bonds. For commodities, the
    strategy uses an equal-weighted equity portfolio signal derived from all equity indices.
    Parameters:
        equity_returns (pd.DataFrame): DataFrame of quarterly log returns for MSCI indices representing various
                                       countries' equity returns.
        asset_returns (pd.DataFrame): DataFrame of quarterly log returns for currencies, bonds, and commodities.
        target_vol (float, optional): Target annual volatility level for the strategy, used in volatility scaling.
                                      Defaults to 0.10.
        asset_class (str, optional): Specifies the asset class under consideration ('Commodity', 'Equity', 'Bond',
                                     'Currency'). Determines how each asset class responds to risk aversion signals.
    Returns:
        tuple:
            adjusted_together_strategy (pd.Series): Combined strategy returns with volatility targeting applied.
            strategy_returns (pd.DataFrame): Individual asset returns based on the risk aversion trend strategy,
                                             with volatility adjustments.
    """
    # Calculate one-year (4-quarter) rolling returns for equities/bonds/currencies and equal w for commodities
    eq_weighted = np.log((equity_returns.mean(axis=1)) + 1).rolling(window=4).sum()

    # Select developed countries for a single developed signal
    target_countries = ['Australia', 'Canada', 'Japan', 'Sweden', 'Switzerland',
                        'United Kingdom', 'Euro area', 'United States', 'New Zealand', 'Norway']
    available_countries = [country for country in target_countries if country in equity_returns.columns]
    dev_equity_only = equity_returns[available_countries]
    one_y_equity_return = (np.log(dev_equity_only + 1)).rolling(window=4).sum()

    # Calculate rolling volatility for asset returns
    rolling_volatility = asset_returns.rolling(window=12).std().dropna(how='all')

    asset_returns = asset_returns.loc[rolling_volatility.index]
    one_y_equity_return = one_y_equity_return.loc[rolling_volatility.index]
    eq_weighted = eq_weighted.loc[rolling_volatility.index]

    # Initialize DataFrame to store the strategy's returns for each asset
    strategy_returns = pd.DataFrame(index=asset_returns.index, columns=asset_returns.columns)

    # Loop through each date and set positions based on the equal-weighted signal
    for current_date in one_y_equity_return.index[:-1]:
        next_date = one_y_equity_return.index[one_y_equity_return.index.get_loc(current_date) + 1]
        vol_weights = ((target_vol / np.sqrt(4)) / rolling_volatility.loc[current_date]).dropna()

        # Use equal weighted signal for only commodities for risk aversion
        if asset_class == 'Commodity':
            for asset_name in asset_returns.columns:
                if eq_weighted[current_date] >= 0:
                    strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                   asset_returns.loc[next_date, asset_name])
                else:
                    strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                   -asset_returns.loc[next_date, asset_name])

        # Use country signal for risk aversion
        if asset_class in ['Equity', 'Bond', 'Currency', 'Interest']:
            for asset_name in asset_returns.columns:
                if asset_name in one_y_equity_return.columns:
                    if one_y_equity_return.loc[current_date, asset_name] >= 0:
                        if asset_class in ['Equity', 'Currency']:
                            strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                           asset_returns.loc[next_date, asset_name])
                        elif asset_class in ['Bond', 'Interest']:
                            strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                           -asset_returns.loc[next_date, asset_name])
                    elif one_y_equity_return.loc[current_date, asset_name] < 0:
                        if asset_class in ['Equity', 'Currency']:
                            strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                           -asset_returns.loc[next_date, asset_name])
                        elif asset_class in ['Bond', 'Interest']:
                            strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                           asset_returns.loc[next_date, asset_name])

    strategy_returns = strategy_returns.apply(pd.to_numeric, errors='coerce')

    # Calculate combined strategy (together strategy) with volatility targeting
    together_strategy = (np.exp(strategy_returns) - 1).mean(axis=1)
    adjusted_together_strategy = vol_scaling(np.log(1 + together_strategy), target_vol)

    return adjusted_together_strategy, strategy_returns


# MONETARY POLICY ECONOMIC TREND
def monetary_policy_economic_trend(two_y_yield, asset_returns, target_vol=0.10, asset_class=None):
    """
    Strategy based on changes in monetary policy, indicated by the two-year yield. If the yearly change in the
    two-year yield is positive, the strategy goes long on currencies and short on commodities, equities, and bonds;
    if the change is negative, the strategy goes long on commodities, equities, and bonds, and short on currencies.
    For commodities, only the USA two-year yield change is used.
    Parameters:
        two_y_yield (pd.DataFrame): DataFrame containing the yearly change in the two-year yield for each country or
                                    region under consideration.
        asset_returns (pd.DataFrame): DataFrame of quarterly log returns for currencies, bonds, equities, and
        commodities.
        target_vol (float, optional): Target annual volatility level for the strategy, used in volatility scaling.
                                      Defaults to 0.10.
        asset_class (str, optional): Specifies the asset class under consideration ('Commodity', 'Equity', 'Bond',
                                     or 'Currency'). Determines how each asset class responds to yield signals.
    Returns:
        tuple:
            adjusted_together_strategy (pd.Series): Combined strategy returns with volatility targeting applied.
            strategy_returns (pd.DataFrame): Individual asset returns based on the monetary policy trend strategy,
                                             with volatility adjustments.
    """
    # Calculate rolling volatility for asset returns
    rolling_volatility = asset_returns.rolling(window=12).std().dropna(how='all')

    asset_returns = asset_returns.loc[rolling_volatility.index]
    two_y_yield = two_y_yield.loc[rolling_volatility.index]

    # Initialize DataFrame to store the strategy's returns for each asset
    strategy_returns = pd.DataFrame(index=asset_returns.index, columns=asset_returns.columns)

    # Loop through each date and set positions based on the equal-weighted signal
    for current_date in two_y_yield.index[:-1]:  # Skip last index due to forward-looking
        next_date = two_y_yield.index[two_y_yield.index.get_loc(current_date) + 1]
        vol_weights = ((target_vol / np.sqrt(4)) / rolling_volatility.loc[current_date]).dropna()

        if asset_class == 'Commodity':
            # Use only the United States yield for commodities
            us_yield = two_y_yield.loc[current_date, 'United States']
            for asset_name in asset_returns.columns:
                if us_yield >= 0:
                    strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                   -asset_returns.loc[next_date, asset_name])
                else:
                    strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                   asset_returns.loc[next_date, asset_name])
        for asset_name in asset_returns.columns:
            if asset_name in two_y_yield.columns:
                if two_y_yield.loc[current_date, asset_name] >= 0:
                    if asset_class in ['Equity', 'Bond', 'Interest']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       -asset_returns.loc[next_date, asset_name])
                    elif asset_class in ['Currency']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       asset_returns.loc[next_date, asset_name])
                elif two_y_yield.loc[current_date, asset_name] < 0:
                    if asset_class in ['Equity', 'Bond', 'Interest']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       asset_returns.loc[next_date, asset_name])
                    elif asset_class in ['Currency']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       -asset_returns.loc[next_date, asset_name])

    strategy_returns = strategy_returns.apply(pd.to_numeric, errors='coerce')

    # Calculate combined strategy (together strategy) with volatility targeting
    together_strategy = (np.exp(strategy_returns) - 1).mean(axis=1)
    adjusted_together_strategy = vol_scaling(np.log(1 + together_strategy), target_vol)

    return adjusted_together_strategy, strategy_returns


# International Trade Economic Trend
def international_trade_economic_trend(int_trade, asset_returns, target_vol=0.10, asset_class=None):
    """
    Strategy based on international trade trends, calculated by assessing relative currency performance. For each
    currency, the strategy computes its relative return to every other currency and sums these returns. A positive
    relative return indicates a trade surplus signal, while a negative return suggests a trade deficit signal.
    Based on this signal, the strategy goes long or short on bonds, currencies, and commodities; and takes the
    opposite position for equities. For commodities, only the U.S. signal is used. For example, if AUDUSD and
    CADUSD appreciate by 5% and 9% respectively, Australia's trade signal would be positive by 1% (5% AUDUSD
    appreciation against USD minus 4% AUDCAD depreciation).
    Parameters:
        int_trade (pd.DataFrame): DataFrame of international trade data where each column represents a country's
                                  relative trade returns signal.
        asset_returns (pd.DataFrame): DataFrame of quarterly log returns for various assets (currencies, bonds,
                                      equities, and commodities).
        target_vol (float, optional): Target annual volatility level for the strategy. Defaults to 0.10.
        asset_class (str, optional): Specifies the asset class under consideration ('Commodity', 'Equity', 'Bond',
                                     'Currency', or 'Interest'). Determines how each asset class responds to trade
                                     signals.
    Returns:
        tuple:
            adjusted_together_strategy (pd.Series): Combined strategy returns with applied volatility targeting.
            strategy_returns (pd.DataFrame): Individual asset returns based on the international trade strategy,
                                             with volatility adjustments.
    """
    # Calculate rolling volatility for asset returns
    rolling_volatility = asset_returns.rolling(window=12).std().dropna(how='all')

    asset_returns = asset_returns.loc[rolling_volatility.index]
    int_trade = int_trade.loc[rolling_volatility.index]

    # Initialize DataFrame to store the strategy's returns for each asset
    strategy_returns = pd.DataFrame(index=asset_returns.index, columns=asset_returns.columns)

    # Loop through each date and set positions based on the equal-weighted signal
    for current_date in int_trade.index[:-1]:
        next_date = int_trade.index[int_trade.index.get_loc(current_date) + 1]
        vol_weights = ((target_vol / np.sqrt(4)) / rolling_volatility.loc[current_date]).dropna()

        if asset_class == 'Commodity':
            # Use only the United States trade for commodities
            us_trade = int_trade.loc[current_date, 'United States']
            for asset_name in asset_returns.columns:
                if us_trade >= 0:
                    strategy_returns.loc[next_date, asset_name] = vol_weights.get(asset_name, 0) * asset_returns.loc[
                        next_date, asset_name]
                else:
                    strategy_returns.loc[next_date, asset_name] = vol_weights.get(asset_name, 0) * -asset_returns.loc[
                        next_date, asset_name]
        for asset_name in asset_returns.columns:
            if asset_name in int_trade.columns:
                if int_trade.loc[current_date, asset_name] >= 0:
                    if asset_class in ['Equity']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       -asset_returns.loc[next_date, asset_name])
                    elif asset_class in ['Bond', 'Currency', 'Interest']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       asset_returns.loc[next_date, asset_name])
                elif int_trade.loc[current_date, asset_name] < 0:
                    if asset_class in ['Equity']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       asset_returns.loc[next_date, asset_name])
                    elif asset_class in ['Bond', 'Currency', 'Interest']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       -asset_returns.loc[next_date, asset_name])

    strategy_returns = strategy_returns.apply(pd.to_numeric, errors='coerce')

    # Calculate combined strategy (together strategy) with volatility targeting
    together_strategy = (np.exp(strategy_returns) - 1).mean(axis=1)
    adjusted_together_strategy = vol_scaling(np.log(1 + together_strategy), target_vol)

    return adjusted_together_strategy, strategy_returns


# GDP GROWTH ECONOMIC TREND
def growth_economic_trend(gdp_growth, asset_returns, target_vol=0.10, asset_class=None):
    """
    Strategy based on GDP Growth forecasts. If the yearly change in GDP growth is positive, go long on equities,
    commodities, and currencies, and short bonds. If negative, go long on bonds and short other asset classes.
    For commodities, only the U.S. GDP growth forecast is used.
    Parameters:
        gdp_growth (pd.DataFrame): A DataFrame of yearly GDP growth forecast changes, where each column represents
                                   a GDP growth forecast for a specific country or region.
        asset_returns (pd.DataFrame): A DataFrame of quarterly log returns for various asset classes (currencies,
                                      bonds, equities, and commodities).
        target_vol (float, optional): Target annual volatility level for the strategy. Defaults to 0.10.
        asset_class (str, optional): Specifies the asset class under consideration ('Commodity', 'Equity',
                                     'Bond', 'Currency', or 'Interest'). Used to apply asset-specific logic.
    Returns:
        tuple:
            adjusted_together_strategy (pd.Series): Combined strategy returns with volatility targeting applied.
            strategy_returns (pd.DataFrame): Individual asset returns based on the GDP growth strategy, with
                                             volatility scaling.
    """
    # Calculate rolling volatility for asset returns
    rolling_volatility = asset_returns.rolling(window=12).std().dropna(how='all')

    asset_returns = asset_returns.loc[rolling_volatility.index]
    gdp_growth = gdp_growth.loc[rolling_volatility.index]

    # Initialize DataFrame to store the strategy's returns for each asset
    strategy_returns = pd.DataFrame(index=asset_returns.index, columns=asset_returns.columns)

    # Loop through each date and set positions based on the equal-weighted signal
    for current_date in gdp_growth.index[:-1]:
        next_date = gdp_growth.index[gdp_growth.index.get_loc(current_date) + 1]
        vol_weights = ((target_vol / np.sqrt(4)) / rolling_volatility.loc[current_date]).dropna()

        if asset_class == 'Commodity':
            # Use only the United States growth for commodities
            us_growth = gdp_growth.loc[current_date, 'United States']
            for asset_name in asset_returns.columns:
                if us_growth >= 0:
                    strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                   asset_returns.loc[next_date, asset_name])
                else:
                    strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                   -asset_returns.loc[next_date, asset_name])
        for asset_name in asset_returns.columns:
            if asset_name in gdp_growth.columns:
                if gdp_growth.loc[current_date, asset_name] >= 0:
                    if asset_class in ['Equity', 'Currency']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       asset_returns.loc[next_date, asset_name])
                    elif asset_class in ['Bond', 'Interest']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       -asset_returns.loc[next_date, asset_name])
                elif gdp_growth.loc[current_date, asset_name] < 0:
                    if asset_class in ['Equity', 'Currency']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       -asset_returns.loc[next_date, asset_name])
                    elif asset_class in ['Bond', 'Interest']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       asset_returns.loc[next_date, asset_name])

    strategy_returns = strategy_returns.apply(pd.to_numeric, errors='coerce')

    # Calculate combined strategy (together strategy) with volatility targeting
    together_strategy = (np.exp(strategy_returns) - 1).mean(axis=1)
    adjusted_together_strategy = vol_scaling(np.log(1 + together_strategy), target_vol)

    return adjusted_together_strategy, strategy_returns


# INFLATION ECONOMIC TREND
def inflation_economic_trend(inflation, asset_returns, target_vol=0.10, asset_class=None):
    """
    Strategy based on inflation forecasts. If the yearly change in inflation is positive, go long on equities,
    currencies, and commodities, and short bonds. If negative, go long on bonds and short other asset classes.
    For commodities, only the U.S. inflation forecast is used.

    Parameters:
        inflation (pd.DataFrame): A DataFrame of yearly inflation forecast changes, where each column represents
                                  an inflation forecast for a specific country or region.
        asset_returns (pd.DataFrame): A DataFrame of quarterly log returns for various asset classes (currencies,
                                      bonds, equities, and commodities).
        target_vol (float, optional): Target annual volatility level for the strategy. Defaults to 0.10.
        asset_class (str, optional): Specifies the asset class under consideration ('Commodity', 'Equity',
                                     'Bond', 'Currency', or 'Interest'). Used to apply asset-specific logic.
    Returns:
        tuple:
            adjusted_together_strategy (pd.Series): Combined strategy returns with volatility targeting applied.
            strategy_returns (pd.DataFrame): Individual asset returns based on the inflation strategy, with
                                             volatility scaling.

    """
    # Calculate rolling volatility for asset returns
    rolling_volatility = asset_returns.rolling(window=12).std().dropna(how='all')

    asset_returns = asset_returns.loc[rolling_volatility.index]
    inflation = inflation.loc[rolling_volatility.index]

    # Initialize DataFrame to store the strategy's returns for each asset
    strategy_returns = pd.DataFrame(index=asset_returns.index, columns=asset_returns.columns)

    # Loop through each date and set positions based on the equal-weighted signal
    for current_date in inflation.index[:-1]:
        next_date = inflation.index[inflation.index.get_loc(current_date) + 1]
        vol_weights = ((target_vol / np.sqrt(4)) / rolling_volatility.loc[current_date]).dropna()

        if asset_class == 'Commodity':
            # Use only the United States inflation for commodities
            us_inflation = inflation.loc[current_date, 'United States']
            for asset_name in asset_returns.columns:
                if us_inflation >= 0:
                    strategy_returns.loc[next_date, asset_name] = vol_weights.get(asset_name, 0) * asset_returns.loc[
                        next_date, asset_name]
                else:
                    strategy_returns.loc[next_date, asset_name] = vol_weights.get(asset_name, 0) * -asset_returns.loc[
                        next_date, asset_name]
        for asset_name in asset_returns.columns:
            if asset_name in inflation.columns:
                if inflation.loc[current_date, asset_name] >= 0:
                    if asset_class in ['Currency']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       asset_returns.loc[next_date, asset_name])
                    elif asset_class in ['Equity', 'Bond', 'Interest']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       -asset_returns.loc[next_date, asset_name])
                elif inflation.loc[current_date, asset_name] < 0:
                    if asset_class in ['Currency']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       -asset_returns.loc[next_date, asset_name])
                    elif asset_class in ['Equity', 'Bond', 'Interest']:
                        strategy_returns.loc[next_date, asset_name] = (vol_weights.get(asset_name, 0) *
                                                                       asset_returns.loc[next_date, asset_name])

    strategy_returns = strategy_returns.apply(pd.to_numeric, errors='coerce')

    # Calculate combined strategy (together strategy) with volatility targeting
    together_strategy = (np.exp(strategy_returns) - 1).mean(axis=1)
    adjusted_together_strategy = vol_scaling(np.log(1 + together_strategy), target_vol)

    return adjusted_together_strategy, strategy_returns


def compute_international_trade_signals(file_path, countries):
    """
    Compute international trade signals for the specified countries.
    Signals are based on the rolling sum of currency returns for a given window.

    Parameters:
        file_path (str): Path to the input Excel file.
        countries (list): List of countries to calculate signals for.
    Returns:
        pd.DataFrame: DataFrame of international trade signals for each country.
    """
    int_trade = pd.read_excel(file_path, 'Currency Returns', index_col=0)
    int_trade = np.log(1 + int_trade)

    # Compute rolling sum of returns
    rolling_int_trade = int_trade.rolling(window=4).sum().dropna()
    # Initialize DataFrame to store signals
    int_trade_signal = pd.DataFrame(index=rolling_int_trade.index, columns=countries)
    # Calculate trade-weighted signals for each country
    for country in countries:
        other_countries = [c for c in countries if c != country]
        for other_country in other_countries:
            pair_name = f"{country[:3]}{other_country[:3]}"
            # Ensure the pair name exists in the DataFrame
            if pair_name not in rolling_int_trade.columns:
                rolling_int_trade[pair_name] = rolling_int_trade[country] - rolling_int_trade[other_country]
        # Sum all relevant currency pairs to get the trade-weighted signal
        relevant_columns = [f"{country[:3]}{v[:3]}" for v in other_countries] + [country]
        int_trade_signal[country] = rolling_int_trade[relevant_columns].sum(axis=1)
    # Add United States signal as the negative sum of all other countries
    int_trade_signal['United States'] = -int_trade_signal.sum(axis=1)
    return int_trade_signal


def output_to_excel(output_path, strategy_df, asset_classes_to_export):
    with (pd.ExcelWriter(output_path, engine='xlsxwriter') as writer):
        for strategy_name, to_export in strategy_df.items():
            # Prepare equal-weighted returns DataFrame
            equal_weighted_df = pd.DataFrame(to_export[-1], columns=[strategy_name])
            equal_weighted_df.index = equal_weighted_df.index.date

            # Write equal-weighted returns to Excel
            equal_weighted_df.to_excel(writer, sheet_name=strategy_name, startcol=0, index=True)

            # Initialize starting row for summary tables
            start_row = 1

            for i, table in enumerate(to_export[len(asset_classes_to_export):-1]):
                # Remove columns with NaN in 'Annual Return' row
                if 'Annual Return' in table.index:
                    table = table.loc[:, ~table.loc['Annual Return'].isna()]

                # Add a label and write each table below the previous one, starting from Column C
                writer.sheets[strategy_name].write(start_row - 1, 3, f'{asset_classes_to_export[i]} Summary Table')
                table.to_excel(writer, sheet_name=strategy_name, startrow=start_row, startcol=3, index=True)
                start_row += len(table) + 2  # Leave space between tables

            # Calculate and write statistics for equal-weighted portfolio
            ew_stats_df = calculate_summary_stats(equal_weighted_df[strategy_name], equal_weighted_df).drop(
                columns=['Combined'])
            ew_stats_df.to_excel(writer, sheet_name=strategy_name, startrow=start_row, startcol=3, index=True)
            writer.sheets[strategy_name].write(start_row - 1, 3, f'Equal-Weighted {strategy_name}')

            # Apply formatting
            workbook = writer.book
            worksheet = writer.sheets[strategy_name]
            percent_format = workbook.add_format({'num_format': '0.0%'})
            general_format = workbook.add_format({'num_format': '0.00'})

            # Set column widths and formats
            worksheet.set_column('A:A', 15)
            worksheet.set_column('B:B', 21, percent_format)
            worksheet.set_column('D:ZZ', 16, percent_format)

            # Apply the general format to each specified row from E to AC
            for row in [5, 13, 21, 29, 37, 45]:
                worksheet.conditional_format(f'E{row}:ZZ{row}', {'type': 'no_blanks', 'format': general_format})

    print(f"Data exported successfully to {output_path}")


def average_vol_scaled(strats, key=None, vol=0.10):
    # Convert log returns to simple returns before summing
    simple_returns = [np.exp(strat[0][key] if key is not None else strat[2]) - 1 for strat in strats]
    averaged_simple = sum(simple_returns) / len(strats)
    averaged_simple = np.log(1 + averaged_simple)
    return vol_scaling(averaged_simple, vol)


def main(file_path, output_path, asset_classes_used, asset_class_w):
    # PRICE
    price_strat = process_trend(asset_classes=asset_classes_used, trend_data=None, trend_function=price_trend,
                                trend_name='Price', asset_class_w=asset_class_w, file_path=file_path)

    # RA
    risk_aversion_equity_signal = pd.read_excel(file_path, 'Equity Returns', index_col=0)
    ra_strat = process_trend(asset_classes=asset_classes_used, trend_data=risk_aversion_equity_signal,
                             trend_function=risk_aversion_economic_trend, trend_name='RA', asset_class_w=asset_class_w,
                             file_path=file_path)

    # MP
    two_year_yields = pd.read_excel(file_path, '2 year yields', index_col=0)
    two_year_yields = two_year_yields - two_year_yields.shift(4)
    mp_strat = process_trend(asset_classes=asset_classes_used, trend_data=two_year_yields,
                             trend_function=monetary_policy_economic_trend, trend_name='MP',
                             asset_class_w=asset_class_w, file_path=file_path)

    # IT
    countries = ['Australia', 'Canada', 'Japan', 'New Zealand', 'Sweden', 'Switzerland', 'United Kingdom', 'Euro area',
                 'Norway']
    it_strat = process_trend(asset_classes=asset_classes_used,
                             trend_data=compute_international_trade_signals(file_path, countries),
                             trend_function=international_trade_economic_trend, trend_name='IT',
                             asset_class_w=asset_class_w, file_path=file_path)

    # Growth
    gdp_growth_data = pd.read_excel(file_path, 'GDP Growth', index_col=0)
    gdp_growth_data = gdp_growth_data - gdp_growth_data.shift(4)
    growth_strat = process_trend(asset_classes=asset_classes_used, trend_data=gdp_growth_data,
                                 trend_function=growth_economic_trend, trend_name='Growth', asset_class_w=asset_class_w,
                                 file_path=file_path)

    # INFLATION
    inflation_data = pd.read_excel(file_path, 'Inflation Rate', index_col=0)
    inflation_data = inflation_data - inflation_data.shift(4)
    inflation_strat = process_trend(asset_classes=asset_classes_used, trend_data=inflation_data,
                                    trend_function=inflation_economic_trend, trend_name='Inflation',
                                    asset_class_w=asset_class_w, file_path=file_path)

    # Prepare strategies for Excel
    # List of economic strategies to be averaged
    trend_strategies = [ra_strat, mp_strat, it_strat, growth_strat, inflation_strat]

    # Calculate each portfolio
    economic_strategy = average_vol_scaled(trend_strategies)
    price_economic_strategy = equal_weighted_strategy({'Price_Strategy': price_strat[2],
                                                       'Economic_Strategy': economic_strategy})

    # Asset-specific strategies
    asset_strategies = {asset: average_vol_scaled(trend_strategies, key=asset) for asset in asset_classes_used}

    # Define strategy components for each type
    trend_data = {
        'International_Trade': it_strat,
        'GDP_Growth': growth_strat,
        'Inflation': inflation_strat,
        'Risk_Aversion': ra_strat,
        'Monetary Policy': mp_strat,
        'Price': price_strat}

    # Generate strategies dictionary
    strategies = {
        **{
            name: [trend[0][key] for key in asset_classes_used if key in trend[0]] +
                  [trend[1][key] for key in asset_classes_used if key in trend[1]] +
                  [trend[2]]
            for name, trend in trend_data.items()},
        **{
            f"{asset} Economic": [asset_strategies[asset]]
            for asset in asset_classes_used if asset in asset_strategies},
        'Economic Strategy': [economic_strategy],
        'Price + Econ Strategy': [price_economic_strategy]}

    output_to_excel(output_path=output_path, strategy_df=strategies, asset_classes_to_export=asset_classes_used)

    return (it_strat, mp_strat, inflation_strat, ra_strat, growth_strat, price_strat, economic_strategy,
            asset_strategies)

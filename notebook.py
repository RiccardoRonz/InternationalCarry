import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import os
    from itertools import product
    import matplotlib.pyplot as plt
    from statsmodels.regression.recursive_ls import RecursiveLS
    from statsmodels.regression.linear_model import OLS

    return OLS, mo, np, pd, plt, product


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # International Carry

    Code for `Half a Century of Risk Premia in a Generalized Multi-Currency Carry Strategy: An Empirical Investigation` paper.

    Authors: Riccardo Rebonato, Riccardo Ronzani and Xuan Feng

    [GitHub Repository](https://github.com/RiccardoRonz/InternationalCarry/tree/main)
    """)
    return


@app.cell
def _(pd):
    # Fixed settings
    currency_options = ["USD", "EUR", "JPN", "GBP"]
    start_date = pd.to_datetime('1974-10-01')
    end_date = pd.to_datetime('2020-02-01')
    holding_period: pd.DateOffset = pd.DateOffset(months=1)
    reference_maturities: list[int] = list(range(1, 9*12+1))

    # Derived quantities
    date_list: list[str] = pd.date_range(
        start=pd.to_datetime(arg=start_date),
        end=pd.to_datetime(arg=end_date),
        freq='MS'
    ).strftime(date_format='%Y-%m-%d').to_list()
    return currency_options, date_list, holding_period, reference_maturities


@app.cell
def _(holding_period: "pd.DateOffset", np, pd):
    # Load data
    fx_rates_data = pd.read_parquet("data/fx_rates.parquet")
    yields_data = pd.read_parquet("data/yields.parquet")

    # Add forward yields
    fwd_ylds = yields_data[['date', 'currency', 'maturity', 'yield']].copy()
    fwd_ylds['maturity'] = fwd_ylds['maturity'] + holding_period.months
    fwd_ylds['date'] = fwd_ylds['date'] - holding_period
    yields = yields_data.merge(fwd_ylds, on=['date', 'currency', 'maturity'], suffixes=('', '_1n1'), how='left')
    yields['P0n'] = np.exp(-yields['yield'] * yields['maturity'] / 12)
    yields['P1n1'] = np.exp(-yields['yield_1n1'] * (yields['maturity'] - holding_period.months ) / 12)
    return fx_rates_data, yields


@app.cell
def _(
    currency_options,
    date_list: list[str],
    fx_rates_data,
    holding_period: "pd.DateOffset",
    pd,
    product,
    reference_maturities: list[int],
    yields,
):
    # Compute uconditional strategy data

    import re
    def compute_unconditional_strategy_data(
        currency_options: list[str],
        date_list: list[str],
        reference_maturities: list[int],
        holding_period: pd.DateOffset,
        fx_rates_data: pd.DataFrame,
        yields: pd.DataFrame
    ) -> pd.DataFrame:

        data = []

        for (X, L, H) in product(currency_options, currency_options, currency_options):

            # Create base dataframe
            df = pd.DataFrame(
                data=[(X, L, H, i[0], i[1]) for i in list(product(reference_maturities, date_list))],
                columns=['X', 'L', 'H', 'maturity', 't0']
            )
            df['t0'] = pd.to_datetime(df['t0'])
            df['t1'] = df['t0'] + holding_period

            # Attach FX data
            FX0XL = fx_rates_data[fx_rates_data['currencies'] == f'{X}_{L}'][['date', 'fx']].rename(columns={'fx': 'FX0XL', 'date': 't0'})
            FX1XL = fx_rates_data[fx_rates_data['currencies'] == f'{X}_{L}'][['date', 'fx']].rename(columns={'fx': 'FX1XL', 'date': 't1'})
            FX0XH = fx_rates_data[fx_rates_data['currencies'] == f'{X}_{H}'][['date', 'fx']].rename(columns={'fx': 'FX0XH', 'date': 't0'})
            FX1XH = fx_rates_data[fx_rates_data['currencies'] == f'{X}_{H}'][['date', 'fx']].rename(columns={'fx': 'FX1XH', 'date': 't1'})
            df = (
                df
                .merge(FX0XL, on='t0', how='left')
                .merge(FX1XL, on='t1', how='left')
                .merge(FX0XH, on='t0', how='left')
                .merge(FX1XH, on='t1', how='left')
            )

            # Attach yield data
            PL01 = yields[(yields['currency'] == L) & (yields['maturity'] == holding_period.months)][['date', 'P0n']].rename(columns={'P0n': 'PL01', 'date': 't0'})
            PH0n = yields[(yields['currency'] == H)][['date', 'maturity', 'P0n']].rename(columns={'P0n': 'PH0n', 'date': 't0'})
            PH1n1 = yields[(yields['currency'] == H)][['date', 'maturity', 'P1n1']].rename(columns={'P1n1': 'PH1n1', 'date': 't0'})
            PX01 = yields[(yields['currency'] == X) & (yields['maturity'] == holding_period.months)][['date', 'P0n']].rename(columns={'P0n': 'PX01', 'date': 't0'})
            PH01 = yields[(yields['currency'] == H) & (yields['maturity'] == holding_period.months)][['date', 'P0n']].rename(columns={'P0n': 'PH01', 'date': 't0'})

            df = (
                df
                .merge(PL01, on=['t0'], how='left')
                .merge(PH0n, on=['t0', 'maturity'], how='left')
                .merge(PH1n1, on=['t0', 'maturity'], how='left')
                .merge(PX01, on=['t0'], how='left')
                .merge(PH01, on=['t0'], how='left')
            )

            # Compute derived quantities
            df['FFX0_XL1'] = df['FX0XL'] * df['PL01'] / df['PX01']
            df['FPH0n1'] = df['PH0n'] / df['PH01']
            df['FFX0_XH1'] = df['FX0XH'] * df['PH01'] / df['PX01']

            # Compute PnL
            df['PnL'] = - df['FX1XL']  / (df['FX0XL'] * df['PL01']) + df['PH1n1'] * df['FX1XH'] / (df['FX0XH'] * df['PH0n'])
            df['PnL_eq4'] = (1/df['PX01']) * (-df['FX1XL']/df['FFX0_XL1'] + df['PH1n1']/df['FPH0n1'] * df['FX1XH']/df['FFX0_XH1'])

            # Assign to the dictionary
            data.append(df)

        # Concatenate all dataframes
        data: pd.DataFrame = pd.concat(data, axis=0).reset_index(drop=True)

        return data

    # Run strategy
    unconditional_strategy_data_across_maturities = compute_unconditional_strategy_data(
        currency_options=currency_options,
        date_list=date_list,
        reference_maturities=reference_maturities,
        holding_period=holding_period,
        fx_rates_data=fx_rates_data,
        yields=yields
    )

    # Aggregate across maturities
    unconditional_strategy_data = (
        unconditional_strategy_data_across_maturities
        .groupby(['t0', 'X', 'L', 'H'])[['PnL']]
        .mean()
        .reset_index()
        .sort_values(['t0', 'X', 'L', 'H'])
        .reset_index(drop=True)
    )
    unconditional_strategy_data['t1'] = unconditional_strategy_data['t0'] + holding_period
    unconditional_strategy_data = unconditional_strategy_data[['t0', 't1', 'X', 'L', 'H', 'PnL']].copy()
    return (unconditional_strategy_data,)


@app.cell
def _(OLS, currency_options, pd, product, unconditional_strategy_data):
    # Compute conditional strategy data
    def compute_conditional_strategy_data(
        unconditional_strategy_data: pd.DataFrame,
        currency_options: list[str]
    ) -> pd.DataFrame:

        # Prepare data for regression
        reg_data = pd.merge(
            unconditional_strategy_data,
            unconditional_strategy_data[['t1', 'X', 'L', 'H', 'PnL']].rename(columns={'t1': 't0', 'PnL': 'PnL_t1'}),
            on=['t0', 'X', 'L', 'H'],
        )
        reg_data['const'] = 1

        # Run regression to estimate beta and gamma
        cond_strat_params = []
        for (X, L, H) in product(currency_options, currency_options, currency_options):
            curr_reg_data = reg_data[(reg_data['X']==X) & (reg_data['L']==L) & (reg_data['H']==H)].copy().set_index('t1').dropna()
            curr_reg_data.index.freq = 'MS'
            regr_model = OLS(curr_reg_data['PnL'], curr_reg_data[['const', 'PnL_t1']])
            res = regr_model.fit()
            curr_cond_strat_params = res.params.to_frame(name='value').reset_index(names='parameter')
            curr_cond_strat_params['X'] = X
            curr_cond_strat_params['L'] = L
            curr_cond_strat_params['H'] = H
            curr_cond_strat_params = curr_cond_strat_params.pivot(index=['X', 'L', 'H'], columns='parameter', values='value').reset_index()
            cond_strat_params.append(curr_cond_strat_params)
        cond_strat_params = pd.concat(cond_strat_params, axis=0).reset_index(drop=True)

        # Conditional strategy
        cond_strat = reg_data.merge(cond_strat_params, on=['X', 'L', 'H'], how='left', suffixes=('', '_params'))
        cond_strat['expected_PnL'] = cond_strat['const_params'] + cond_strat['PnL_t1'] * cond_strat['PnL_t1_params']
        cond_strat['cond_strat_PnL'] = cond_strat['PnL'] * cond_strat['expected_PnL']

        return cond_strat[['t0', 't1', 'X', 'L', 'H', 'cond_strat_PnL']].rename(columns={'cond_strat_PnL': 'PnL'}).copy()

    # Run strategy
    conditional_strategy_data = compute_conditional_strategy_data(
        unconditional_strategy_data=unconditional_strategy_data,
        currency_options=currency_options
    )
    return (conditional_strategy_data,)


@app.cell
def _(conditional_strategy_data, pd, unconditional_strategy_data):
    # Aggregate results
    strategies_data = pd.merge(
        unconditional_strategy_data.rename(columns={'PnL': 'unconditional_PnL'}),
        conditional_strategy_data.rename(columns={'PnL': 'conditional_PnL'}),
        on=['t0', 't1', 'X', 'L', 'H'],
    )
    return (strategies_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Strategy Performance

    Choose currencies and the time periods for which to visualize the strategy performance.
    """)
    return


@app.cell
def _(currency_options, mo):
    multiselect_X_currency = mo.ui.multiselect(options=currency_options, value=currency_options, label="Investor Currencies (X)")
    multiselect_L_currency = mo.ui.multiselect(options=currency_options, value=currency_options, label="Funding Currencies (L)")
    multiselect_H_currency = mo.ui.multiselect(options=currency_options, value=currency_options, label="Investment Currencies (H)")

    select_viz_start_date = mo.ui.date(label="Start Date", value='1974-10-01', start='1974-10-01', stop='2020-02-01')
    select_viz_end_date = mo.ui.date(label="End Date", value='2020-02-01', start='1974-10-01', stop='2020-02-01')
    return (
        multiselect_H_currency,
        multiselect_L_currency,
        multiselect_X_currency,
        select_viz_end_date,
        select_viz_start_date,
    )


@app.cell
def _(mo, select_viz_end_date, select_viz_start_date):
    mo.md(
        f"""
        Select parameters for the visualization of the strategy performance:
    
        {select_viz_start_date} {select_viz_end_date}
        """
    )
    return


@app.cell
def _(
    mo,
    multiselect_H_currency,
    multiselect_L_currency,
    multiselect_X_currency,
):
    mo.vstack([
        mo.hstack([multiselect_X_currency, mo.md(f"Has value: {multiselect_X_currency.value}")], justify='start', gap=5),
        mo.hstack([multiselect_L_currency, mo.md(f"Has value: {multiselect_L_currency.value}")], justify='start', gap=5),
        mo.hstack([multiselect_H_currency, mo.md(f"Has value: {multiselect_H_currency.value}")], justify='start', gap=3.6),
    ])
    return


@app.cell
def _(
    multiselect_H_currency,
    multiselect_L_currency,
    multiselect_X_currency,
    pd,
    select_viz_end_date,
    select_viz_start_date,
    strategies_data,
):
    # Extract visualization data
    viz_start_date = pd.to_datetime(select_viz_start_date.value)
    viz_end_date = pd.to_datetime(select_viz_end_date.value)
    x_currencies = list(multiselect_X_currency.value)
    l_currencies = list(multiselect_L_currency.value)
    h_currencies = list(multiselect_H_currency.value)

    # Filter strategy data based on user selections
    viz_data = strategies_data[
        (strategies_data['t0'] >= viz_start_date) &
        (strategies_data['t0'] <= viz_end_date) &
        (strategies_data['X'].isin(x_currencies)) &
        (strategies_data['L'].isin(l_currencies)) &
        (strategies_data['H'].isin(h_currencies))
    ].copy()
    return h_currencies, l_currencies, viz_data, x_currencies


@app.cell
def _(holding_period: "pd.DateOffset", np, pd, viz_data):
    # Compute performance stats
    def compute_performance_stats(
        strategies_data: pd.DataFrame,
        holding_period: pd.DateOffset
    ) -> pd.DataFrame:

        # Compute Sharpe ratios
        sharpe_ratios = pd.merge(
            strategies_data.groupby(['X', 'L', 'H'])[['unconditional_PnL', 'conditional_PnL']].mean().rename(columns={'unconditional_PnL': 'unconditional_mean', 'conditional_PnL': 'conditional_mean'}),
            strategies_data.groupby(['X', 'L', 'H'])[['unconditional_PnL', 'conditional_PnL']].std().rename(columns={'unconditional_PnL': 'unconditional_std', 'conditional_PnL': 'conditional_std'}),
            on=['X', 'L', 'H']
            )
        sharpe_ratios['unconditional_sharpe'] = sharpe_ratios['unconditional_mean'] / sharpe_ratios['unconditional_std'] * np.sqrt(12 / holding_period.months)
        sharpe_ratios['conditional_sharpe'] = sharpe_ratios['conditional_mean'] / sharpe_ratios['conditional_std'] * np.sqrt(12 / holding_period.months)
        sharpe_ratios['difference_in_sharpe'] = sharpe_ratios['conditional_sharpe'] - sharpe_ratios['unconditional_sharpe'].mean()

        return sharpe_ratios

    # Compute performance stats
    sharpe_ratios = compute_performance_stats(
        strategies_data=viz_data,
        holding_period=holding_period
    )
    return (sharpe_ratios,)


@app.cell
def _(mo, pd, plt, sharpe_ratios):
    # Visualize sharpe ratios
    def plot_sharpe_ratios(
        sharpe_ratios: pd.DataFrame
    ):

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        sharpe_ratios['unconditional_sharpe'].plot(kind='hist', ax=axs[0], alpha=0.5, label='Unconditional')
        sharpe_ratios['conditional_sharpe'].plot(kind='hist', ax=axs[0], alpha=0.5, label='Conditional')
        sharpe_ratios['difference_in_sharpe'].plot(kind='hist', ax=axs[1], alpha=0.5, label='Difference (Cond - Uncond)', color='blue')
        axs[0].set_title('Sharpe Ratios')
        axs[1].set_title('Difference in Sharpe Ratios')
        axs[0].set_xlabel('Sharpe ratio')
        axs[1].set_xlabel('Sharpe ratio')
        axs[0].legend()
        axs[1].legend()
        axs[0].grid()
        axs[1].grid()
        sharpe_hist = mo.ui.matplotlib(plt.gca())
        return sharpe_hist

    plot_sharpe_ratios(sharpe_ratios)
    return


@app.cell
def _(viz_data):
    viz_data
    return


@app.cell
def _(
    h_currencies,
    l_currencies,
    mo,
    pd,
    plt,
    product,
    viz_data,
    x_currencies,
):
    # Visulize cumulative PnL
    def plot_cumulative_pnl(
        viz_data: pd.DataFrame,
        x_currencies: list[str],
        l_currencies: list[str],
        h_currencies: list[str]
    ):

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        for (x1, l1, h1) in product(x_currencies, l_currencies, h_currencies):
            curr_pnls = viz_data[(viz_data['X'] == x1) & (viz_data['L'] == l1) & (viz_data['H'] == h1)].set_index('t1')[['unconditional_PnL', 'conditional_PnL']].dropna()
            curr_pnls = (1+ curr_pnls).cumprod() - 1
            curr_pnls[['unconditional_PnL']].plot(ax=axs[0])
            curr_pnls[['conditional_PnL']].plot(ax=axs[1])
        # Remove legends for individual lines to avoid clutter, and add a single legend for the whole plot
        axs[0].set_title('Unconditional strategy cumulative PnL')
        axs[1].set_title('Conditional strategy cumulative PnL')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Cumulative PnL')
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Cumulative PnL')
        if len(x_currencies) * len(l_currencies) * len(h_currencies) > 10:
            axs[0].legend().remove()
            axs[1].legend().remove()
        cum_plot = mo.ui.matplotlib(plt.gca())
        return cum_plot

    plot_cumulative_pnl(
        viz_data=viz_data,
        x_currencies=x_currencies,
        l_currencies=l_currencies,
        h_currencies=h_currencies
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

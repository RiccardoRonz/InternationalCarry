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

    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # International Carry
    """)
    return


@app.cell
def _(pd):
    # Load data
    fx_rates_data = pd.read_parquet("data/fx_rates.parquet")
    yields_data = pd.read_parquet("data/yields.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

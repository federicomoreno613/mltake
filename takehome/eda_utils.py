
import pandas as pd
import numpy as np
import pandas as pd
def analisis_dataset(df):
    resultado = pd.DataFrame()

    for columna in df.columns:
        datos = {}
        datos['nulls'] = df[columna].isnull().sum()
        datos['% nulls'] = (df[columna].isnull().mean() * 100).round(2)
        datos['unique_values'] = df[columna].nunique()
        datos['dtype'] = df[columna].dtype

        if df[columna].dtype in ['int64', 'float64']:
            datos["max"] = df[columna].max()
            datos["min"] = df[columna].min()
            datos["mean"] = df[columna].mean()
            datos["std"] = df[columna].std()
            datos["1stq"] = df[columna].quantile(0.25)
            datos["2ndq"] = df[columna].quantile(0.50)
            datos["3rdq"] = df[columna].quantile(0.75)
        else:
            muestra_valores = df[columna].dropna().unique()
            datos['sample_5'] = muestra_valores[:5]

        resultado[columna] = pd.Series(datos)

    return resultado.transpose().sort_values(by='nulls', ascending=False)



def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


import pandas as pd
from scipy import stats

from scipy import stats
def evaluate_statistical_difference(df, target_column, feature_column):
    """
    Evaluates the statistical difference in feature_column between two groups defined by target_column.

    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    target_column (str): The name of the column in df that contains the target variable.
    feature_column (str): The name of the column in df to evaluate for statistical difference.

    Returns:
    None: Prints the results of the statistical tests.
    """

    # Check for null values
    if df[feature_column].isnull().any():
        print(f"There are null values in the {feature_column}.")
        df = df.dropna(subset=[feature_column])

    # Normality test (e.g., Shapiro-Wilk)
    stat, p = stats.shapiro(df[feature_column])
    print(f'Shapiro-Wilk statistic={stat:.3f}, p-value={p:.3f}')

    # Ensure p_val is defined in both branches of the conditional
    p_val = np.nan  # Initialize p_val as None to handle it later

    # T-test or Mann-Whitney U test based on normality
    if p > 0.05:
        print('Distribution appears to be normal. Proceeding with t-test.')
        t_stat, p_val = stats.ttest_ind(df[df[target_column] == 0][feature_column],
                                        df[df[target_column] == 1][feature_column],
                                        equal_var=False, nan_policy='omit')
        print(f"Welch's t-test statistic: {t_stat}, p-value: {p_val}")
    else:
        print('Distribution does not appear to be normal. Proceeding with Mann-Whitney U test.')
        u_stat, p_val = stats.mannwhitneyu(df[df[target_column] == 0][feature_column],
                                           df[df[target_column] == 1][feature_column],
                                           alternative='two-sided')
        print(f"Mann-Whitney U statistic: {u_stat}, p-value: {p_val}")

    # Interpret the meaning of the p-value
    alpha = 0.05
    if p_val is not None and p_val < alpha:
        print('Significant difference between groups (reject H0)')
    else:
        print('No significant difference between groups (fail to reject H0)')
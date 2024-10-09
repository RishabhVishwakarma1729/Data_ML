import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.holtwinters import Holt

# Reusable Functions

# Read file
def read_file(file_path, file_type):
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'excel':
            df = pd.read_excel(file_path)
        else:
            st.error("Unsupported file type.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Clean data
def clean_data(df):
    df_cleaned = df.drop_duplicates()
    df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
    df_cleaned = df_cleaned.dropna()
    return df_cleaned

# Describe data
def describe_data(df):
    st.write("Data Information:")
    buffer = pd.io.common.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.write("Statistical Summary:")
    st.write(df.describe())

# Rename columns
def rename_columns(df, new_columns):
    try:
        df = df.rename(columns=new_columns)
        return df
    except Exception as e:
        st.error(f"Error renaming columns: {e}")
        return df 

# Change data types
def change_data_types(df, column_types): 
    try:
        df = df.astype(column_types)
        return df
    except Exception as e:
        st.error(f"Error changing data types: {e}")
        return df

# Handle missing values
def handle_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        return df.dropna()
    else:
        st.error("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'.")
        return df

# Handle outliers
def handle_outliers(df, column, method='iqr'):
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]
    elif method == 'zscore':
        df = df[(np.abs(stats.zscore(df[column])) < 3)]
    else:
        st.error("Invalid method. Choose 'iqr' or 'zscore'.")
    return df

# Subset DataFrame
def sub_setting(df, condition):
    return df.query(condition)

# Sample data
def sample_data(df, n, method='random'):
    if method == 'random':
        return df.sample(n=n, random_state=42)
    elif method == 'stratified':
        return df.groupby('strata').apply(lambda x: x.sample(n=n, random_state=42)).reset_index(drop=True)
    else:
        st.error("Invalid method. Choose 'random' or 'stratified'.")
        return df

# Create new column
def create_new_column(df, new_column_name, calculation):
    df[new_column_name] = calculation
    return df

# Binning
def bin_data(df, column, bins, labels):
    try:
        df[f'{column}_binned'] = pd.cut(df[column], bins=bins, labels=labels)
        return df
    except Exception as e:
        st.error(f"Error binning data: {e}")
        return df

# Replace values
def replace_values(df, column_name, to_replace, value):
    df[column_name] = df[column_name].replace(to_replace, value)
    return df

# Visualization Functions
def plot_histogram(df, column):
    plt.figure()
    sns.histplot(df[column], kde=True)
    st.pyplot(plt)

def plot_boxplot(df, column):
    plt.figure()
    sns.boxplot(x=df[column])
    st.pyplot(plt)

def plot_scatter_plot(df, x_column, y_column):
    plt.figure()
    sns.scatterplot(x=df[x_column], y=df[y_column])
    st.pyplot(plt)

# Cross Tab Analysis
def cross_tab_analysis(df, column1, column2):
    return pd.crosstab(df[column1], df[column2])

# Pivot Table Analysis
def pivot_table_analysis(df, values, index, columns):
    return df.pivot_table(values=values, index=index, columns=columns, aggfunc='mean')

# Point Estimation
def point_estimation(df, column):
    return df[column].mean()

# Confidence Interval
def confidence_interval(df, column, confidence=0.95):
    mean = df[column].mean()
    sem = stats.sem(df[column])
    ci = stats.t.interval(confidence, len(df[column])-1, loc=mean, scale=sem)
    return ci

# Hypothesis Testing
def hypothesis_testing(df, null_hypothesis, alternative_hypothesis, alpha=0.05):
    st.write("Null Hypothesis:", null_hypothesis)
    st.write("Alternative Hypothesis:", alternative_hypothesis)
    
# t-test
def t_test(df, column1, column2):
    t_stat, p_value = stats.ttest_ind(df[column1], df[column2])
    return t_stat, p_value

# Z-test
def z_test(df, column, population_mean):
    z_stat = (df[column].mean() - population_mean) / (df[column].std() / np.sqrt(len(df[column])))
    p_value = stats.norm.sf(abs(z_stat)) * 2  # two-tailed test
    return z_stat, p_value

# Chi-Square Test
def chi_square_test(df, column1, column2):
    contingency_table = pd.crosstab(df[column1], df[column2])
    chi2, p, _, _ = stats.chi2_contingency(contingency_table)
    return chi2, p

# ANOVA
def anova(df, column1, column2):
    groups = [group[column1].values for name, group in df.groupby(column2)]
    f_stat, p_value = stats.f_oneway(*groups)
    return f_stat, p_value

# Mann-Whitney U Test
def mann_whitney_u_test(df, column1, column2):
    stat, p_value = stats.mannwhitneyu(df[column1], df[column2])
    return stat, p_value

# Wilcoxon Signed-Rank Test
def wilcoxon_signed_rank_test(df, column1, column2):
    stat, p_value = stats.wilcoxon(df[column1], df[column2])
    return stat, p_value

# Simple Linear Regression
def simple_linear_regression(df, x_column, y_column):
    model = LinearRegression()
    model.fit(df[[x_column]], df[y_column])
    return model.coef_, model.intercept_

# Multiple Regression
def multiple_regression(df, y_column, x_columns):
    model = LinearRegression()
    model.fit(df[x_columns], df[y_column])
    return model.coef_, model.intercept_

# Logistic Regression
def logistic_regression(df, y_column, x_columns):
    model = LogisticRegression()
    model.fit(df[x_columns], df[y_column])
    return model.coef_, model.intercept_

# Pearson Correlation Coefficient
def pearson_correlation(df, column1, column2):
    correlation, _ = stats.pearsonr(df[column1], df[column2])
    return correlation

# Spearman Rank Correlation
def spearman_rank_correlation(df, column1, column2):
    correlation, _ = stats.spearmanr(df[column1], df[column2])
    return correlation

# Streamlit App
st.title("Advanced Data Analysis App")

# File upload widget
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=['csv', 'xlsx'])
file_type = st.selectbox("Select file type", options=["csv", "excel"])

if uploaded_file is not None:
    df = read_file(uploaded_file, file_type)

    if df is not None:
        st.write("Data Preview:")
        st.dataframe(df.head())

        # Step 3: Clean Data
        if st.button("Clean Data"):
            df = clean_data(df)
            st.write("Cleaned Data:")
            st.dataframe(df.head())

        # Step 4: Describe Data
        if st.button("Describe Data"):
            describe_data(df)

        # Data Manipulation Options
        st.subheader("Data Manipulation")
        if st.button("Rename Columns"):
            new_columns = st.text_input("Enter new column names as comma-separated values")
            new_columns = [col.strip() for col in new_columns.split(',')]
            df.columns = new_columns
            st.write("Renamed Columns:", df.columns)

        if st.button("Change Data Types"):
            column_types = st.text_input("Enter column types as key=value pairs (e.g., column1=int, column2=float)")
            column_types = dict(pair.split('=') for pair in column_types.split(','))
            df = change_data_types(df, column_types)
            st.write("Data types changed.")

        if st.button("Handle Missing Values"):
            strategy = st.selectbox("Select strategy", options=['mean', 'median', 'mode', 'drop'])
            df = handle_missing_values(df, strategy)
            st.write("Missing values handled.")

        if st.button("Handle Outliers"):
            column = st.selectbox("Select column for outlier handling", options=df.columns)
            method = st.selectbox("Select method", options=['iqr', 'zscore'])
            df = handle_outliers(df, column, method)
            st.write("Outliers handled.")

        if st.button("Subset Data"):
            condition = st.text_input("Enter condition for subsetting (e.g., column1 > 0)")
            df_subset = sub_setting(df, condition)
            st.write("Subset Data:")
            st.dataframe(df_subset)

        if st.button("Sample Data"):
            n = st.number_input("Number of samples to draw", min_value=1, max_value=len(df))
            method = st.selectbox("Sampling method", options=['random', 'stratified'])
            sampled_df = sample_data(df, n, method)
            st.write("Sampled Data:")
            st.dataframe(sampled_df)

        if st.button("Create New Column"):
            new_column_name = st.text_input("Enter new column name")
            calculation = st.text_input("Enter calculation (e.g., column1 + column2)")
            df = create_new_column(df, new_column_name, eval(calculation))
            st.write("New Column Created.")

        if st.button("Binning"):
            column = st.selectbox("Select column for binning", options=df.columns)
            bins = st.text_input("Enter bin edges as comma-separated values")
            labels = st.text_input("Enter labels as comma-separated values")
            bins = [float(b) for b in bins.split(',')]
            labels = [label.strip() for label in labels.split(',')]
            df = bin_data(df, column, bins, labels)
            st.write("Binned Data Created.")

        if st.button("Replace Values"):
            column_name = st.selectbox("Select column for replacement", options=df.columns)
            to_replace = st.text_input("Enter value to replace")
            value = st.text_input("Enter new value")
            df = replace_values(df, column_name, to_replace, value)
            st.write("Values replaced.")

        # Visualization Options
        st.subheader("Data Visualization")
        if st.button("Histogram"):
            column = st.selectbox("Select column for histogram", options=df.columns)
            plot_histogram(df, column)

        if st.button("Boxplot"):
            column = st.selectbox("Select column for boxplot", options=df.columns)
            plot_boxplot(df, column)

        if st.button("Scatter Plot"):
            x_column = st.selectbox("Select X column", options=df.columns)
            y_column = st.selectbox("Select Y column", options=df.columns)
            plot_scatter_plot(df, x_column, y_column)

        # Statistical Analysis
        st.subheader("Statistical Analysis")
        if st.button("Cross Tab Analysis"):
            column1 = st.selectbox("Select first column", options=df.columns)
            column2 = st.selectbox("Select second column", options=df.columns)
            cross_tab_result = cross_tab_analysis(df, column1, column2)
            st.write(cross_tab_result)

        if st.button("Pivot Table Analysis"):
            values = st.selectbox("Select values column", options=df.columns)
            index = st.selectbox("Select index column", options=df.columns)
            columns = st.selectbox("Select columns column", options=df.columns)
            pivot_table_result = pivot_table_analysis(df, values, index, columns)
            st.write(pivot_table_result)

        if st.button("Point Estimation"):
            column = st.selectbox("Select column for point estimation", options=df.columns)
            estimate = point_estimation(df, column)
            st.write("Point Estimate:", estimate)

        if st.button("Confidence Interval"):
            column = st.selectbox("Select column for confidence interval", options=df.columns)
            ci = confidence_interval(df, column)
            st.write("Confidence Interval:", ci)

        if st.button("Hypothesis Testing"):
            null_hypothesis = st.text_input("Enter null hypothesis")
            alternative_hypothesis = st.text_input("Enter alternative hypothesis")
            hypothesis_testing(df, null_hypothesis, alternative_hypothesis)

        if st.button("T-Test"):
            column1 = st.selectbox("Select first column for t-test", options=df.columns)
            column2 = st.selectbox("Select second column for t-test", options=df.columns)
            t_stat, p_value = t_test(df, column1, column2)
            st.write("T-Statistic:", t_stat)
            st.write("P-Value:", p_value)

        if st.button("Z-Test"):
            column = st.selectbox("Select column for z-test", options=df.columns)
            population_mean = st.number_input("Enter population mean", value=0)
            z_stat, p_value = z_test(df, column, population_mean)
            st.write("Z-Statistic:", z_stat)
            st.write("P-Value:", p_value)

        if st.button("Chi-Square Test"):
            column1 = st.selectbox("Select first column for chi-square test", options=df.columns)
            column2 = st.selectbox("Select second column for chi-square test", options=df.columns)
            chi2, p = chi_square_test(df, column1, column2)
            st.write("Chi-Squared Statistic:", chi2)
            st.write("P-Value:", p)

        if st.button("ANOVA"):
            column1 = st.selectbox("Select dependent variable", options=df.columns)
            column2 = st.selectbox("Select independent variable", options=df.columns)
            f_stat, p_value = anova(df, column1, column2)
            st.write("F-Statistic:", f_stat)
            st.write("P-Value:", p_value)

        if st.button("Mann-Whitney U Test"):
            column1 = st.selectbox("Select first column", options=df.columns)
            column2 = st.selectbox("Select second column", options=df.columns)
            stat, p_value = mann_whitney_u_test(df, column1, column2)
            st.write("Statistic:", stat)
            st.write("P-Value:", p_value)

        if st.button("Wilcoxon Signed-Rank Test"):
            column1 = st.selectbox("Select first column", options=df.columns)
            column2 = st.selectbox("Select second column", options=df.columns)
            stat, p_value = wilcoxon_signed_rank_test(df, column1, column2)
            st.write("Statistic:", stat)
            st.write("P-Value:", p_value)

        if st.button("Simple Linear Regression"):
            x_column = st.selectbox("Select independent variable", options=df.columns)
            y_column = st.selectbox("Select dependent variable", options=df.columns)
            coef, intercept = simple_linear_regression(df, x_column, y_column)
            st.write("Coefficient:", coef)
            st.write("Intercept:", intercept)

        if st.button("Multiple Regression"):
            y_column = st.selectbox("Select dependent variable", options=df.columns)
            x_columns = st.multiselect("Select independent variables", options=df.columns)
            coef, intercept = multiple_regression(df, y_column, x_columns)
            st.write("Coefficients:", coef)
            st.write("Intercept:", intercept)

        if st.button("Logistic Regression"):
            y_column = st.selectbox("Select dependent variable (binary)", options=df.columns)
            x_columns = st.multiselect("Select independent variables", options=df.columns)
            coef, intercept = logistic_regression(df, y_column, x_columns)
            st.write("Coefficients:", coef)
            st.write("Intercept:", intercept)

        if st.button("Pearson Correlation"):
            column1 = st.selectbox("Select first column", options=df.columns)
            column2 = st.selectbox("Select second column", options=df.columns)
            correlation = pearson_correlation(df, column1, column2)
            st.write("Pearson Correlation Coefficient:", correlation)

        if st.button("Spearman Correlation"):
            column1 = st.selectbox("Select first column", options=df.columns)
            column2 = st.selectbox("Select second column", options=df.columns)
            correlation = spearman_rank_correlation(df, column1, column2)
            st.write("Spearman Rank Correlation Coefficient:", correlation)

st.write("Powered by Streamlit")

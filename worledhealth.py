import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

# Ignore warnings
warnings.filterwarnings("ignore")

# Set page layout
st.set_page_config(layout='wide')

# Load the dataset
data = pd.read_csv("./world_health_data.csv")

# Title and Sidebar
st.title("Advanced Data Analysis & Forecasting App")
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose an Option", [
    "Data Cleaning", "Exploratory Data Analysis", "Predictive Modeling", "Time Series Forecasting", "Country Insights"
])

# Sidebar Inputs for Model Hyperparameters
st.sidebar.header("Model Hyperparameters")
n_estimators = st.sidebar.slider("n_estimators", min_value=50, max_value=500, value=200, step=50)
max_depth = st.sidebar.slider("max_depth", min_value=1, max_value=50, value=20, step=1)
max_features = st.sidebar.selectbox("max_features", ["sqrt", "log2", None])
min_samples_leaf = st.sidebar.slider("min_samples_leaf", min_value=1, max_value=10, value=1, step=1)
min_samples_split = st.sidebar.slider("min_samples_split", min_value=2, max_value=10, value=2, step=1)
random_state = st.sidebar.slider("random_state", min_value=0, max_value=100, value=42, step=1)

if data is not None:
    st.write("### Dataset Preview")
    st.write(data.head())

    if option == "Data Cleaning":
        st.subheader("Missing Value Analysis")
        missing_values = data.isnull().sum()
        missing_values_sorted = missing_values[missing_values > 0].sort_values(ascending=False)
        if not missing_values_sorted.empty:
            st.write(missing_values_sorted)
            fig, ax = plt.subplots()
            missing_values_sorted.plot(kind='bar', color='salmon', ax=ax)
            ax.set_title("Missing Values per Column")
            st.pyplot(fig)

            if st.checkbox("Interpolate Missing Values"):
                data = data.interpolate()
                st.success("Missing values have been interpolated.")
        else:
            st.success("No missing values detected.")

    elif option == "Exploratory Data Analysis":
        st.subheader("Visualizations")
        selected_column = st.selectbox("Select a Column for Distribution Analysis", data.select_dtypes(include=np.number).columns)
        if selected_column:
            fig, ax = plt.subplots()
            sns.histplot(data[selected_column], kde=True, ax=ax, color="blue")
            ax.set_title(f"Distribution of {selected_column}")
            st.pyplot(fig)

        st.subheader("Correlation Analysis")
        if st.checkbox("Show Correlation Heatmap"):
            fig, ax = plt.subplots(figsize=(10, 6))
            data_numeric = data.select_dtypes(include=[float, int])
            sns.heatmap(data_numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        st.subheader("Country Insights: Health Data")
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        country_stats = data.groupby('country')[numeric_columns].mean()

        top_life_expectancy = country_stats['life_expect'].sort_values(ascending=False).head(10)
        top_maternal_mortality = country_stats['maternal_mortality'].sort_values(ascending=False).head(10)
        top_undernourishment = country_stats['prev_undernourishment'].sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(12, 6))
        top_life_expectancy.plot(kind='bar', color='green', ax=ax)
        ax.set_title('Top 10 Countries with Highest Average Life Expectancy')
        ax.set_ylabel('Life Expectancy (years)')
        ax.set_xlabel('Country')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        top_maternal_mortality.plot(kind='bar', color='red', ax=ax)
        ax.set_title('Top 10 Countries with Highest Maternal Mortality Rates')
        ax.set_ylabel('Maternal Mortality Rate (per 100,000 live births)')
        ax.set_xlabel('Country')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        top_undernourishment.plot(kind='bar', color='orange', ax=ax)
        ax.set_title('Top 10 Countries with Highest Prevalence of Undernourishment')
        ax.set_ylabel('Prevalence of Undernourishment (%)')
        ax.set_xlabel('Country')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    elif option == "Predictive Modeling":
        st.subheader("Predictive Modeling with Machine Learning")
        
        label_encoder_country = LabelEncoder()
        data['country_encoded'] = label_encoder_country.fit_transform(data['country'])

        columns_to_fill = ['life_expect', 'health_exp', 'infant_mortality', 'neonatal_mortality', 
                          'under_5_mortality', 'maternal_mortality', 'prev_undernourishment', 
                          'prev_hiv', 'inci_tuberc']
        for col in columns_to_fill:
            data[col] = data.groupby('country')[col].transform(
                lambda x: x.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            )

        data[columns_to_fill] = data[columns_to_fill].fillna(data[columns_to_fill].median())

        X = data.drop(columns=['country', 'country_code', 'life_expect'])
        y = data['life_expect']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            'Random Forest': RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                                   max_features=max_features, min_samples_leaf=min_samples_leaf, 
                                                   min_samples_split=min_samples_split, random_state=random_state),
            'Linear Regression': LinearRegression(),
            'Support Vector Machine': SVR(),
            'Decision Tree': DecisionTreeRegressor(max_depth=max_depth, max_features=max_features, 
                                                   min_samples_leaf=min_samples_leaf, 
                                                   min_samples_split=min_samples_split, random_state=random_state),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=2)
        }

        model_names = []
        mse_scores = []
        r2_scores = []

        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_names.append(model_name)
            mse_scores.append(mse)
            r2_scores.append(r2)

        results_df = pd.DataFrame({
            'Model': model_names,
            'Mean Squared Error (MSE)': mse_scores,
            'R-squared (R²)': r2_scores
        })

        styled_results = results_df.style.applymap(
            lambda x: 'background-color: #d65f5f' if x == results_df['Mean Squared Error (MSE)'].min() or x == results_df['R-squared (R²)'].max() else '', 
            subset=['Mean Squared Error (MSE)', 'R-squared (R²)']
        )

        st.write(styled_results)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.barh(results_df['Model'], results_df['Mean Squared Error (MSE)'], color='lightcoral')
        ax1.set_title('Mean Squared Error (MSE) Comparison')
        ax1.set_xlabel('MSE')
        st.pyplot(fig)

        fig, ax2 = plt.subplots(figsize=(10, 6))
        ax2.barh(results_df['Model'], results_df['R-squared (R²)'], color='lightgreen')
        ax2.set_title('R-squared (R²) Comparison')
        ax2.set_xlabel('R²')
        st.pyplot(fig)

    elif option == "Time Series Forecasting":
        st.subheader("Time Series Forecasting: Maternal Mortality Over Time")

        time_series_data = data.groupby('year')['maternal_mortality'].mean()

        plt.figure(figsize=(12, 6))
        plt.plot(time_series_data, label='Original Data', marker='o', color='black')
        plt.title('Maternal Mortality Over Time')
        plt.xlabel('Year')
        plt.ylabel('Maternal Mortality (per 100,000 live births)')
        plt.grid(True)
        st.pyplot(plt)

        arima_model = ARIMA(time_series_data, order=(1, 1, 1))
        arima_results = arima_model.fit()

        sarima_model = sm.tsa.SARIMAX(time_series_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_results = sarima_model.fit()

        arima_forecast = arima_results.get_forecast(steps=12)
        arima_predicted_mean = arima_forecast.predicted_mean
        sarima_forecast = sarima_results.get_forecast(steps=12)
        sarima_predicted_mean = sarima_forecast.predicted_mean

        time_series_data = time_series_data.dropna()
        arima_predicted_mean = arima_predicted_mean[~np.isnan(arima_predicted_mean)]

        min_length = min(len(time_series_data), len(arima_predicted_mean))
        time_series_data = time_series_data[-min_length:]
        arima_predicted_mean = arima_predicted_mean[-min_length:]

        arima_rmse = np.sqrt(mean_squared_error(time_series_data, arima_predicted_mean))
        sarima_rmse = np.sqrt(mean_squared_error(time_series_data, sarima_predicted_mean))

        plt.figure(figsize=(12, 6))
        plt.plot(time_series_data, label='Actual Data', color='black', marker='o')
        plt.plot(arima_predicted_mean, label='ARIMA Predictions', color='blue', linestyle='--', marker='x')
        plt.plot(sarima_predicted_mean, label='SARIMA Predictions', color='green', linestyle='--', marker='o')
        plt.title('ARIMA vs SARIMA Model Predictions', fontsize=18)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Maternal Mortality (per 100,000 live births)', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True)
        st.pyplot(plt)

        st.write("Handling NaN values in the predictions:")
        st.write(f"ARIMA RMSE: {arima_rmse}")
        st.write(f"SARIMA RMSE: {sarima_rmse}")
        # Data cleaning and preparation (assuming the relevant columns are 'year' and 'maternal_mortality')
        df = data[['year', 'maternal_mortality']].dropna()

        # Train a linear regression model
        X = df['year'].values.reshape(-1, 1)
        y = df['maternal_mortality']
        model = LinearRegression()
        model.fit(X, y)

        # User input to choose the year for prediction
        year_to_predict = st.number_input("Enter the year for prediction:", min_value=df['year'].max() + 1, max_value=df['year'].max() + 10, value=df['year'].max() + 1)

        # Predict maternal mortality for the selected year
        future_year = np.array([[year_to_predict]])
        future_prediction = model.predict(future_year)

        # Display the predicted value
        st.write(f"### Predicted Maternal Mortality for the Year {year_to_predict}:")
        st.write(f"{future_prediction[0]:.1f} per 100,000 live births")

        # Plot the results with emphasis on trends
        plt.figure(figsize=(18, 10))

        # Line plot for the original data (trend over time)
        plt.plot(df['year'], df['maternal_mortality'], label='Original Data', color='grey', alpha=0.5, linestyle='-', marker='o')

        # Regression line for the original data
        plt.plot(df['year'], model.predict(X), label='Regression Line (Original Data)', color='blue', linestyle='--')

        # Plot the selected year prediction
        plt.scatter(year_to_predict, future_prediction, color='purple', label=f'Prediction for {year_to_predict}', s=100, zorder=5)

        # Add annotations for future predictions
        plt.text(year_to_predict, future_prediction, f'{future_prediction[0]:.1f}', fontsize=12, ha='center', color='purple')

        # Add labels and title
        plt.title('Maternal Mortality Prediction (Linear Regression)', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Maternal Mortality (per 100,000 live births)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)

        # Show the plot
        st.pyplot(plt)

else:
    st.info("Dataset not found! Please check the file path.")


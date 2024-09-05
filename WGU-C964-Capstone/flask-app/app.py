from flask import Flask, render_template, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load the dataset
def load_data():
    df = pd.read_csv('Amazon-Products.csv')

    # Clean and process data
    # Drop unnecessary columns
    df_new = df.drop(['Unnamed: 0.1', 'Unnamed: 0', 'image', 'link'], axis=1)

    # Rename columns to match the expected names in the code
    df_new.columns = ['Name', 'Main_Category', 'Sub_Category', 'Ratings', 'No_of_Ratings', 'Discount_Price', 'Actual_Price']

    # Drop rows with missing values in important columns
    df_cleaned = df_new.dropna()

    # Handle non-numeric Ratings
    df_cleaned.loc[:, 'Ratings'] = pd.to_numeric(df_cleaned['Ratings'].replace('[a-zA-Z₹]', '', regex=True), errors='coerce')

    # Handle non-numeric No_of_Ratings
    df_cleaned.loc[:, 'No_of_Ratings'] = pd.to_numeric(df_cleaned['No_of_Ratings'].str.replace(',', ''), errors='coerce')

    # Handle non-numeric Discount_Price and Actual_Price
    df_cleaned.loc[:, 'Discount_Price'] = pd.to_numeric(df_cleaned['Discount_Price'].str.replace(',', '').str.replace('₹', ''), errors='coerce')
    df_cleaned.loc[:, 'Actual_Price'] = pd.to_numeric(df_cleaned['Actual_Price'].str.replace(',', '').str.replace('₹', ''), errors='coerce')

    # Drop missing values
    df_cleaned = df_cleaned.dropna()

    # Simulate sales volume
    df_cleaned['Base_Sales'] = 1000  # Example base sales
    price_elasticity = -1.5
    df_cleaned['Price_Change'] = (df_cleaned['Discount_Price'] - df_cleaned['Actual_Price']) / df_cleaned['Actual_Price']
    df_cleaned['Simulated_Sales_Volume'] = df_cleaned['Base_Sales'] * (1 + price_elasticity * df_cleaned['Price_Change'])

    return df_cleaned

df_cleaned = load_data()


# Create model pipeline
def create_model(df_cleaned):
    X = df_cleaned[['Main_Category', 'Sub_Category', 'Ratings', 'Discount_Price', 'Actual_Price']]
    y = df_cleaned['Simulated_Sales_Volume']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Main_Category', 'Sub_Category']),
            ('num', SimpleImputer(strategy='mean'), ['Ratings', 'Discount_Price', 'Actual_Price'])
        ])

    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('model', RandomForestRegressor(n_estimators=1, random_state=42))])

    model_pipeline.fit(X_train, y_train)

    return model_pipeline, X_test, y_test

model_pipeline, X_test, y_test = create_model(df_cleaned)

# Generate plot and return it as base64
def generate_plot(plt_func):
    img = io.BytesIO()
    plt_func()  # Call the plot function
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to predict and display results (residual plot)
@app.route('/predict', methods=['GET'])
def predict():
    y_pred = model_pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    def residual_plot():
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred - y_test, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot: Predicted vs Actual Sales Volume')
        plt.xlabel('Actual Sales Volume')
        plt.ylabel('Residuals (Predicted - Actual)')

    plot_url = generate_plot(residual_plot)

    return jsonify({'rmse': rmse, 'r2': r2, 'plot_url': plot_url})

# Route to generate the heatmap
@app.route('/heatmap', methods=['GET'])
def heatmap():
    def heatmap_plot():
        plt.figure(figsize=(10, 8))
        corr_matrix = df_cleaned[['Ratings', 'Discount_Price', 'Actual_Price', 'Simulated_Sales_Volume']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')

    plot_url = generate_plot(heatmap_plot)
    return jsonify({'plot_url': plot_url})

# Route to generate the box plot
@app.route('/boxplot', methods=['GET'])
def boxplot():
    def boxplot_plot():
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Main_Category', y='Simulated_Sales_Volume', data=df_cleaned)
        plt.title('Simulated Sales Volume by Main Category')
        plt.xlabel('Main Category')
        plt.ylabel('Simulated Sales Volume')
        plt.xticks(rotation=45)  # Rotate x-axis labels
        plt.tight_layout()
    plot_url = generate_plot(boxplot_plot)
    return jsonify({'plot_url': plot_url})

# Route to generate the scatter plot
@app.route('/scatterplot', methods=['GET'])
def scatterplot():
    def scatterplot_plot():
        plt.figure(figsize=(10, 6))
        plt.scatter(df_cleaned['Discount_Price'], df_cleaned['Simulated_Sales_Volume'], alpha=0.5)
        plt.title('Discount Price vs Simulated Sales Volume')
        plt.xlabel('Discount Price')
        plt.ylabel('Simulated Sales Volume')

    plot_url = generate_plot(scatterplot_plot)
    return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True)

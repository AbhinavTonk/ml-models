import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def predict_agile_effort(filePath, input_story_points, input_complexity):
    # 1- Read csv file
    df = pd.read_csv(filePath)

    # 2- Define Input Features and Output Target
    X = df[['Story_Points', 'Complexity']]
    y = df['Effort_Hours']

    # 3- Create Linear Regression Model and Train it
    model = LinearRegression()
    model.fit(X, y)

    # 4- Make Predictions on the training data only
    # Reason- To evaluate model performance, To evaluate variance in prediction from actual result
    df['Predicted_Effort'] = model.predict(X) #predict always returns a list/array (Numpy Array) of predictions.

    # 5 (a)- User Input into DataFrame
    user_input = pd.DataFrame({
        'Story_Points': [input_story_points],
        'Complexity': [input_complexity]
    })

    # (b) Predict effort for the new user input
    predicted_user_effort = model.predict(user_input)[0] #This is a NumPy array with only one element in this case

    # (c) Add predicted effort and placeholder for actual effort
    user_input['Effort_Hours'] = 10  # No actual value known (it should be ideally None, but for calculating variance we have randomly hardcoded to 10
    user_input['Predicted_Effort'] = predicted_user_effort

    # (d) Append user_input to main DataFrame
    df = pd.concat([df, user_input], ignore_index=True)

    # 6 (a) Print the results
    print("\nData with Predictions:")
    print(df)

    # (b) Get intercept and coefficients
    intercept = model.intercept_
    coefficients = model.coef_

    print(f"\n📈 Model Intercept (I₀): {intercept:.2f}")
    print(f"📊 Coefficients (weights): {coefficients}")

    print(f"\n🔮 Predicted Effort for input (Story Points = {input_story_points}, Complexity = {input_complexity}): {predicted_user_effort:.2f} hours")

    # 7- Save the updated DataFrame with predictions to a new CSV
    df.to_csv('output/linear_regression/agile_effort_predicted_effort_output.csv', index=False)
    print("\n✅ Predictions saved to 'agile_effort_predicted_effort_output.csv'")


    # 8 - Calculate R2 Variance for Model Performance
    r2 = r2_score(df['Effort_Hours'], df['Predicted_Effort'])
    print("\n📉 Model Evaluation Metrics:")
    print(f"🔹 R² Score: {r2:.2f}") # Should be in between 0.80-1.0
    # If it's not in acceptable range then a)Add more or better features b)Use Advanced Models (DecisionTreeRegressor, RandomForestRegressor etc..)

    # 9 - Create scatter plot using Matplotlib (Data Visulization)
    plt.figure(figsize=(8, 5))
    plt.scatter(df['Effort_Hours'], df['Predicted_Effort'], color='blue', alpha=0.7, label='Predicted vs Actual')

    # Add ideal line (y = x)
    plt.plot([df['Effort_Hours'].min(), df['Effort_Hours'].max()],
             [df['Effort_Hours'].min(), df['Effort_Hours'].max()],
             color='red', linestyle='--', label='Ideal Prediction')

    # Set labels and title with R² score
    plt.xlabel('Actual Effort Hours')
    plt.ylabel('Predicted Effort Hours')
    plt.title(f'Actual vs Predicted Effort (R² = {r2:.2f})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show plot
    plt.show()

if __name__=='__main__':
    predict_agile_effort('data/linear_regression/agile_effort.csv', 4, 8)
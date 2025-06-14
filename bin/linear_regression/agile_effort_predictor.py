import pandas as pd
from sklearn.linear_model import LinearRegression

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
    user_input['Effort_Hours'] = None  # No actual value known
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


if __name__=='__main__':
    predict_agile_effort('data/linear_regression/agile_effort.csv', 4, 8)
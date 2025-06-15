import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def epic_tshirt_size_prediction(filepath, input_user_stories, input_complexity, input_dependencies, input_unknowns, input_team_familiarity):
    # 1- Read Csv File
    df= pd.read_csv(filepath)

    # 2 (a)- LabelEncoders (As DecisionTree, RandomForest can only work with numbers, so we need LabelEncoders to convert the text of these columns into Numbers)
    le_complexity = LabelEncoder()
    le_dependencies = LabelEncoder()
    le_unknowns = LabelEncoder()
    le_team_familiarity = LabelEncoder()
    le_tshirt_size = LabelEncoder()

    df['complexity_encoded'] = le_complexity.fit_transform(df['complexity'])
    df['dependencies_encoded'] = le_dependencies.fit_transform(df['dependencies'])
    df['unknowns_encoded'] = le_unknowns.fit_transform(df['unknowns'])
    df['team_familiarity_encoded'] = le_team_familiarity.fit_transform(df['team_familiarity'])
    df['tshirt_size_encoded'] = le_tshirt_size.fit_transform(df['tshirt_size'])

    # (b) Prepare features (encoded features) and target
    X = df[['user_stories', 'complexity_encoded', 'dependencies_encoded', 'unknowns_encoded', 'team_familiarity_encoded']]
    y = df['tshirt_size_encoded']

    # (c) Train-Test Split (80% train, 20% test) - optional (rest of the code will change)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # 3- Train Decision tree/RandomForest
    # model = DecisionTreeClassifier(max_depth=3, random_state=42)  # Only One Line Change to switch from DecisionTree to RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 4- Make Predictions on the training data only
    # Reason- To evaluate model performance, To evaluate variance in prediction from actual result
    df['predicted_tshirt_size_encoded'] = model.predict(X)
    df['predicted_tshirt_size'] = le_tshirt_size.inverse_transform(df['predicted_tshirt_size_encoded'])

    # 5 (a) - User Input into Dataframe
    user_input = pd.DataFrame([{
        'user_stories': input_user_stories,
        'complexity_encoded': le_complexity.transform([input_complexity])[0],
        'dependencies_encoded': le_dependencies.transform([input_dependencies])[0],
        'unknowns_encoded': le_unknowns.transform([input_unknowns])[0],
        'team_familiarity_encoded': le_team_familiarity.transform([input_team_familiarity])[0]
    }])

    # (b) Predict effort for the new user input
    predicted_label_encoded = model.predict(user_input)[0]
    predicted_label = le_tshirt_size.inverse_transform([predicted_label_encoded])[0]

    # (c) Add all placeholders for user_input
    user_input['complexity'] = input_complexity
    user_input['dependencies'] = input_dependencies
    user_input['unknowns'] = input_unknowns
    user_input['team_familiarity'] = input_team_familiarity
    user_input['tshirt_size'] = 'S'
    user_input['tshirt_size_encoded'] = le_tshirt_size.transform(['S'])[0]
    user_input['predicted_tshirt_size'] = predicted_label
    user_input['predicted_tshirt_size_encoded'] = predicted_label_encoded

    # (d) Append user_row in csv
    df = pd.concat([df, user_input], ignore_index=True)

    # 6 - Print dataframe
    print(df)

    # 7 - Save result to CSV
    output_path = 'output/decision_tree/tshirt_size_prediction_output.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Prediction saved to: {output_path}")
    print(f"🎯 Predicted T-Shirt Size: {predicted_label}")

    # 8 - Model Evaluation (Classification Metrics)
    print("\n📊 Classification Report:")
    print(classification_report(df['tshirt_size_encoded'], df['predicted_tshirt_size_encoded'], target_names=le_tshirt_size.classes_))

    accuracy = accuracy_score(df['tshirt_size_encoded'], df['predicted_tshirt_size_encoded'])
    print(f"\n✅ Accuracy on training data: {accuracy:.2f}")

    print("\n🔁 Confusion Matrix:")
    print(confusion_matrix(df['tshirt_size_encoded'], df['predicted_tshirt_size_encoded']))

    # 9. Optional: Visualize the decision tree
    # plt.figure(figsize=(12, 6))
    # plot_tree(model, feature_names=X.columns, class_names=le_tshirt_size.classes_, filled=True)
    # plt.title("Decision Tree for Agile T-Shirt Sizing")
    # plt.show()

if __name__=='__main__':
    epic_tshirt_size_prediction('data/decision_tree/epic_tshirt_sizing_data.csv', 4, 'high','critical','very_high','low')

from flask import Flask, render_template, request
import os
import joblib
import pandas as pd


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_data = request.form.to_dict()

        print(f"input_data : {input_data}")

        selected_model = input_data['MODEL']
        input_data.pop('MODEL')

        print(f"selected_model : {selected_model}")

        # Convert relevant columns to numeric values
        numeric_columns = ['LATITUDE', 'LONGITUDE', 'YEAR']

        for col in numeric_columns:
            input_data[col] = float(input_data[col])

        # Convert Yes/No fields to 0 and 1
        yes_no_columns = ['AUTOMOBILE', 'CYCLIST',
                          'MOTORCYCLE', 'PEDESTRIAN', 'TRSN_CITY_VEH', 'TRUCK']
        for col in yes_no_columns:
            input_data[col] = 1 if input_data[col] == 'Yes' else 0

        # Map string values to numeric values for TRAFFCTL
        traffic_control_mapping = {
            "Traffic Signal": 1,
            "No Control": 2,
            "Stop Sign": 3,
            "Pedestrian Crossover": 4,
            "Traffic Controller": 5,
            "Yield Sign": 6,
            "School Guard": 7,
            "Police Control": 8,
            "Traffic Gate": 9,
            "Streetcar (Stop for)": 10
        }
        input_data['TRAFFCTL'] = traffic_control_mapping.get(
            input_data['TRAFFCTL'], 0)

        # Get the current script's directory
#        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#        print("Parent directory:", parent_directory)
        model_path = f'{selected_model}_best_estimator.pkl'
#        modelPKL_path = os.path.abspath(os.path.join(script_dir, 'models', model_path))
        modelPKL_path = os.path.abspath(os.path.join(script_dir, 'BackEnd/project_files', model_path))
        pkl_file = open(modelPKL_path, "rb")
        model = joblib.load(pkl_file)
        
        # Convert input data to a DataFrame
        input_df = pd.DataFrame.from_dict([input_data])

        # Make a prediction
        prediction = model.predict(input_df)
        if prediction == 1:
            prediction = "FATAL"
        else:
            prediction = "NON-FATAL"

        return render_template('result.html', prediction=prediction)

    return render_template('predictor.html')


if __name__ == '__main__':
    app.run(debug=True)

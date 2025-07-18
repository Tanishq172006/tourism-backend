from flask import Flask, request, jsonify
import joblib
import pandas as pd


model = joblib.load("tourism_model.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([{
            "Year": data["year"],
            "Month": data["month"],
            "State": data["state"],
            "Season": data["season"],
            "Temp": data["temp"],
            "Rainfall": data["rainfall"],
            "Holiday": data["holiday"]
        }])
        prediction = model.predict(input_df)[0]
        return jsonify({"predicted_tourists": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

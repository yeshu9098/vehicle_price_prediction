from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('notebook/vehicle_price_predictor.pkl')

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            features = [
                float(request.form["present_price"]),
                float(request.form["kms_driven"]),
                int(request.form['fuel_type']),
                int(request.form['seller_type']),
                int(request.form['transmission']),
                int(request.form['owner']),
                int(request.form['age'])
            ]

            input_data = np.array(features, dtype=float).reshape(1, -1)
            print(input_data)
            predicted_price = model.predict(input_data)[0]

            return render_template("index.html", predicted_price=round(predicted_price, 1))

        except Exception as e:
            return str(e) 

    return render_template("index.html", predicted_price=None)

if __name__ == "__main__":
    app.run(debug=True)

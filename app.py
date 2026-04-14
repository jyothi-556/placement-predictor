from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        attendance = float(request.form["attendance"])
        marks = float(request.form["marks"])
        projects = float(request.form["projects"])
        internships = float(request.form["internships"])
        communication = float(request.form["communication"])
        backlogs = float(request.form["backlogs"])
        cgpa = float(request.form["cgpa"])

        # Arrange in correct order (VERY IMPORTANT)
        features = np.array([[attendance, marks, projects, internships, communication, backlogs, cgpa]])

        prediction = model.predict(features)

        result = "Placed ✅" if prediction[0] == 1 else "Not Placed ❌"

        return render_template("index.html", prediction_text=result)

    except:
        return "Error: Please enter valid values"

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request
from pickle import load

# load model
with open("sna.pkl", "rb") as f:
    model = load(f)

# load scaler
with open("scaler.pkl", "rb") as f:
    scaler = load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        age = int(request.form.get("age"))
        estSal = float(request.form.get("estSal"))
        gender = request.form.get("gender")   # "female" or "male"

        # order must match training: [Age, EstimatedSalary, Gender_Female, Gender_Male]
        if gender == "female":
            d = [[age, estSal, 1, 0]]
        else:  # male
            d = [[age, estSal, 0, 1]]

        d_scaled = scaler.transform(d)
        res = model.predict(d_scaled)[0]
        result = "Yes" if res == 1 else "No"
        return render_template("home.html", msg=f"Will the User Purchase? : {result}")

    return render_template("home.html", msg=None)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)

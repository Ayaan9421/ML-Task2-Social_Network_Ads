from pickle import load

f = open("sna.pkl", "rb")
model = load(f)
f.close()

f = open("scaler.pkl", "rb")
scaler = load(f)
f.close()

age = int(input("Enter Age: "))
estSal = float(input("Enter Estimated Salary: "))
gender = int(input("Enter Gender: 1-Female, 2-Male: "))

if gender == 1:
        d = [[age, estSal, 1,0]]
elif gender == 2:
        d = [[age, estSal, 0,1]]

d_scaled = scaler.transform(d)

res = model.predict(d_scaled)
print(res)
print(model.predict_proba(d_scaled))

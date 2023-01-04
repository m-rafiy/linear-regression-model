import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

root = tk.Tk()
root.title("Linear Regression Results")

# Add a tkinter Entry widget to the window
input_entry = tk.Entry(root)
input_entry.pack()

def predict_output(predictions):
    # Retrieve the value entered by the user
    input_value = input_entry.get()
    arr = list(map(float, input_value.split(',')))
    if len(arr) != 5:
        print("Error: input must have 5 features (G1, G2, studytime, failures, absences)")
        return
    input_array = np.array(arr).reshape(1, -1)
    prediction = linear.predict(input_array)
    print("Prediction:", prediction[0])



predict_button = tk.Button(root, text ="Predict Output", command = lambda: predict_output(predictions))
predict_button.pack()
figure = plt.Figure(figsize=(5,4), dpi=100)
ax = figure.add_subplot(111)

ax.scatter(y_test, predictions, color = "b")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')

canvas = FigureCanvasTkAgg(figure, root)
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

root.mainloop()

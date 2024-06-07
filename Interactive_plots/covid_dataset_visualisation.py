import numpy as np
from sklearn.metrics import roc_curve
from ipywidgets import interact, FloatSlider
import matplotlib.pyplot as plt

# Convert the cases to a single list and create the corresponding labels
positive_cases = [(36.882026172983835, 93.20906687761264), (38.120446599600726, 92.63873563175518), (37.93377899507498, 94.94363554332269), (37.72713675348149, 94.28089367691892), (37.060837508246415, 91.54743479533664), (36.747039536578804, 91.7396033060679), (36.62891748979678, 92.53034835929269), (35.134877311993804, 95.8046832823551), (37.76638960717923, 94.37689493574526), (36.73467938495014, 95.1123306844595), (37.07747371284846, 92.66970031843329), (36.009601765888036, 91.92751262744555), (36.61514534036386, 97.3575591423193), (37.60118992439221, 94.6401503283753), (36.806336591296024, 92.85849475697891), (40.97538769761589, 96.95327807296742)]

negative_cases = [(37.20007860418361, 95.77380499571852), (37.48936899205287, 93.97838972486225), (36.511361060061795, 95.85666374106084), (37.475044208762796, 95.13303444476634), (36.92432139585115, 95.60494379547956), (36.94839057410322, 93.73135581263807), (37.20529925096918, 94.27451766802572), (37.07202178558044, 93.65507910444809), (37.380518862573496, 93.37370743591109), (37.221931616372714, 95.3548522845075), (37.166837163687134, 94.19643812758348), (36.8974208681171, 95.92556451105155), (37.156533850825454, 93.18540327123351), (36.57295213034914, 95.10389079159228), (35.72350509208296, 96.45818112435508), (37.32680929772018, 95.25796582151482), (37.432218099429754, 97.27880136908661), (38.27281716270062, 93.63037981811938), (37.022879258650725, 93.25840570163624), (36.906408074987084, 93.84230067047118), (37.189081259801085, 96.80165297390838), (36.55610712618494, 95.93132487946092), (36.826043925336926, 97.9765043875912), (37.07817448455199, 98.79177835206117), (36.84884862471233, 97.10890345386227), (38.475723517466456, 94.19364610605363), (38.289991031410516, 97.44489014076485), (36.14686490468749, 95.41654995615372), (36.745173909124176, 95.71273279434881), (36.78096284919441, 96.4131463363839), (36.373602319975035, 95.02100004144164), (37.38874517791596, 98.57174098781167), (36.193051076221025, 95.25382418540724), (36.89362985989302, 95.8039787268894)]

data = positive_cases + negative_cases
labels = [1]*len(positive_cases) + [0]*len(negative_cases)

# Use a combination of body temperature and oxygen level as the score
scores = [case[0] - case[1] for case in data]

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(labels, scores)

def plot_roc(threshold):
    # Find the threshold closest to the one selected
    closest_threshold = min(thresholds, key=lambda x:abs(x-threshold))
    index = list(thresholds).index(closest_threshold)

    # Create a figure
    plt.figure(figsize=(20, 6))

    # Create the first subplot
    plt.subplot(1, 2, 1)

    # Plotting positive cases
    positive_body_temp = [case[0] for case in positive_cases]
    positive_oxygen_level = [case[1] for case in positive_cases]
    plt.scatter(positive_body_temp, positive_oxygen_level, color='red', label='Positive')

    # Plotting negative cases
    negative_body_temp = [case[0] for case in negative_cases]
    negative_oxygen_level = [case[1] for case in negative_cases]
    plt.scatter(negative_body_temp, negative_oxygen_level, color='blue', label='Negative')

    # Threshold lines
    x = np.linspace(min(positive_body_temp + negative_body_temp), max(positive_body_temp + negative_body_temp), 100)
    y = x - closest_threshold
    plt.plot(x, y, color='gray', linestyle='--', label='Decision Boundary')

    # Plot settings
    plt.title('COVID Test: Body Temperature vs. Oxygen Level')
    plt.xlabel('Body Temperature (°C)')
    plt.ylabel('Oxygen Level (%)')
    plt.legend()
    plt.grid(True)

    # Create the second subplot
    plt.subplot(1, 2, 2)

    plt.plot(fpr, tpr, label='ROC curve')
    plt.scatter(fpr[index], tpr[index], color='red', label='Operating Point')  # plot the operating point
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Show the figure
    plt.show()

# Create the interactive plot
interact(plot_roc, threshold=FloatSlider(min=min(thresholds), max=max(thresholds), step=0.01, value=0.5));
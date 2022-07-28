"""importing"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_validate

from sklearn.linear_model import ElasticNet


"""Preprocessing"""

data = pd.read_excel("D://uni/project/try/result_shot.xlsx")  # >>>>>> Importing data from Excel file
data = pd.DataFrame(data, index=np.arange(71), columns=np.array(
    ["H", "A", "N", "T", "L", "CL", "M", "D", "PI", "VI", "P", "S", "AW", "RA", "SC", "OLS", "ALS", "RC", "V", "V_t",
     "LL", "Y", "NETA", "Wl"]))  # >>>>>> Defining and symbolizing all of columns
data = data.drop(["T", "L", "RA", "V", "SC", "RC", "OLS", "ALS", "NETA", "N", "AW"], axis=1)  # >>>>>> Dropping dispensable columns
x = data.drop("Y", axis=1)
y = data[:]["Y"]
y = np.array(y).reshape(71, 1)
y = pd.DataFrame(y, columns=["Y"])
y = pd.Series.ravel(y)  # >>>>>> Setting the target functions

Initial_explanatory_variables = ['P', 'S', 'CL', 'LL', 'H', 'M', 'PI', 'VI', 'D', 'V_t', 'A', 'Wl']  # Initial explanatory variables


"""hyper_parameters"""
alpha = []
r2 = []
for i in range(1,1000):  # >>>>>>  Alpha tuning
    reg = ElasticNet(alpha=i/1000)
    crvh_predict = cross_val_predict(reg, x, y, cv=6)  # >>>>>> 6_fold cross-validation is chosen for shotcrete
    R2_h = metrics.r2_score(y, crvh_predict)
    alpha.append(i/1000)
    r2.append(R2_h)
plt.plot(alpha,r2)  # >>>>>> Visualization of determining hyper_parameter(alpha)
plt.title("hyper_parameter_alpha")
plt.xlabel("alpha")
plt.ylabel("R2_square")
plt.show()
a = 0.008  # >>>>>>  Optimum value for alpha

l1_ratio = []
r22 = []
for i in range(1,1000):  # >>>>>>  lambda tuning
    reg = ElasticNet(alpha=a,l1_ratio=i/1000)
    crvhh_predict = cross_val_predict(reg, x, y, cv=6)  # >>>>>> 6_fold cross-validation is chosen for shotcrete
    R2_hh = metrics.r2_score(y, crvhh_predict)
    l1_ratio.append(i/1000)
    r22.append(R2_hh)
plt.plot(l1_ratio,r22)                                  # >>>>>> Visualization of determining hyper_parameter(lambda)
plt.title("hyper_parameter_l1_ratio")
plt.xlabel("l1_ratio")
plt.ylabel("R2_square")
plt.show()
l = 0.468  # >>>>>>  Optimum value for lambda



"""Initial model """

reg = ElasticNet(alpha=0.008,l1_ratio=0.468)                # Developing initial model in elastic net for shotcrete
crv_predict = cross_val_predict(reg, x, y, cv=6)            # 6_fold cross-validation is chosen for shotcrete
R2_initial = metrics.r2_score(y, crv_predict)               # R2_squared for initial model
mse_initial = metrics.mean_squared_error(y, crv_predict)    # Mean squared error for initial model
mae_initial = metrics.mean_absolute_error(y, crv_predict)   # Mean absolute error for initial model
# result
print("R2_initial:", R2_initial)
print("mse_initial:", mse_initial)
print("mae_initial:", mae_initial)




"""Model_reduction"""

xx = x
t = 0
omitted_variable = ["NON"]
r = [R2_initial]
useless = []   # >>>>>> List of useless variables
for i in Initial_explanatory_variables:    # >>>>>> Each explanatory variable is crossed out and then added to the model
    xx = x
    if t <= 11:
        reg = ElasticNet(alpha=0.008,l1_ratio=0.468)
        xx = xx.drop(i, axis=1)
        crvm_predict = cross_val_predict(reg, xx, y, cv=6)

        R2 = metrics.r2_score(y, crvm_predict)
        omitted_variable.append(i)
        r.append(R2)
        decrease_in_accuracy = R2_initial - R2   # >>>>>> Changing in accuracy (R-squared)
        if decrease_in_accuracy <= 0.01:    # Threshold, # Adding the useless variable to the List
            useless.append(i)

        print(i, ">>>>>>", R2, "difference", decrease_in_accuracy)
        t += 1



# print(non_important_parameter)


plt.plot(omitted_variable, r)
plt.title("model reduction")
plt.xlabel("omitted_variable")
plt.ylabel("R2_square")
plt.show()


"""final_model(Elastic net)"""

x_model = x.drop(useless, axis=1)
print("Variables of model:", list(x_model.columns))

reg = ElasticNet(alpha=0.008,l1_ratio=0.468)                   # Developing final model in elastic net for shotcrete
y_predict = cross_val_predict(reg, x_model, y,cv=6)            # 6_fold cross-validation is chosen for shotcrete
R2_final = metrics.r2_score(y, y_predict)                      # R2_squared for final model
mse_final = metrics.mean_squared_error(y, y_predict)           # Mean squared error for final model
mae_final = metrics.mean_absolute_error(y, y_predict)          # Mean absolute error for final model
# result
print("final_R2", R2_final)
print("final_mse", mse_final)
print("final_mae", mae_final)

# Quality of prediction, plot
plt.plot(y, y, marker="+", c="black")
plt.scatter(y, y_predict, c="blue", marker="*")
plt.title("Elastic net")
plt.xlabel("Actual values")
plt.ylabel("predict values")
plt.show()

R = y - y_predict  # Residual
plt.hist(R)
plt.title("histogram for Elastic net")
plt.show()

# diagnostics
"""
import statsmodels.api as sm

sm.qqplot(R, line='s')

plt.show()

"""


"""The Most effective variables"""

xxx = x_model
final_variables = list(x_model.columns)
l = len(final_variables)
t = 0
r = []
d = dict()
for i in final_variables:  # >>>>>> Each final variable is crossed out and then added to the model
    xxx = x_model
    if t <= (l - 1):
        reg = ElasticNet(alpha=0.008,l1_ratio=0.468)
        xxx = x_model.drop(i, axis=1)
        crvmm_predict = cross_val_predict(reg, xxx, y, cv=6)
        R2 = metrics.r2_score(y, crvmm_predict)
        r.append(R2)
        decrease_in_accuracy = R2_final - R2
        d[i] = decrease_in_accuracy
        print(i, ">>>>>>", R2, "difference", decrease_in_accuracy)
        t += 1

print(d)



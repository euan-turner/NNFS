import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x**2

x = np.array(np.arange(0,5,0.001))
y = f(x)


plt.plot(x,y)

colours = ['k','g','r','b','c']

def tangent_line(x, approx_deriv):
    return approx_deriv*x + b

for i in range(5):
    dx = 0.0001
    x1 = i
    x2 = x1+dx

    y1 = f(x1)
    y2 = f(x2)

    print((x1,y1),(x2,y2))
    approx_deriv = (y2-y1)/(x2-x1)
    b = y2-(approx_deriv*x2)

    tangentx = [x1-0.9,x1,x1+0.9]
    plt.scatter(x1,y1,c=colours[i])
    plt.plot([point for point in tangentx],[tangent_line(point,approx_deriv) for point in tangentx],c=colours[i])

    print("Approximate derivative for f(x)", f"where x = {x1} is {approx_deriv}")

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import math

data = [
  (0.50, 0), (0.75, 0), (1.00, 0), (1.25, 0),
  (1.50, 0), (1.75, 0), (1.75, 1), (2.00, 0),
  (2.25, 1), (2.50, 0), (2.75, 1), (3.00, 0),
  (3.25, 1), (3.50, 0), (4.00, 1), (4.25, 1),
  (4.50, 1), (4.75, 1), (5.00, 1), (5.50, 1)
]

beta = [0.1, 0.1]
learning_rate = 0.01

def sigmoid(x):
  return 1 / (1 + math.exp(-(beta[0] + beta[1]*x)))

def steepest_descent():
  global beta0, beta1
  for i in range(0, 10000):
    total_error = 0
    step = [0, 0]
    for case in data:
      error = case[1] - sigmoid(case[0])
      step[0] += error
      step[1] += error * case[0]
      total_error += error

    beta[0] = beta[0] + learning_rate * step[0]
    beta[1] = beta[1] + learning_rate * step[1]
    print 'Average error:', total_error / len(data)

steepest_descent()

print 'Beta 0:', beta[0], 
print 'Beta 1:', beta[1]

# Show charts.
x = np.arange(0, 6, 0.1);
y = []
for x_ in x:
  y.append(sigmoid(x_))

data_x = [d[0] for d in data]
data_y = [d[1] for d in data]
plt.plot(x, y, '-', data_x, data_y, 'ro')
plt.show()

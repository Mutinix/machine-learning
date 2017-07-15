from random import *

# Chapter 1: Real-valued Circuits

# Base Case: Single Gate in the Circuit
def forwardMultiplyGate(x, y):
  return x*y

forwardMultiplyGate(-2, 3)

# Goal: Tweak the input slightly to increase the output

# Strategy #1: Random Local Search

x, y = -2, 3

# Change x, y randomly by small amounts and keep track fo what works
tweak_amount = 0.01
best_out = float("-inf")
best_x, best_y = x, y

for k in range(100):
  x_try = x + tweak_amount * (random() * 2 - 1)
  y_try = y + tweak_amount * (random() * 2 - 1)
  out = forwardMultiplyGate(x_try, y_try)
  if out > best_out:
    best_out = out
    best_x = x_try
    best_y = y_try

# Strategy #2: Numerical Gradient

x, y = -2, 3
out = forwardMultiplyGate(x, y)
h = 0.0001

# Compute derivative with respect to x
xph = x + h
out2 = forwardMultiplyGate(xph, y) # -5.9997
x_derivative = (out2 - out) / h # +3.0

# Compute derivative with respect to y
yph = y + h
out3 = forwardMultiplyGate(x, yph) # -6.0002
y_derivative = (out3 - out) / h # -2.0

# Effect of derivative in action
step_size = 0.01
out = forwardMultiplyGate(x, y)
x = x + step_size * x_derivative # x becomes -1.97
y = y + step_size * y_derivative # y becomes 2.98
out_new = forwardMultiplyGate(x, y) # -5.87

# Strategy #3: Analytic Gradient

x, y = -2, 3
out = forwardMultiplyGate(x, y) # Before: -6
x_gradient = y # Maths!
y_gradient = x

step_size = 0.01
x += step_size * x_gradient # -1.97
y += step_size * y_gradient # 2.98
out_new = forwardMultiplyGate(x, y)
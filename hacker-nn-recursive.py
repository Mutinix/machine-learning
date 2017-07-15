# Recursive Case: Circuits with Multiple Gates

def forwardMultiplyGate(a, b):
  return a * b

def forwardAddGate(a, b):
  return a + b

def forwardCircuit(x, y, z):
  q = forwardAddGate(x, y)
  f = forwardMultiplyGate(q, z)
  return f

x = -2
y = 5
z = -4
q = forwardAddGate(x, y) # 3
f = forwardMultiplyGate(q, z) # -12

# Derivative of the multiply gate with respect to its inputs
derivative_f_wrt_z = q # 3
derivative_f_wrt_q = z # -4

# Derivative of the add gate with respect to its inputs
derivative_q_wrt_x = 1.0
derivative_q_wrt_y = 1.0

# Chain rule!
derivative_f_wrt_x = derivative_f_wrt_q * derivative_q_wrt_x # -4
derivative_f_wrt_y = derivative_f_wrt_q * derivative_q_wrt_y # -4

# Final gradient
gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]

step_size = 0.01
x = x + step_size * derivative_f_wrt_x # -2.04
y = y + step_size * derivative_f_wrt_y # 4.96
z = z + step_size * derivative_f_wrt_z # -3.97

# Better outputs
q = forwardAddGate(x, y) # q becomes 2.92
f = forwardMultiplyGate(q, z) # Output is -11.59, up from -12

# Sanity check
x = -2
y = 5
z = -4

# Calculate numerical gradient (to compare with prev result)
h = 0.0001
x_derivative = (forwardCircuit(x+h, y, z) - forwardCircuit(x, y, z)) / h # -4
y_derivative = (forwardCircuit(x, y+h, z) - forwardCircuit(x, y, z)) / h # -4
z_derivative = (forwardCircuit(x, y, z+h) - forwardCircuit(x, y, z)) / h # 3

# Matches previous result, yay!
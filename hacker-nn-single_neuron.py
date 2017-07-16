import math

# Example: Single Neuron

# Every unit corresponds to a wire in the diagrams
class Unit(object):
  def __init__(self, value, grad):
    # Value computed in the forward pass
    self.value = value
    # Derivative of the output w.r.t. this unit, computed in the backward pass
    self.grad = grad

class  multiplyGate(object):
  def forward(self, u0, u1):
    self.u0 = u0
    self.u1 = u1
    # Output unit utop
    self.utop = Unit(u0.value * u1.value, 0.0)
    return self.utop

  def backward(self):
    # Take the gradient in the output unit and chain it with the
    # local gradients, which we derived for the multiply gate before
    # then write those gradients to those Units.
    self.u0.grad += self.u1.value * self.utop.grad
    self.u1.grad += self.u0.value * self.utop.grad

class addGate(object):
  def forward(self, u0, u1):
    self.u0 = u0
    self.u1 = u1
    self.utop = Unit(u0.value + u1.value, 0.0)
    return self.utop

  def backward(self):
    self.u0.grad += 1 * self.utop.grad
    self.u1.grad += 1 * self.utop.grad

class sigmoidGate(object):
  def sig(self, x):
    return 1 / (1 + math.exp(-x))

  def forward(self, u0):
    self.u0 = u0
    self.utop = Unit(self.sig(self.u0.value), 0.0)
    return self.utop

  def backward(self):
    s = self.sig(self.u0.value)
    self.u0.grad += (s * (1 - s)) * self.utop.grad


# Examples

# Input units
a = Unit(1.0, 0.0)
b = Unit(2.0, 0.0)
c = Unit(-3.0, 0.0)
x = Unit(-1.0, 0.0)
y = Unit(3.0, 0.0)

# Gates
mulg0 = multiplyGate()
mulg1 = multiplyGate()
addg0 = addGate()
addg1 = addGate()
sg0 = sigmoidGate()

s = None
# Forward pass
# Note: Removed the forwardNeuron function defined by Andrej
ax = mulg0.forward(a, x) # a*x = -1
by = mulg1.forward(b, y) # b*y = 6
axpby = addg0.forward(ax, by) # a*x + b*y = 5
axpbypc = addg1.forward(axpby, c) # a*x + b*y + c = 2
s = sg0.forward(axpbypc)

s.grad = 1.0
sg0.backward()      # Writes gradient into axpbypc
addg1.backward()  # Writes gradients into axpby and c
addg0.backward()  # Writes gradients into ax and by
mulg1.backward()  # Writes gradients into b and y
mulg0.backward()  # Writes gradients into a and x

step_size = 0.01
a.value += step_size * a.grad   # a.grad is -0.105
b.value += step_size * b.grad   # b.grad is 0.315
c.value += step_size * c.grad   # c.grad is 0.105
x.value += step_size * x.grad   # x.grad is 0.105
y.value += step_size * y.grad   # y.grad is 0.210

# Forward pass
ax = mulg0.forward(a, x) # a*x = -1
by = mulg1.forward(b, y) # b*y = 6
axpby = addg0.forward(ax, by) # a*x + b*y = 5
axpbypc = addg1.forward(axpby, c) # a*x + b*y + c = 2
s = sg0.forward(axpbypc)

# Check numerical gradient
def forwardCircuitFast(a,b,c,x,y):
  return 1/(1+math.exp(-(a*x+b*y+c)))

a = 1
b = 2
c = -3
x = -1
y = 3
h = 0.0001

a_grad = (forwardCircuitFast(a+h,b,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h
b_grad = (forwardCircuitFast(a,b+h,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h
c_grad = (forwardCircuitFast(a,b,c+h,x,y) - forwardCircuitFast(a,b,c,x,y))/h
x_grad = (forwardCircuitFast(a,b,c,x+h,y) - forwardCircuitFast(a,b,c,x,y))/h
y_grad = (forwardCircuitFast(a,b,c,x,y+h) - forwardCircuitFast(a,b,c,x,y))/h

# Same values, yay!
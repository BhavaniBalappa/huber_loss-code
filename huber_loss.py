def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    if abs(error) <= delta:
        return 0.5 * error**2
    else:
        return delta * (abs(error) - 0.5 * delta)

# Feed-forward NN
x = 4
w = 2
b = 1
y_true = 12

y_pred = w*x + b
loss = huber_loss(y_true, y_pred)

print("Prediction:", y_pred)
print("Huber Loss:", loss)

_______________________________________________________________________________________________________________________________________________________________________________________________________________________
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    if abs(error) <= delta:
        return 0.5 * error**2
    else:
        return abs(error) - 0.5

x = [1,2,3,4,5,6,7,8,9,10]
y_true = [3,6,7,9,12,14,15,18,19,25]

w, b = 2, 1
losses = []

for xi, yi in zip(x, y_true):
    y_pred = w*xi + b
    losses.append(huber_loss(yi, y_pred))

print("Huber losses:", losses)
print("Mean Huber Loss:", sum(losses)/len(losses))

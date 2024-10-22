"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# TODO: Implement for Task 2.5.

class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        in_size = 2
        out_size = 1

        self.layer1 = Linear(in_size, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, out_size)

    def forward(self, x):
        """
        Perform a forward pass through the network:
        x -> layer1 -> relu -> layer2 -> relu -> layer3 -> sigmoid
        """
        middle = self.layer1.forward(x).relu()

        end = self.layer2.forward(middle).relu()

        return self.layer3.forward(end).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.weights = self.add_parameter("weights", RParam(in_size, out_size).value)
        self.bias = self.add_parameter("bias", RParam(out_size).value)

        self.weights.value.requires_grad_(True)
        self.bias.value.requires_grad_(True)

    def forward(self, inputs: minitorch.Tensor) -> minitorch.Tensor:
        if len(inputs.shape) == 1:
            inputs = inputs.view(1, -1)


        batch_size = inputs.shape[0]
        in_size = inputs.shape[1]
        bias = self.bias.value
        out_size = self.out_size

        weights = self.weights
        weights_t = self.weights.value.permute(1, 0)

        inputs_reshaped = inputs.view(batch_size, 1, in_size)

        output = inputs_reshaped * weights_t
        output = output.sum(2)
        output = output.view(batch_size, out_size)


        return output + bias







def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)

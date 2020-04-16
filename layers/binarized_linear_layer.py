import torch

from ops.binarized_linear_op import binary_linear


class BinarizedLinear(torch.nn.Linear):

    def __init__(self, in_features, out_features, bias=True, mode="determistic"):
        super().__init__(in_features, out_features, bias)
        self.mode = mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # weight clamp [-1, 1]
        self.weight_clipping()

        if self.bias is not None:
            return binary_linear(input, self.weight, self.bias)
        return binary_linear(input, self.weight)

    def reset_parameters(self):
        # xavier initialization
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def weight_clipping(self):
        with torch.no_grad():
            for key in self.state_dict():
                if "weight" in key:
                    value = self.state_dict().get(key)
                    value.clamp_(-1, 1)


if __name__ == "__main__":
    import os
    import torch.optim as optim
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = "stochastic"
    model = torch.nn.Sequential(BinarizedLinear(10, 32, bias=False, mode=mode),
                                BinarizedLinear(32, 32, bias=False, mode=mode),
                                BinarizedLinear(32, 2, bias=False, mode=mode)
                                )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    data = torch.rand((1, 10)).to(device)
    target = torch.tensor([1]).to(device)

    output = model(data)
    output = output.view(1, -1)
    print(f"shape of output : {output.shape}")
    print(f"shape of target : {target.shape}")
    print(f"output : {output}")
    print(f"target : {target}")
    loss = criterion(output, target)
    print(loss)
    loss.backward()
    optimizer.step()

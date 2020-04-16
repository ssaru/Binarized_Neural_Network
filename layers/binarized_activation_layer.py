import torch

from ops.binarized_activation_op import binary_activation


class BinActivation(torch.nn.modules.Module):

    def __init__(self, mode="stochastic"):
        super(BinActivation, self).__init__()
        self.mode = mode

    def forward(self, input):
        return binary_activation(input, self.mode)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bin_activation = BinActivation(mode="stochastic")
    data = torch.randn((3, 3)).to(device)
    data.requires_grad = True
    target = torch.tensor([10]).to(device)
    print(f"data : {data}, target : {target}")

    activations = bin_activation(data)
    print(f"activations : {activations}")

    output = torch.sum(activations)
    loss = output - target
    print(f"loss : {loss}, output : {output}")

    loss.backward()
    print(f"loss grad : {loss.grad}")
    print(f"activations grad : {activations.grad}")
    print(f"data grad : {data.grad}")

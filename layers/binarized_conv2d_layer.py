
import torch

from ops.binarized_conv2d_op import binary_conv


class BinarizedConv2d(torch.nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 mode="stochastic"):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         groups,
                         bias,
                         padding_mode)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.mode = mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # weight clamp [-1, 1]
        self.weight_clipping()

        return binary_conv(input,
                           self.weight,
                           self.bias,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.groups,
                           self.mode)

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

    model = torch.nn.Sequential(
        BinarizedConv2d(in_channels=1,
                        out_channels=10,
                        kernel_size=3,
                        bias=True,
                        mode="stochastic"),

        BinarizedConv2d(in_channels=10,
                        out_channels=10,
                        kernel_size=3,
                        bias=True,
                        mode="stochastic"),

        BinarizedConv2d(in_channels=10,
                        out_channels=10,
                        kernel_size=3,
                        bias=True,
                        mode="stochastic"),

        BinarizedConv2d(in_channels=10,
                        out_channels=2,
                        kernel_size=4,
                        bias=True,
                        mode="stochastic"))

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    optimizer.zero_grad()
    data = torch.rand((1, 1, 10, 10)).to(device)
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

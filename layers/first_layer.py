import torch
import numpy as np


class FirstLinear(torch.nn.Linear):

    def __init__(self, in_features=8, out_features=1, bias=None):
        super(FirstLinear, self).__init__(8, 1, None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input data type should be should be 8-bit fixed points

        with torch.no_grad():
            b, n = input.shape
            out = torch.zeros((b, n))

            device = input.device
            input_np = input.cpu().data.numpy().astype(np.uint8)
            bits_unpack = np.unpackbits(input_np, axis=1)
            reshape_bits = bits_unpack.reshape((b, n, -1))
            input = torch.from_numpy(reshape_bits).float().to(device)
            factor = torch.tensor([2**i for i in range(0, 8)])
            input = torch.mul(input, factor)

        out = torch.nn.functional.linear(input,
                                         self.weight,
                                         None)

        out = out.view((b, n))

        return out


if __name__ == "__main__":
    dummy_input = torch.tensor([[0,   0, 238,   8,   3, 255, 251,   0, 236,  12],
                                [16, 254,   0, 253,  20, 235,   5,   7,   0, 241],
                                [240,   0,   2, 253,  11,   0, 252, 236,  13,  16],
                                [0, 247,  13, 251, 246,   0,   1,  17,   1,  11]])

    dummy_input = dummy_input.byte()
    dummy_target = torch.zeros((4)).long()
    criterion = torch.nn.CrossEntropyLoss()

    test_layer = FirstLinear()
    linear_layer = torch.nn.Linear(10, 1)
    out = test_layer(dummy_input)
    out = linear_layer(out)

    loss = criterion(out, dummy_target)
    loss.backward()

import torch


class binary_activation_op(torch.autograd.Function):
    """
    Refer https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(ctx, input, mode="stochastic"):

        ctx.save_for_backward(input)

        with torch.no_grad():
            if mode == "determistic":
                output = input.sign()
                output[output == 0] = 1.
            elif mode == "stochastic":
                p = torch.sigmoid(input)
                uniform_matrix = torch.empty(p.shape).uniform_(0, 1)
                uniform_matrix = uniform_matrix.to(input.device)
                output = (p >= uniform_matrix).type(torch.float32)
                output[output == 0] = -1.
            else:
                raise RuntimeError(f"{mode} not supported")

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zeros = torch.zeros(grad_input.shape).to(input.device)
        ones = torch.ones(grad_input.shape).to(input.device)
        threshold = torch.abs(grad_input)
        grad_input = torch.where(threshold <= 1, ones, zeros)

        return grad_input, None


binary_activation = binary_activation_op.apply

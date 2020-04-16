import torch


class binary_linear_op(torch.autograd.Function):
    """
    Refer https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None, mode="determistic"):

        # Binarization
        with torch.no_grad():
            if mode == "determistic":
                # Deterministic method
                bin_weight = weight.sign()
                bin_weight[bin_weight == 0] = 1
            elif mode == "stochastic":
                # Stochastic method
                p = torch.sigmoid(weight)
                uniform_matrix = torch.empty(p.shape).uniform_(0, 1)
                bin_weight = (p >= uniform_matrix).type(torch.float32)
                bin_weight[bin_weight == 0] = -1.
            else:
                raise RuntimeError(f"{mode} not supported")

        # Save input, binarized weight, bias in context object
        ctx.save_for_backward(input, bin_weight, bias)

        with torch.no_grad():
            output = input.mm(bin_weight.t())

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, bin_weight, bias = ctx.saved_variables

        grad_input = grad_weight = grad_bias = None

        with torch.no_grad():
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(bin_weight)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


binary_linear = binary_linear_op.apply

import torch
import torch.nn.functional as F


class binary_conv_op(torch.autograd.Function):
    """
    Refer https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(ctx,
                input,
                weight,
                bias=None,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                mode="determistic"):

        # Binarization
        with torch.no_grad():
            if mode == "determistic":
                # Deterministic method
                bin_weight = weight.sign()
                bin_weight[bin_weight == 0] = 1.
            elif mode == "stochastic":
                # Stochastic method
                p = torch.sigmoid(weight)
                uniform_matrix = torch.empty(p.shape).uniform_(0, 1)
                uniform_matrix = uniform_matrix.to(weight.device)
                bin_weight = (p >= uniform_matrix).type(torch.float32)
                bin_weight[bin_weight == 0] = -1.
            else:
                raise RuntimeError(f"{mode} not supported")

        with torch.no_grad():
            output = F.conv2d(input, bin_weight, bias, stride,
                              padding, dilation, groups)

        # Save input, binarized weight, bias in context object
        ctx.save_for_backward(input, bin_weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, bin_weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None

        with torch.no_grad():
            if ctx.needs_input_grad[0]:
                grad_input = torch.nn.grad.conv2d_input(input.shape,
                                                        bin_weight,
                                                        grad_output,
                                                        stride,
                                                        padding,
                                                        dilation,
                                                        groups)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.nn.grad.conv2d_weight(input,
                                                          bin_weight.shape,
                                                          grad_output,
                                                          stride,
                                                          padding,
                                                          dilation,
                                                          groups)

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


binary_conv = binary_conv_op.apply

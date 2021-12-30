import torch.nn.functional as F


def process_image(model, inp, input_size, input_channels):
    # divide to patches
    h, w = input_size
    input_b = inp.shape[0]

    def get_pad_len(l, patch_size):  # TODO: What if odd padding?
        pad = l % patch_size
        if pad:
            return (patch_size - pad) // 2

        return 0

    pad_axis2 = get_pad_len(inp.shape[2], h)
    pad_axis3 = get_pad_len(inp.shape[3], w)
    inp = F.pad(input=inp, pad=(pad_axis3, pad_axis3, pad_axis2, pad_axis2))

    # based on https://discuss.pytorch.org/t/creating-nonoverlapping-patches-from-3d-data-and-reshape-them-back-to-the-image/51210/6

    if input_channels == 5:
        input_c, output_c = 5, 5
    elif input_channels == 1:
        input_c, output_c = 1, 1
    else:
        input_c, output_c = 4, 1

    input_patches = inp.unfold(2, h, h).unfold(3, w, w)  # to patches
    unfold_shape = list(input_patches.shape)

    input_patches = input_patches.permute(0, 3, 2, 1, 4, 5).contiguous().view(-1, input_c, h, w)  # reshape for model
    pred_patches = model.forward(input_patches.to(model.device))  # inference

    unfold_shape[1] = unfold_shape[3]
    pred_unfold_shape = unfold_shape.copy()
    unfold_shape[3] = input_c
    pred_unfold_shape[3] = output_c

    # Reshape back
    # patches_permuted = input_patches.view(unfold_shape)
    pred_orig = pred_patches.view(pred_unfold_shape)
    output_h = unfold_shape[2] * unfold_shape[4]
    output_w = unfold_shape[1] * unfold_shape[5]

    # change between channel 1 and 3 --> change to 3,4,2,5 for unfolding

    # patches_orig = patches_permuted.permute(0, 3, 2, 4, 1, 5).contiguous()

    pred_orig = pred_orig.permute(0, 3, 2, 4, 1, 5).contiguous()

    # input_orig = patches_orig.view(input_b, input_c, output_h, output_w).detach().cpu().numpy()

    pred = pred_orig.view(input_b, output_c, output_h, output_w)

    # Check for equality
    # is_equal = input_orig == inp.cpu().numpy()
    # is_equal = is_equal.all() if is_equal is not False else is_equal
    # assert (is_equal,
    #         'error in division to patches in inference')

    s_axis2 = slice(pad_axis2, -pad_axis2) if pad_axis2 else slice(None)
    s_axis3 = slice(pad_axis3, -pad_axis3) if pad_axis3 else slice(None)

    return pred[:, :, s_axis2, s_axis3]

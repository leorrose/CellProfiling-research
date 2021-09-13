


from visuals.visualize import show_input_and_target

def process_image(model, input, input_size, input_channels):
    # based on https://discuss.pytorch.org/t/creating-nonoverlapping-patches-from-3d-data-and-reshape-them-back-to-the-image/51210/6

    # divide to patches
    h, w = input_size, input_size

    if input_channels == 5:
        input_c, output_c = 5,5
    elif input_channels == 1:
        input_c, output_c = 1, 1
    else:
        input_c, output_c = 4, 1

    input_patches = input.unfold(2, h, w).unfold(3, h, w)  # to patches
    unfold_shape = list(input_patches.shape)
    input_patches = input_patches.permute(0, 3, 2, 1, 4, 5).contiguous().view(-1, input_c, h, w)  # reshape for model

    pred_patches = model.forward(input_patches.to(model.device))  # inference
    # show_input_and_target(input_patches[0,:,:,:].cpu().detach().numpy().squeeze(),pred=pred_patches[0,:,:,:].cpu().detach().numpy().squeeze())
    unfold_shape[1] = unfold_shape[3]

    pred_unfold_shape = unfold_shape.copy()
    # pred_unfold_shape[1] = unfold_shape[3]
    unfold_shape[3] = input_c
    pred_unfold_shape[3] = output_c

    # pred_unfold_shape[1] = output_c  # mapped from 4 to 1

    # Reshape back
    patches_permuted = input_patches.view(unfold_shape)
    pred_orig = pred_patches.view(pred_unfold_shape)
    output_h = unfold_shape[2] * unfold_shape[4]
    output_w = unfold_shape[1] * unfold_shape[5]

    # change between channel 1 and 3 --> change to 3,4,2,5 for unfolding
    # patches_permuted = patches_permuted.permute(0, 3, 2, 1, 4, 5).contiguous()
    # patches_orig = patches_permuted.permute(0, 1, 2, 4, 3, 5).contiguous()
    patches_orig = patches_permuted.permute(0, 3, 2, 4, 1, 5).contiguous()
    # if output_c > 1:
    pred_orig = pred_orig.permute(0, 3, 2, 4, 1, 5).contiguous()
    # else:
    #     pred_orig = pred_orig.permute(0, 1, 3, 4, 2, 5).contiguous()
    input_orig = patches_orig.view(1, input_c, output_h, output_w).numpy().squeeze()
    # pred_orig = pred_orig.permute(0, 1, 3, 4, 2, 5).contiguous() - important!!
    # pred_orig = pred_orig.permute(0, 1, 3, 4, 2, 5).contiguous()

    pred = pred_orig.view(1, output_c, output_h, output_w).cpu().detach().numpy()[0,:,:,:]

    # new_pred = pred_orig.permute(0, 1, 2, 3, 4, 5).contiguous()
    # new_pred = new_pred.view(1, 1, output_h, output_w).cpu().detach().numpy().squeeze()
    # show_input_and_target(input[0, :, :, :].cpu().detach().numpy().squeeze(), new_pred)

    # Check for equality
    assert ((input_orig == input[:, :output_h, :output_w].cpu().numpy().squeeze()).all(),'error in division to patches in inference')

    return pred


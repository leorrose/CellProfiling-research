import os
from math import floor, ceil

from matplotlib import pyplot as plt

from data_layer.channels import Channels

CMAP = 'gray'  # 'viridis'


def show_input_and_target(input, target=None, pred=None, title='', save_dir=None, target_channel=None):
    input_c = input.shape[0]
    target_c = 0 if target is None else target.shape[0]
    pred_c = 0 if pred is None else pred.shape[0]
    num_images = input_c + target_c + pred_c
    images_in_row = 3 if num_images > 6 else 2

    num_rows, num_cols = ceil(num_images / images_in_row), images_in_row

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols*2*1.2, num_rows*2*1.3))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    chan_name_idx = 0
    for c in range(input_c):
        ax[floor(c / num_cols), c % num_cols].imshow(input[c, :, :], cmap=CMAP)
        chan = list(Channels)[chan_name_idx]
        if chan == target_channel:
            chan_name_idx += 1
            chan = list(Channels)[chan_name_idx]

        ax[floor(c / num_cols), c % num_cols].set_title(chan.name)
        chan_name_idx += 1

    if target is not None:
        for c in range(target_c):
            curr_fig = input_c + c
            ax[floor(curr_fig / num_cols), curr_fig % num_cols].imshow(target[c, :, :], cmap=CMAP)
            ax[floor(curr_fig / num_cols), curr_fig % num_cols].set_title('Target ' + target_channel.name)
            # pos1 =ax[1,input_c % num_cols].imshow(target[:, :])
            # ax[1, input_c % num_cols].set_title('Target')
        # fig.colorbar(pos1,ax=ax[1, num_channels % num_cols])
    if pred is not None:
        for c in range(pred_c):
            curr_fig = input_c + target_c + c
            ax[floor(curr_fig / num_cols), curr_fig % num_cols].imshow(pred[c, :, :], cmap=CMAP)
            ax[floor(curr_fig / num_cols), curr_fig % num_cols].set_title('Prediction ' + target_channel.name)
    else:
        curr_fig = input_c + target_c
        ax[floor(curr_fig / num_cols), curr_fig % num_cols].axis('off')

        # pos2 = ax[1,(input_c+1) % num_cols].imshow(pred[:, :])
        # ax[ 1, (input_c+1) % num_cols].set_title('Prediction')
        # fig.colorbar(pos2, ax=ax[1, (num_channels+1) % num_cols])

    fig.suptitle(title)
    plt.show()

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, title + '.jpg'))

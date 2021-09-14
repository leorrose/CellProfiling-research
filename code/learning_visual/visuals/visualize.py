import os
from math import floor, ceil

from matplotlib import pyplot as plt


def show_input_and_target(input, target=None, pred=None, title='', save_dir=None):
    input_c = input.shape[0]
    target_c = 0 if target is None else target.shape[0]
    pred_c = 0 if pred is None else pred.shape[0]
    num_images = input_c + target_c + pred_c
    images_in_row = 3 if num_images > 6 else 2

    num_rows, num_cols = ceil(num_images / images_in_row), images_in_row

    fig, ax = plt.subplots(num_rows, num_cols)
    for c in range(input_c):
        ax[floor(c / num_cols), c % num_cols].imshow(input[c, :, :])
        ax[floor(c / num_cols), c % num_cols].set_title('Input channel ' + str(c))

    if target is not None:
        for c in range(target_c):
            curr_fig = input_c + c
            ax[floor(curr_fig / num_cols), curr_fig % num_cols].imshow(target[c, :, :])
            ax[floor(curr_fig / num_cols), curr_fig % num_cols].set_title('Target' + str(c))
            # pos1 =ax[1,input_c % num_cols].imshow(target[:, :])
            # ax[1, input_c % num_cols].set_title('Target')
        # fig.colorbar(pos1,ax=ax[1, num_channels % num_cols])
    if pred is not None:
        for c in range(pred_c):
            curr_fig = input_c + target_c + c
            ax[floor(curr_fig / num_cols), curr_fig % num_cols].imshow(pred[c, :, :])
            ax[floor(curr_fig / num_cols), curr_fig % num_cols].set_title('Prediction' + str(c))

        # pos2 = ax[1,(input_c+1) % num_cols].imshow(pred[:, :])
        # ax[ 1, (input_c+1) % num_cols].set_title('Prediction')
        # fig.colorbar(pos2, ax=ax[1, (num_channels+1) % num_cols])

    fig.suptitle(title)
    plt.show()
    plt.tight_layout()

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, title + '.jpg'))

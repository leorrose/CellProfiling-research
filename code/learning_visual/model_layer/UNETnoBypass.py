import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn
from argparse import Namespace

from model_layer.UNET import unify_test_function


class UnetnoBypass(pl.LightningModule):

    # fixed bug for pl loading model_checkpoint according to https://github.com/PyTorchLightning/pytorch-lightning/issues/2909
    def __init__(self, *args, **kwargs):
        super(UnetnoBypass, self).__init__()

        if isinstance(kwargs, dict):
            hparams = Namespace(**kwargs)

        self.save_hyperparameters(kwargs)
        self.n_channels = hparams.n_input_channels
        self.n_classes = hparams.n_classes
        self.input_size = hparams.input_size
        self.h = hparams.input_size[0]
        self.w = hparams.input_size[1]
        self.minimize_net_factor = hparams.minimize_net_factor
        self.bilinear = True

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    self.up = nn.ConvTranpose2d(in_channels // 2, in_channels // 2,
                                                kernel_size=2, stride=2)

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
                x = torch.cat([x1], dim=1)
                return self.conv(x)

        first_layer_depth = int(64 / self.minimize_net_factor)
        self.inc = double_conv(self.n_channels, first_layer_depth)
        self.down1 = down(first_layer_depth, first_layer_depth * 2)
        self.down2 = down(first_layer_depth * 2, first_layer_depth * 4)
        self.down3 = down(first_layer_depth * 4, first_layer_depth * 8)
        self.down4 = down(first_layer_depth * 8, first_layer_depth * 8)
        self.up1 = up(first_layer_depth * 8, first_layer_depth * 4)
        self.up2 = up(first_layer_depth * 4, first_layer_depth * 2)
        self.up3 = up(first_layer_depth * 2, first_layer_depth)
        self.up4 = up(first_layer_depth, first_layer_depth)
        self.out = nn.Conv2d(first_layer_depth, self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        _, loss, pcc = unify_test_function(self, batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_pcc', pcc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.log('avg_val_loss', avg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-8)

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=6, verbose=True)
        checkpoint = ModelCheckpoint(monitor="val_loss")
        return [early_stop, checkpoint]

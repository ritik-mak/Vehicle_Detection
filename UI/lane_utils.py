import torch
import torch.nn as nn
import torchvision.transforms.functional as TF  # required for center_crop
import cv2
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        return self.conv(inputs)

class LaneNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(LaneNet, self).__init__()
        self.conv_1 = DoubleConv(in_channels, 16)
        self.maxpool_1 = nn.MaxPool2d(2, 2)

        self.conv_2 = DoubleConv(16, 32)
        self.maxpool_2 = nn.MaxPool2d(2, 2)

        self.conv_3 = DoubleConv(32, 64)
        self.maxpool_3 = nn.MaxPool2d(2, 2)

        self.conv_4 = DoubleConv(64, 128)

        self.upconv_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up_3 = DoubleConv(128, 64)

        self.upconv_2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up_2 = DoubleConv(64, 32)

        self.upconv_1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up_1 = DoubleConv(32, 16)

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, inputs):
        x1 = self.conv_1(inputs)
        x2 = self.conv_2(self.maxpool_1(x1))
        x3 = self.conv_3(self.maxpool_2(x2))
        x4 = self.conv_4(self.maxpool_3(x3))

        x = self.upconv_3(x4)
        x = self.conv_up_3(torch.cat([self._crop_tensor(x3, x), x], dim=1))

        x = self.upconv_2(x)
        x = self.conv_up_2(torch.cat([self._crop_tensor(x2, x), x], dim=1))

        x = self.upconv_1(x)
        x = self.conv_up_1(torch.cat([self._crop_tensor(x1, x), x], dim=1))

        return self.final_conv(x)

    def _crop_tensor(self, enc_feat, dec_feat):
        _, _, h, w = dec_feat.shape
        return TF.center_crop(enc_feat, [h, w])


def load_lane_model():
    """
    Load the trained LaneNet model from disk.
    """
    model = LaneNet()
    model.load_state_dict(torch.load("PROJECT/models/LANE_MODEL.pth", map_location="cpu"))
    model.eval()
    return model


def segment_lane(frame, model):
    """
    Apply the LaneNet model to detect lanes and overlay the mask on the original frame.
    :param frame: BGR image as numpy array
    :param model: loaded LaneNet model
    :return: overlayed BGR image
    """
    orig = frame.copy()
    input_img = cv2.resize(frame, (572, 572))
    input_tensor = torch.from_numpy(input_img / 255.0).float().permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(orig, 1.0, mask_color, 0.5, 0)

import torch
import torch.nn as nn


class AddCoords(nn.Module):
    ''' Adding coordinates to the input tensor
    '''
    def __init__(self, rank, with_r=False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, input_tensor):
        if self.rank == 1:
            batch_size, channels, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            # Normalize and zero center
            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size, 1, 1)

            if input_tensor.is_cuda:
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel])

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))

        elif self.rank == 2:
            batch_size, channels, dim_y, dim_x = input_tensor.shape

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)

            xx_channel, yy_channel = torch.meshgrid([xx_range, yy_range])

            # Normalize and zero center
            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.view(1, 1, xx_channel.shape[0], xx_channel.shape[1])
            yy_channel = yy_channel.view(1, 1, yy_channel.shape[0], yy_channel.shape[1])
            xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)
            if dim_y == 1 and dim_x ==1:
                xx_channel = torch.zeros([batch_size, 1,1,1])
                yy_channel = torch.zeros([batch_size, 1,1,1])
            if input_tensor.is_cuda:
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()

            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))

        elif self.rank == 3:
            batch_size, channels, dim_z, dim_y, dim_x = input_tensor.shape

            xx_range = torch.arange(dim_x, dtype=torch.int32)
            yy_range = torch.arange(dim_y, dtype=torch.int32)
            zz_range = torch.arange(dim_x, dtype=torch.int32)

            xx_channel, yy_channel, zz_channel = torch.meshgrid([xx_range,
                                                                 yy_range,
                                                                 zz_range])

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_z - 1)
            zz_channel = yy_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1
            zz_channel = zz_channel * 2 - 1

            xx_channel = xx_channel.view(1, 1, xx_channel.shape[0],
                                               xx_channel.shape[1],
                                               xx_channel.shape[2])
            yy_channel = yy_channel.view(1, 1, yy_channel.shape[0],
                                               yy_channel.shape[1],
                                               yy_channel.shape[2])
            zz_channel = zz_channel.view(1, 1, zz_channel.shape[0],
                                               zz_channel.shape[1],
                                               zz_channel.shape[2])
            xx_channel = xx_channel.repeat(batch_size, 1, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size, 1, 1, 1, 1)
            zz_channel = zz_channel.repeat(batch_size, 1, 1, 1, 1)

            if input_tensor.is_cuda:
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
                zz_channel = zz_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
        else:
            raise NotImplementedError

        if self.with_r:
            out = torch.cat([out, rr], dim=1)

        return out


class CoordConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv1d, self).__init__()
        self.rank = 1
        self.add_coords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv1d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

        def forward(self, input_tensor):
            out = self.add_coords(input_tensor)
            out = self.conv(out)
            return out


class CoordConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConvTranspose1d, self).__init__()
        self.rank = 1
        self.add_coords = AddCoords(self.rank, with_r)
        self.convT = nn.ConvTranspose1d(in_channels + self.rank + int(with_r), out_channels,
                                        kernel_size, stride, padding, output_padding, groups,
                                        bias, dilation)

    def forward(self, input_tensor):
        out = self.add_coords(input_tensor)
        out = self.convT(out)
        return out


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv2d, self).__init__()
        self.rank = 2
        self.add_coords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        out1 = self.add_coords(input_tensor)
        out = self.conv(out1)
        return out
    


class CoordConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1, with_r=False):
        super(CoordConvTranspose2d, self).__init__()
        self.rank = 2
        self.add_coords = AddCoords(self.rank, with_r)
        self.convT = nn.ConvTranspose2d(in_channels + self.rank + int(with_r), out_channels,
                                        kernel_size, stride, padding, output_padding,
                                        groups, bias, dilation)

    def forward(self, input_tensor):
        out1 = self.add_coords(input_tensor)
        out = self.convT(out1)
        return out
    
class CoordConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv3d, self).__init__()
        self.rank = 3
        self.add_coords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv3d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        out = self.add_coords(input_tensor)
        out = self.conv(out)
        return out


class CoordConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1, with_r=False):
        super(CoordConvTranspose3d, self).__init__()
        self.rank = 3
        self.add_coords = AddCoords(self.rank, with_r)
        self.convT = nn.ConvTranspose3d(in_channels + self.rank + int(with_r), out_channels,
                                        kernel_size, stride, padding, output_padding, groups,
                                        bias, dilation)

    def forward(self, input_tensor):
        out = self.add_coords(input_tensor)
        out = self.convT(out)
        return out

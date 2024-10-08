import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary

Nc = 32  # The number of subcarriers
Nt = 32  # The number of transmit antennas

class Encoder(nn.Module):
    # input: (batch_size, Nc, Nt) channel matrix
    # output: (batch_size, encoded_dim) codeword
    # CSI_NET
    def __init__(self, encoded_dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=2),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.fc = nn.Linear(in_features=2 * Nc * Nt, out_features=encoded_dim)
        self.leakyrelu = nn.LeakyReLU()
        self.name = "Csinet"

    def forward(self, x):
        out = self.conv_block(x)
        # out.shape = (batch_size, 2, Nc, Nt)
        # if test: show(out)
        out = torch.reshape(out, (out.shape[0], -1))
        # out.shape = (batch_size, 2*Nc*Nt)
        out = self.fc(out)
        # if test: show(torch.reshape(out, (batch_size, 1, 4, encoded_dim//4)))
        # out.shape = (batch_size, encoded_dim)

        return out


class Refinenet(nn.Module):
    # input: (batch_size, 2, Nc, Nt)
    # output: (batch_size, 2, Nc, Nt)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=2),
            nn.LeakyReLU(negative_slope=0.3),
        )

    def forward(self, x):
        skip_connection = x
        out = self.conv1(x)
        # out.shape = (batch_size, 8, Nc, Nt)
        out = self.conv2(out)
        # out.shape = (batch_size, 16, Nc, Nt)
        out = self.conv3(out)
        # out.shape = (batch_size, 2, Nc, Nt)
        out = out + skip_connection

        return out


class Decoder(nn.Module):
    # input: (batch_size, encoded_dim) codeword
    # output: (batch_size, Nc, Nt) reconstructed channel matrix
    # CSI_NET
    def __init__(self, encoded_dim):
        super().__init__()
        self.fc = nn.Linear(in_features=encoded_dim, out_features=2 * Nc * Nt)
        self.refine1 = Refinenet()
        self.refine2 = Refinenet()
        self.name = "Csinet"

    def forward(self, x):
        # x.shape = (batch_size, encoded_dim)
        out = self.fc(x)
        # out.shape = (batch_size, 2*Nc*Nt)
        out = torch.reshape(out, (out.shape[0], 2, Nc, Nt))
        # if test: show(out)
        # out.shape = (batch_size, 2, Nc, Nt)
        out = self.refine1(out)
        # if test: show(out)
        # out.shape = (batch_size, 2, Nc, Nt)
        out = self.refine2(out)
        # if test: show(out)
        # out.shape = (batch_size, 2, Nc, Nt)

        # channel_real = out[:, 0, :, :]
        # channel_imag = out[:, 1, :, :]
        # out = channel_real + 1j * channel_imag
        # out.shape = (batch_size, Nc, Nt)
        return out
    


class Csinet(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        self.encoder = Encoder(encoded_dim)
        self.decoder = Decoder(encoded_dim)
        self.name = self.encoder.name + '-' + self.decoder.name
    
    def forward(self, x):
        encoded_vector = self.encoder(x)
        x_recovered = self.decoder(encoded_vector)

        return encoded_vector, x_recovered
    

if __name__ == "__main__":
    encoded_dim = 32
    encoder = Encoder(encoded_dim)
    decoder = Decoder(encoded_dim)
    autoencoder = Csinet(encoded_dim)
    summary(encoder, input_size=(16, 2, 32, 32))
    summary(decoder, input_size=(16, 32))
    summary(autoencoder, input_size=(16, 2, 32, 32))
    
    print("done")

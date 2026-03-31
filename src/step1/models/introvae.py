import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import math
import time
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import data_parallel
import pdb


class _Residual_Block(nn.Module):
    def __init__(self, norm, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(midc, momentum=0.5)
        elif norm == 'instance':
            self.bn1 = nn.InstanceNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        if norm == 'batch':
            self.bn2 = nn.BatchNorm2d(outc, momentum=0.5)
        elif norm == 'instance':
            self.bn2 = nn.InstanceNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output, identity_data)))
        return output


class NCEEncoder(nn.Module):
    def __init__(self, norm, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(NCEEncoder, self).__init__()

        self.hdim = hdim
        cc = channels[0]
        if norm == 'batch':
            self.main = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2),
            )
        elif norm == 'instance':
            self.main = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.InstanceNorm2d(cc),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2),
            )

        image_size = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(image_size), _Residual_Block(norm, cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(image_size // 2), nn.AvgPool2d(2))
            cc, image_size = ch, image_size // 2

        self.main.add_module('res_in_{}1'.format(image_size), _Residual_Block(norm, cc, cc, scale=1.0))
        self.main.add_module('res_in_{}2'.format(image_size), _Residual_Block(norm, cc, cc, scale=1.0))
        self.main.add_module('res_in_{}3'.format(image_size), _Residual_Block(norm, cc, cc, scale=1.0))

        self.SubNetwork = nn.Sequential()
        self.SubNetwork.add_module('fc1', nn.Linear(cc, cc))
        self.SubNetwork.add_module('ReLU1', nn.ReLU(True))
        self.SubNetwork.add_module('fc2', nn.Linear(cc, cc))
        self.SubNetwork.add_module('ReLU2', nn.ReLU(True))
        self.SubNetwork.add_module('fc3', nn.Linear(cc, cc))

        self.ProjectionHead = ProjectionHead(cc, cc)


    def forward(self, x):
        y = self.main(x)
        y = y.permute(0, 2, 3, 1).contiguous()
        y_shape = y.shape
        y = y.view(-1, y.size(3))
        y = self.SubNetwork(y).view_as(torch.Tensor(y_shape))
        y = y.permute(0, 3, 1, 2).contiguous()
        return y


class ProjectionHead(nn.Module):
    def __init__(self, inc=256, outc=256):
        super(ProjectionHead, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(inc, inc),
            nn.ReLU(inplace=True),
            nn.Linear(inc, outc)
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, x.size(3))
        return self.main(x)


class IntroAEEncoder(nn.Module):
    def __init__(self, norm, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(IntroAEEncoder, self).__init__()

        self.hdim = hdim
        cc = channels[0]
        if norm == 'batch':
            self.main = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2),
            )
        elif norm == 'instance':
            self.main = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.InstanceNorm2d(cc),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2),
            )

        image_size = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(image_size), _Residual_Block(norm, cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(image_size // 2), nn.AvgPool2d(2))
            cc, image_size = ch, image_size // 2

        self.main.add_module('res_in_{}1'.format(image_size), _Residual_Block(norm, cc, cc, scale=1.0))
        self.main.add_module('res_in_{}2'.format(image_size), _Residual_Block(norm, cc, cc, scale=1.0))
        self.main.add_module('res_in_{}3'.format(image_size), _Residual_Block(norm, cc, cc, scale=1.0))
        #self.main.add_module('res_in_{}4'.format(image_size), _Residual_Block(norm, cc, cc, scale=1.0))

        self._vq_vae = VectorQuantizerEMA(num_embeddings=64, embedding_dim=hdim,
                                          commitment_cost=0.25, decay=0.99)


    def forward(self, x):
        y = self.main(x)

        return self._vq_vae(y)


class IntroAEDecoder(nn.Module):
    def __init__(self, norm, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(IntroAEDecoder, self).__init__()

        cc = channels[-1]
        self.fc = nn.Sequential(
            nn.Linear(hdim, cc * 4 * 4),
            nn.ReLU(True),
        )

        sz = 16

        self.main = nn.Sequential()

        self.main.add_module('res_in_{}1'.format(sz), _Residual_Block(norm, cc, cc, scale=1.0))
        self.main.add_module('res_in_{}2'.format(sz), _Residual_Block(norm, cc, cc, scale=1.0))
        #self.main.add_module('res_in_{}3'.format(sz), _Residual_Block(norm, cc, cc, scale=1.0))

        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(norm, cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(norm, cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))
        # add a layer
        self.main.add_module('tanh', nn.Tanh())

    def forward(self, z):

        y = self.main(z)

        return y


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC

        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class IntroAE(nn.Module):
    def __init__(self, norm, gpuId, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(IntroAE, self).__init__()

        self.hdim = hdim
        self.gpuId = gpuId
        self.encoder = IntroAEEncoder(norm, cdim, hdim, channels, image_size)

        self.decoder = IntroAEDecoder(norm, cdim, hdim, channels, image_size)

    def forward(self, x):
        loss_vq, quantized_latent, perplexity, encodings = self.encoder(x)

        x_recon = self.decoder(quantized_latent)

        return quantized_latent, x_recon, loss_vq


    def sample(self, z):
        y = self.decode(z)
        return y

    def encode(self, x):
        mu, logvar = data_parallel(self.encoder, x, device_ids=self.gpuId, output_device=self.gpuId)
        return mu, logvar

    def decode(self, z):
        y = data_parallel(self.decoder, z, device_ids=self.gpuId, output_device=self.gpuId)
        return y

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu)

    # 1 + logvar - [(μ - μp)^2 + e^logvar]
    def kl_loss(self, mu, logvar, prior_mu=0):
        v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        v_kl = v_kl.sum(dim=-1).mul_(-0.5)  # (batch, 2)
        return v_kl

    def reconstruction_loss(self, prediction, target, size_average=False):
        error = (prediction - target).view(prediction.size(0), -1)
        error = error ** 2
        error = torch.sum(error, dim=-1)

        if size_average:
            error = error.mean()
        else:
            error = error.sum()

        return error
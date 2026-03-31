import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import data_parallel


class _Residual_Block(nn.Module):
    """增强的残差块，包含注意力机制"""

    def __init__(self, norm, inc, outc, scale=1.0, use_attention=False):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(inc, outc, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(outc, outc, 3, 1, 1, bias=False)

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(outc)
            self.bn2 = nn.BatchNorm2d(outc)
        elif norm == 'instance':
            self.bn1 = nn.InstanceNorm2d(outc)
            self.bn2 = nn.InstanceNorm2d(outc)

        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        # 通道注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(outc, outc // 16, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(outc // 16, outc, 1, bias=False),
                nn.Sigmoid()
            )

        if inc != outc:
            self.conv_expand = nn.Conv2d(inc, outc, 1, 1, 0, bias=False)
        else:
            self.conv_expand = None

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)

        # 应用通道注意力
        if self.use_attention:
            attention_weights = self.channel_attention(output)
            output = output * attention_weights

        output = self.relu2(self.bn2(torch.add(output, identity_data)))
        return output


class VectorQuantizerEMA(nn.Module):
    """优化的VQ层，更适合小尺寸图像"""

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


def Normalize(in_channels):
    """添加归一化类，与introvae保持一致"""
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class MultiHeadAttnBlock(nn.Module):
    """复用introvae中的多头注意力块，适配改进版模型"""
    def __init__(self, in_channels, head_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.head_size = head_size
        self.att_size = in_channels // head_size
        assert(in_channels % head_size == 0), '通道数必须能被头数整除'

        self.norm1 = Normalize(in_channels)
        self.norm2 = Normalize(in_channels)

        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):
        h_ = x
        h_ = self.norm1(h_)
        y = self.norm2(y) if y is not None else h_

        q = self.q(y)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, self.head_size, self.att_size, h*w).permute(0, 3, 1, 2)
        k = k.reshape(b, self.head_size, self.att_size, h*w).permute(0, 3, 1, 2)
        v = v.reshape(b, self.head_size, self.att_size, h*w).permute(0, 3, 1, 2)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        scale = int(self.att_size) **(-0.5)
        q.mul_(scale)
        w_ = torch.matmul(q, k)
        w_ = F.softmax(w_, dim=3)
        w_ = w_.matmul(v)

        w_ = w_.transpose(1, 2).contiguous().view(b, h, w, -1).permute(0, 3, 1, 2)
        w_ = self.proj_out(w_)

        return x + w_




class NCEEncoder(nn.Module):

    def __init__(self, norm, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512], image_size=96):
        super(NCEEncoder, self).__init__()
        self.hdim = hdim
        cc = channels[0]  # 初始通道为64

        # 初始卷积层：保持双层卷积设计，输入通道cdim->64
        if norm == 'batch':
            self.initial_conv = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),
                nn.Conv2d(cc, cc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),
            )
        elif norm == 'instance':
            self.initial_conv = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.InstanceNorm2d(cc),
                nn.LeakyReLU(0.2),
                nn.Conv2d(cc, cc, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(cc),
                nn.LeakyReLU(0.2),
            )

        # 下采样块：适配5个通道的结构（channels长度为5，channels[1:]长度为4）
        self.image_size = image_size
        self.downsample_blocks = nn.ModuleList()
        for i, ch in enumerate(channels[1:]):  # 遍历[128, 256, 512, 512]
            # 最后一个通道（512）使用注意力机制
            use_attention = (i == len(channels[1:]) - 1)  # i=3时为True
            block = nn.Sequential(
                _Residual_Block(norm, cc, ch, use_attention=use_attention),
                _Residual_Block(norm, ch, ch, use_attention=use_attention),
                nn.AvgPool2d(2) if i < 3 else nn.Identity()  # 前3个阶段下采样（到512通道时停止）
            )
            self.downsample_blocks.append(block)
            # 更新当前通道数和图像尺寸（仅前3次下采样生效）
            cc, image_size = ch, image_size // 2 if i < 3 else image_size

        # 瓶颈层：输入通道为512（channels[-1]），保持注意力机制
        self.bottleneck = nn.Sequential(
            _Residual_Block(norm, cc, cc, use_attention=True),
            _Residual_Block(norm, cc, cc, use_attention=True),
            _Residual_Block(norm, cc, cc, use_attention=True),
        )

        # 注意力模块：适配hdim=512（与最终通道数一致）
        self.attn = MultiHeadAttnBlock(in_channels=hdim, head_size=8)  # 512通道可被8整除
        self.block = _Residual_Block(norm, cc, cc, use_attention=True)
        self.attn1 = MultiHeadAttnBlock(in_channels=hdim, head_size=8)
        self.block1 = _Residual_Block(norm, cc, cc, use_attention=True)

        # 输出层：确保输出通道为hdim=512
        self.norm_out = Normalize(hdim)
        self.conv_out = torch.nn.Conv2d(hdim, hdim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, vqvae):
        # 初始特征提取（输出64通道）
        zd = self.initial_conv(x)

        # 下采样过程：逐步提升到512通道
        for block in self.downsample_blocks:
            zd = block(zd)  # 最终输出512通道特征图

        # 瓶颈层处理（保持512通道）
        zd = self.bottleneck(zd)

        # VQ-VAE量化（输入512通道，与vqvae的embedding_dim=512匹配）
        _, zp, _, _ = vqvae(zd)

        # 注意力机制处理（512通道兼容）
        z = self.attn(zd, zp)
        z = self.block(z)
        z = self.attn1(zd, z)
        z = self.block1(z)

        # 输出层（确保输出512通道）
        z = self.conv_out(self.norm_out(z))
        return z, zp


class EnhancedIntroAEEncoder(nn.Module):
    """专门为96x96图像优化的编码器"""

    def __init__(self, norm, cdim=3, hdim=256, channels=[64, 128, 256, 512, 512], image_size=96):
        super(EnhancedIntroAEEncoder, self).__init__()

        self.hdim = hdim
        cc = channels[0]

        # 使用与EnhancedIntroAEEncoder相同的初始卷积层
        if norm == 'batch':
            self.initial_conv = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),
                nn.Conv2d(cc, cc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(cc),
                nn.LeakyReLU(0.2),
            )
        elif norm == 'instance':
            self.initial_conv = nn.Sequential(
                nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
                nn.InstanceNorm2d(cc),
                nn.LeakyReLU(0.2),
                nn.Conv2d(cc, cc, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(cc),
                nn.LeakyReLU(0.2),
            )

        image_size = image_size
        self.downsample_blocks = nn.ModuleList()

        # 适配的下采样块
        for i, ch in enumerate(channels[1:]):
            block_layers = []
            use_attention = (i == len(channels[1:]) - 1)
            block_layers.append(_Residual_Block(norm, cc, ch, scale=1.0, use_attention=use_attention))
            block_layers.append(_Residual_Block(norm, ch, ch, scale=1.0, use_attention=use_attention))
            if i < 3:
                block_layers.append(nn.AvgPool2d(2))

            self.downsample_blocks.append(nn.Sequential(*block_layers))
            cc, image_size = ch, image_size // 2 if i < 3 else image_size

        # 适配的瓶颈层
        self.bottleneck = nn.Sequential(
            _Residual_Block(norm, cc, cc, scale=1.0, use_attention=True),
            _Residual_Block(norm, cc, cc, scale=1.0, use_attention=True),
            _Residual_Block(norm, cc, cc, scale=1.0, use_attention=True),
        )

        # 优化的VQ层 - 更适合小尺寸图像
        self._vq_vae = VectorQuantizerEMA(
            num_embeddings=256,  # 减少codebook大小，避免过拟合
            embedding_dim=hdim,
            commitment_cost=0.15,  # 降低commitment cost
            decay=0.99
        )

    def forward(self, x):
        # 初始卷积
        y = self.initial_conv(x)

        # 下采样
        for block in self.downsample_blocks:
            y = block(y)

        # 瓶颈层
        y = self.bottleneck(y)

        # VQ-VAE
        return self._vq_vae(y)


class EnhancedIntroAEDecoder(nn.Module):
    """专门为96x96图像优化的解码器"""

    def __init__(self, norm, cdim=3, hdim=256, channels=[16, 32, 64, 128, 256], image_size=96):
        super(EnhancedIntroAEDecoder, self).__init__()

        cc = channels[-1]

        # 计算最终的瓶颈特征图尺寸 - 由于减少了下采样，特征图更大
        # 对于96x96图像，经过4次下采样后：96->48->24->12->6
        final_size = image_size // (2 ** 4)  # 6x6特征图
        self.final_size = final_size
        self.target_size = image_size  # 目标输出尺寸

        # 优化的初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(hdim, cc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cc) if norm == 'batch' else nn.InstanceNorm2d(cc),
            nn.LeakyReLU(0.2),
            _Residual_Block(norm, cc, cc, scale=1.0, use_attention=True),
        )

        self.upsample_blocks = nn.ModuleList()

        # 优化的上采样块 - 使用转置卷积获得更好的细节
        for i, ch in enumerate(channels[::-1][1:]):
            block_layers = []
            # 每个上采样阶段包含2个残差块
            block_layers.append(_Residual_Block(norm, cc, ch, scale=1.0, use_attention=(i < 2)))
            block_layers.append(_Residual_Block(norm, ch, ch, scale=1.0, use_attention=(i < 2)))
            # 使用转置卷积上采样
            block_layers.append(nn.ConvTranspose2d(ch, ch, 4, 2, 1, bias=False))
            block_layers.append(nn.BatchNorm2d(ch) if norm == 'batch' else nn.InstanceNorm2d(ch))
            block_layers.append(nn.LeakyReLU(0.2))

            self.upsample_blocks.append(nn.Sequential(*block_layers))
            cc = ch

        # 增强的输出层 - 多尺度特征融合
        self.output_layers = nn.Sequential(
            _Residual_Block(norm, cc, cc, scale=1.0, use_attention=True),
            _Residual_Block(norm, cc, 128, scale=1.0, use_attention=True),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64) if norm == 'batch' else nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32) if norm == 'batch' else nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, cdim, 5, 1, 2),  # 5x5卷积获得更好的细节
            nn.Tanh()
        )

        # 最终尺寸调整层，确保输出尺寸与输入一致
        self.final_adjust = nn.Upsample(size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)

    def forward(self, z):
        # 初始卷积调整特征图
        y = self.initial_conv(z)

        # 上采样
        for block in self.upsample_blocks:
            y = block(y)

        # 输出层
        y = self.output_layers(y)

        # 确保输出尺寸正确
        if y.size(2) != self.target_size or y.size(3) != self.target_size:
            y = self.final_adjust(y)

        return y


class EnhancedIntroAE(nn.Module):
    """专门优化的IntroAE模型，针对小尺寸图像重建"""

    def __init__(self, norm, gpuId, cdim=3, hdim=256, channels=[16, 32, 64, 128, 256], image_size=96):
        super(EnhancedIntroAE, self).__init__()

        self.hdim = hdim
        self.gpuId = gpuId

        # 使用优化的编码器和解码器
        self.encoder = EnhancedIntroAEEncoder(norm, cdim, hdim, channels, image_size)
        self.decoder = EnhancedIntroAEDecoder(norm, cdim, hdim, channels, image_size)
        # 新增NCE编码器
        self.nce_encoder = NCEEncoder(norm, cdim, hdim, channels, image_size)

    def forward(self, x):
        loss_vq, quantized_latent, perplexity, encodings = self.encoder(x)
        x_recon = self.decoder(quantized_latent)

        # NCE编码，用于对比学习
        #nce_features, zp = self.nce_encoder(x, self.encoder)

        return quantized_latent, x_recon, loss_vq

    def encode(self, x):
        """编码方法，返回量化后的latent"""
        loss_vq, quantized_latent, perplexity, encodings = self.encoder(x)
        return quantized_latent

    def decode(self, z):
        """解码方法"""
        return self.decoder(z)

    def sample(self, z):
        """采样方法"""
        return self.decode(z)

    def get_latent_representation(self, x):
        """获取latent表示，包括编码索引和量化向量"""
        loss_vq, quantized_latent, perplexity, encodings = self.encoder(x)
        return {
            'quantized': quantized_latent,
            'encodings': encodings,
            'perplexity': perplexity
        }

    def nce_loss(self, features, zp, temperature=0.1):
        """
        对比学习损失函数
        """
        batch_size = features.shape[0]

        # 归一化特征
        features = F.normalize(features, p=2, dim=1)
        zp = F.normalize(zp, p=2, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features.view(batch_size, -1),
                                         zp.view(batch_size, -1).t()) / temperature

        # 正样本对在对角线上
        labels = torch.arange(batch_size).to(features.device)

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


# 兼容原有接口的包装类
class IntroAE(EnhancedIntroAE):
    """兼容原有IntroAE接口的包装类"""

    def __init__(self, norm, gpuId, cdim=3, hdim=256, channels=[16, 32, 64, 128, 256], image_size=96):
        super(IntroAE, self).__init__(norm, gpuId, cdim, hdim, channels, image_size)
        print("使用ImprovedIntroAE_nce")

    def reparameterize(self, mu, logvar):
        """保持与原有VAE接口兼容"""
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def kl_loss(self, mu, logvar, prior_mu=0):
        """保持与原有VAE接口兼容"""
        v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        v_kl = v_kl.sum(dim=-1).mul_(-0.5)
        return v_kl

    def reconstruction_loss(self, prediction, target, size_average=False):
        """保持与原有损失函数接口兼容"""
        error = (prediction - target).view(prediction.size(0), -1)
        error = error ** 2
        error = torch.sum(error, dim=-1)

        if size_average:
            error = error.mean()
        else:
            error = error.sum()

        return error
import math

from matplotlib import pyplot as plt

from gconv import *


def normalization(channels):
    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class ResBlock(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            emb_ch,
            using_eq=True
    ):
        super().__init__()
        self.in_ch = in_ch
        self.emb_ch = emb_ch
        self.out_ch = out_ch
        self.using_eq = using_eq
        self.in_layers = nn.Sequential(
            normalization(in_ch),
            nn.SiLU(),
            PDRREConv(in_ch, out_ch, 3, 1, 1) if using_eq else
            nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_ch,
                out_ch,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(out_ch),
            nn.SiLU(),
            PDRREConv(out_ch, out_ch, 3, 1, 1),
        )

        if self.in_ch == out_ch:
            self.skip_connection = PDRREConv(in_ch, out_ch, 3, 1, 1) if self.using_eq else \
                nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        else:
            self.skip_connection = PointwiseRREConv(in_ch, out_ch) if self.using_eq else \
                nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class UNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, ch=96, using_eq=True):
        super().__init__()
        self.using_eq = using_eq
        self.ch = ch
        time_embed_dim = ch * 4
        self.time_embed = nn.Sequential(
            nn.Linear(ch, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.lift = nn.Sequential(
            nn.GroupNorm(1, in_ch),
            nn.SiLU(),
            RRLConv(in_ch, ch, 2, 2, 0) if using_eq else nn.Conv2d(in_ch, ch, 2, 2, 0)
        )
        # Downsample blocks
        self.down1 = ResBlock(ch, ch, time_embed_dim, using_eq)
        self.down2 = ResBlock(ch, ch * 2, time_embed_dim, using_eq)
        self.down3 = ResBlock(ch * 2, ch * 4, time_embed_dim, using_eq)

        # Middle blocks
        self.mid1 = ResBlock(ch * 4, ch * 4, time_embed_dim, using_eq)
        self.mid2 = ResBlock(ch * 4, ch * 4, time_embed_dim, using_eq)

        # Upsample blocks
        self.up1 = ResBlock(ch * 8, ch * 2, time_embed_dim, using_eq)
        self.up2 = ResBlock(ch * 4, ch, time_embed_dim, using_eq)
        self.up3 = ResBlock(ch * 2, ch, time_embed_dim, using_eq)

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            PDRREUp(ch, out_ch) if using_eq else nn.Upsample(scale_factor=2)
        )

    def forward(self, t, x):
        # Time embedding
        emb = self.time_embed(timestep_embedding(t, self.ch))
        x = self.lift(x)
        # Downsample
        h1 = self.down1(x, emb)
        h2 = self.down2(h1, emb)
        h3 = self.down3(h2, emb)

        # Middle
        h = self.mid1(h3, emb)
        h = self.mid2(h, emb)

        # Upsample with skip connections
        h = torch.cat([h, h3], dim=1)
        h = self.up1(h, emb)

        h = torch.cat([h, h2], dim=1)
        h = self.up2(h, emb)

        h = torch.cat([h, h1], dim=1)
        h = self.up3(h, emb)

        return self.out(h).mean(dim=2) if self.using_eq else self.out(h)


if __name__ == '__main__':
    x = torch.randn(1, 4, 6, 6)
    x_rot = torch.rot90(x, dims=(-2, -1))
    unet = UNet()
    emd = torch.rand([1])
    z = unet(emd, x)
    z = torch.rot90(z, dims=(-2, -1))
    z_rot = unet(emd, x_rot)
    fig, ax = plt.subplots(2, 4)
    for i in range(4):
        ax[0][i].imshow(z[0, 0, i, :, :].detach().numpy())
        ax[1][i].imshow(z_rot[0, 0, i, :, :].detach().numpy())
    plt.show()
#
# net = UNet()
# def get_parameter_number(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(get_parameter_number(net))

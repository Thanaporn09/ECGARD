import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.layers import DropPath

from engine.registry import MODELS


def build_mamba_layer(d_model: int, causal: bool = True) -> nn.Module:
    try:
        return Mamba(d_model=d_model, causal=causal)
    except TypeError:
        return Mamba(d_model=d_model)


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int = 512, code_dim: int = 128, beta: float = 0.25):
        super().__init__()
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.beta = float(beta)

        self.codebook = nn.Embedding(self.num_codes, self.code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.num_codes, 1.0 / self.num_codes)

    def forward(self, z_e: torch.Tensor):
        b, n, d = z_e.shape
        if d != self.code_dim:
            raise ValueError(f"code_dim mismatch: got {d}, expected {self.code_dim}")

        z_flat = z_e.reshape(-1, d)
        codebook = self.codebook.weight

        z_sq = z_flat.pow(2).sum(dim=1, keepdim=True)
        e_sq = codebook.pow(2).sum(dim=1, keepdim=True).t()
        ze = z_flat @ codebook.t()
        dist = z_sq - 2.0 * ze + e_sq

        codes_idx = torch.argmin(dist, dim=1)
        z_q = self.codebook(codes_idx).view(b, n, d)

        z_q_st = z_e + (z_q - z_e).detach()
        vq_loss = F.mse_loss(z_q.detach(), z_e) + self.beta * F.mse_loss(z_q, z_e.detach())

        stats = {"codes_idx": codes_idx.view(b, n)}
        return z_q_st, vq_loss, stats


class HybridBiMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        kernel_size: int = 3,
        fusion: str = "add",
        alpha: float = 0.5,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        self.mamba_fwd = build_mamba_layer(d_model, causal=True)
        self.mamba_bwd = build_mamba_layer(d_model, causal=True)

        self.fusion_proj = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.fusion = fusion
        self.alpha = float(alpha)

        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
        self.activation = nn.GELU()

        if fusion == "gate":
            self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)

        h_fwd = self.mamba_fwd(h)

        h_rev = torch.flip(h, dims=[1])
        h_bwd = self.mamba_bwd(h_rev)
        h_bwd = torch.flip(h_bwd, dims=[1])

        h_mamba = self.fusion_proj(torch.cat([h_fwd, h_bwd], dim=-1))

        h_conv = self.conv(h.transpose(1, 2)).transpose(1, 2)
        h_conv = self.activation(h_conv)

        if self.fusion == "add":
            h_out = self.alpha * h_mamba + (1.0 - self.alpha) * h_conv
        elif self.fusion == "gate":
            g = torch.sigmoid(self.gate(torch.cat([h_conv, h_mamba], dim=-1)))
            h_out = g * h_mamba + (1.0 - g) * h_conv
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion}")

        h_out = self.dropout(h_out)
        return x + self.drop_path(h_out)


class HybridCausalMambaBlockParallel(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        kernel_size: int = 3,
        fusion: str = "add",
        alpha: float = 0.5,
        dilation: int = 1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = build_mamba_layer(d_model, causal=True)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.fusion = fusion
        self.alpha = float(alpha)

        pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
        )
        self.activation = nn.GELU()

        if fusion == "gate":
            self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)

        h_conv = self.conv(h.transpose(1, 2))
        h_conv = self.activation(h_conv)
        h_conv = h_conv[:, :, : h.size(1)]
        h_conv = h_conv.transpose(1, 2)

        h_mamba = self.mamba(h)

        if self.fusion == "add":
            h_out = self.alpha * h_mamba + (1.0 - self.alpha) * h_conv
        elif self.fusion == "gate":
            g = torch.sigmoid(self.gate(torch.cat([h_conv, h_mamba], dim=-1)))
            h_out = g * h_mamba + (1.0 - g) * h_conv
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion}")

        h_out = self.dropout(h_out)
        return x + self.drop_path(h_out)


class PatchEmbed1D(nn.Module):
    def __init__(self, in_ch: int = 1, embed_dim: int = 128, patch_size: int = 16):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv1d(in_ch, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.shape[1] != 1:
            x = x.transpose(1, 2)
        x = self.proj(x)
        return x.transpose(1, 2)


class HierarchicalMambaEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        d_model: int = 128,
        depths=(2, 2, 2, 2),
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        downsample: bool = True,
        expand_dim: bool = True,
        fusion: str = "add",
        alpha: float = 0.5,
        use_dilated_conv: bool = False,
        dilation_growth: int = 2,
        causal: bool = True,
    ):
        super().__init__()
        self.num_stages = len(depths)
        self.downsample = bool(downsample)
        self.expand_dim = bool(expand_dim)
        self.use_dilated_conv = bool(use_dilated_conv)
        self.dilation_growth = int(dilation_growth)
        self.causal = bool(causal)

        self.input_proj = nn.Linear(in_channels, d_model)
        self.stages = nn.ModuleList()
        in_dim = d_model
        dilation = 1

        if drop_path_rate > 0.0:
            dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        else:
            dpr = [0.0] * sum(depths)
        dp_idx = 0

        for i, depth in enumerate(depths):
            if self.causal:
                blocks = nn.ModuleList([
                    HybridCausalMambaBlockParallel(
                        in_dim,
                        dropout=dropout,
                        drop_path=dpr[dp_idx + j],
                        fusion=fusion,
                        alpha=alpha,
                        dilation=dilation,
                    )
                    for j in range(depth)
                ])
            else:
                blocks = nn.ModuleList([
                    HybridBiMambaBlock(
                        in_dim,
                        dropout=dropout,
                        drop_path=dpr[dp_idx + j],
                        fusion=fusion,
                        alpha=alpha,
                    )
                    for j in range(depth)
                ])

            dp_idx += depth
            self.stages.append(nn.Sequential(*blocks))

            if i < len(depths) - 1 and self.downsample:
                out_dim = in_dim * 2 if self.expand_dim else in_dim
                self.stages.append(
                    nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
                )
                in_dim = out_dim

            if self.causal and self.use_dilated_conv:
                dilation *= self.dilation_growth

        self.out_norm = nn.LayerNorm(in_dim)
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor):
        h = self.input_proj(x)
        feats = []

        for stage in self.stages:
            if isinstance(stage, nn.Sequential):
                h = stage(h)
                feats.append(h)
            else:
                h = stage(h.transpose(1, 2)).transpose(1, 2)

        h = self.out_norm(h)
        if feats:
            feats[-1] = h
        return feats[::-1]


class TemporalWeightedFusion1D(nn.Module):
    def __init__(self, in_dims, reduction: int = 16, kernel_size: int = 3):
        super().__init__()
        self.num_levels = len(in_dims)
        out_dim = in_dims[0]

        self.lateral_convs = nn.ModuleList([
            nn.Conv1d(in_dims[i], out_dim, kernel_size=1) for i in range(self.num_levels)
        ])
        self.weights = nn.Parameter(torch.ones(self.num_levels))

        hidden = max(out_dim // reduction, 8)
        self.ca_fc = nn.Sequential(
            nn.Linear(out_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        self.ta_conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.freq_conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

        self.smooth = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def _apply_ca(self, f: torch.Tensor) -> torch.Tensor:
        avg_pool = f.mean(dim=-1)
        max_pool = f.max(dim=-1).values
        ca = self.ca_fc(avg_pool + max_pool)
        ca = torch.sigmoid(ca).unsqueeze(-1)
        return f * ca

    def _apply_ta_time(self, f: torch.Tensor) -> torch.Tensor:
        t_avg = f.mean(dim=1, keepdim=True)
        t_max = f.max(dim=1, keepdim=True).values
        t_mask = torch.sigmoid(self.ta_conv(torch.cat([t_avg, t_max], dim=1)))
        return f * t_mask

    def _apply_ta_fft(self, f: torch.Tensor) -> torch.Tensor:
        fft_vals = torch.fft.rfft(f, dim=-1)
        fft_mag = torch.abs(fft_vals)

        f_avg = fft_mag.mean(dim=1, keepdim=True)
        f_max = fft_mag.max(dim=1, keepdim=True).values
        f_mask = torch.sigmoid(self.freq_conv(torch.cat([f_avg, f_max], dim=1)))

        t_mask = F.interpolate(f_mask, size=f.size(-1), mode="linear", align_corners=False)
        return f * t_mask

    def forward(self, feats, target_tokens: int) -> torch.Tensor:
        feats = [f.transpose(1, 2) for f in feats]
        weights = F.softmax(self.weights, dim=0)

        processed = []
        ref_len = feats[0].size(-1)

        for i in range(self.num_levels):
            f = self.lateral_convs[i](feats[i])
            if f.size(-1) != ref_len:
                f = F.interpolate(f, size=ref_len, mode="linear", align_corners=False)

            f = self._apply_ca(f)
            f = self._apply_ta_time(f)
            f = self._apply_ta_fft(f)
            processed.append(f)

        fused = torch.zeros_like(processed[0])
        for w, f in zip(weights, processed):
            fused = fused + w * f

        out = self.act(self.smooth(fused))
        out = F.interpolate(out, size=target_tokens, mode="linear", align_corners=False)
        return out.transpose(1, 2)


class MambaDecoderCausalARCondFiLM(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.0,
        patch_size: int = 16,
        cond_mode: str = "concat",
        enable_condition: bool = True,
        fusion: str = "add",
        alpha: float = 0.5,
        use_dilated_conv: bool = False,
        dilation_growth: int = 2,
        disable_ar: bool = False,
    ):
        super().__init__()
        self.enable_condition = bool(enable_condition)
        self.cond_mode = cond_mode
        self.d_model = int(d_model)
        self.patch_size = int(patch_size)
        self.disable_ar = bool(disable_ar)

        self.yproj = None
        self.base_dim = None
        self.input_proj = None

        self.h_to_gamma = nn.Linear(d_model, d_model) if cond_mode == "film" else None
        self.h_to_beta = nn.Linear(d_model, d_model) if cond_mode == "film" else None

        dilation = 1
        self.layers = nn.ModuleList([
            HybridCausalMambaBlockParallel(
                d_model,
                dropout=dropout,
                drop_path=0.0,
                fusion=fusion,
                alpha=alpha,
                dilation=dilation,
            )
            for _ in range(n_layers)
        ])

        if use_dilated_conv:
            for i in range(1, len(self.layers)):
                dilation *= int(dilation_growth)
                self.layers[i].conv.dilation = (dilation,)
                k = self.layers[i].conv.kernel_size[0]
                self.layers[i].conv.padding = (dilation * (k - 1),)

        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, patch_size)

    def build_input_proj(self, base_dim: int):
        self.base_dim = int(base_dim)
        self.yproj = nn.Linear(1, self.base_dim)

        if not self.enable_condition:
            in_ch_total = self.base_dim * 2
        else:
            in_ch_total = self.base_dim * 2 + (0 if self.cond_mode == "film" else self.d_model)
        self.input_proj = nn.Linear(in_ch_total, self.d_model)

    def forward(
        self,
        h_enc: torch.Tensor,
        x_noisy: torch.Tensor,
        y_gt: torch.Tensor,
        return_hidden: bool = False,
        gamma: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
    ):
        if self.input_proj is None:
            self.build_input_proj(x_noisy.size(-1))

        if y_gt.dim() == 4 and y_gt.size(-1) == 1:
            y_gt = y_gt.squeeze(-1)

        b, n, _ = x_noisy.shape

        y_prev = torch.roll(y_gt, shifts=1, dims=1)
        y_prev[:, 0, :] = 0.0

        if self.disable_ar:
            y_prev_token = torch.zeros(b, n, self.base_dim, device=y_prev.device, dtype=y_prev.dtype)
        else:
            y_prev_token = self.yproj(y_prev.mean(dim=-1, keepdim=True))

        if not self.enable_condition:
            h = self.input_proj(torch.cat([x_noisy, y_prev_token], dim=-1))
        else:
            if self.cond_mode == "concat":
                h = self.input_proj(torch.cat([x_noisy, y_prev_token, h_enc], dim=-1))
            elif self.cond_mode == "film":
                h = self.input_proj(torch.cat([x_noisy, y_prev_token], dim=-1))
                if gamma is None or beta is None:
                    gamma = self.h_to_gamma(h_enc)
                    beta = self.h_to_beta(h_enc)
                h = h * (1.0 + gamma) + beta
            else:
                raise ValueError(f"Unknown cond_mode: {self.cond_mode}")

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        if return_hidden:
            return self.output_head(h), h
        return self.output_head(h)


@MODELS.register
class ECGARD(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        patch_size: int = 16,
        dropout: float = 0.0,
        cond_mode: str = "concat",
        enable_condition: bool = True,
        depths=(2, 2, 2, 2),
        dec_num_layers: int = 4,
        downsample: bool = True,
        expand_dim: bool = True,
        fusion: str = "add",
        alpha: float = 0.5,
        use_dilated_conv: bool = False,
        dilation_growth: int = 2,
        vq_num_codes: int = 512,
        vq_beta: float = 0.15,
        alpha_vq: float = 0.1,
        drop_path_rate: float = 0.1,
        disable_ar: bool = False,
    ):
        super().__init__()
        self.is_autoregressive = True
        self.enable_condition = bool(enable_condition)
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.disable_ar = bool(disable_ar)

        self.patch_embed = PatchEmbed1D(
            in_ch=in_channels,
            embed_dim=d_model,
            patch_size=patch_size,
        )

        self.encoder = HierarchicalMambaEncoder(
            in_channels=d_model,
            d_model=d_model,
            depths=depths,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            downsample=downsample,
            expand_dim=expand_dim,
            fusion=fusion,
            alpha=alpha,
            use_dilated_conv=use_dilated_conv,
            dilation_growth=dilation_growth,
            causal=False,
        )

        if expand_dim:
            ch = [d_model * (2 ** i) for i in range(len(depths))][::-1]
        else:
            ch = [d_model] * len(depths)

        self.fusion = TemporalWeightedFusion1D(in_dims=ch)

        latent_dim = int(self.encoder.out_dim)
        self.vq_cont_proj = nn.Linear(latent_dim, latent_dim)
        self.vq_in_proj = nn.Linear(latent_dim, latent_dim)
        self.vq_out_proj = nn.Linear(latent_dim, latent_dim)
        self.vq = VectorQuantizer(num_codes=vq_num_codes, code_dim=latent_dim, beta=vq_beta)
        self.alpha_vq = float(alpha_vq)

        self.decoder = MambaDecoderCausalARCondFiLM(
            d_model=latent_dim,
            n_layers=dec_num_layers,
            dropout=dropout,
            patch_size=patch_size,
            cond_mode=cond_mode,
            enable_condition=enable_condition,
            fusion=fusion,
            alpha=alpha,
            use_dilated_conv=use_dilated_conv,
            dilation_growth=dilation_growth,
            disable_ar=disable_ar,
        )
        self.decoder.build_input_proj(base_dim=d_model)

        self.last_vq_loss = None
        self.last_vq_stats = None

    def set_disable_ar(self, flag: bool = True):
        self.disable_ar = bool(flag)
        self.decoder.disable_ar = bool(flag)

    def _apply_vq(self, h: torch.Tensor):
        h_cont = self.vq_cont_proj(h)
        h_vq_in = self.vq_in_proj(h)
        z_q, vq_loss, vq_stats = self.vq(h_vq_in)
        h_vq = self.vq_out_proj(z_q)
        h_fused = h_cont + self.alpha_vq * h_vq
        return h_fused, vq_loss, vq_stats

    def _make_patches(self, y: torch.Tensor, n_tokens: int) -> torch.Tensor:
        y_seq = y.transpose(1, 2).squeeze(-1)
        patches = y_seq.unfold(dimension=1, size=self.patch_size, step=self.patch_size)
        if patches.size(1) != n_tokens:
            patches = patches[:, :n_tokens, :]
        return patches

    def forward(self, x_noisy: torch.Tensor, y_gt: torch.Tensor | None = None) -> torch.Tensor:
        self.decoder.disable_ar = self.disable_ar

        b = x_noisy.size(0)
        if y_gt is None:
            y_gt = torch.zeros_like(x_noisy)

        x_patches = self.patch_embed(x_noisy)
        n = x_patches.size(1)
        y_patches = self._make_patches(y_gt, n)

        feats = self.encoder(x_patches)
        h = self.fusion(feats, target_tokens=n)

        h_fused, vq_loss, vq_stats = self._apply_vq(h)
        self.last_vq_loss = vq_loss
        self.last_vq_stats = vq_stats

        y_hat_patches, _ = self.decoder(h_fused, x_patches, y_patches, return_hidden=True)

        x_hat = y_hat_patches.reshape(b, 1, -1)
        if self.out_channels != 1:
            x_hat = x_hat.repeat(1, self.out_channels, 1)
        return x_hat

    @torch.no_grad()
    def generate(self, x_noisy: torch.Tensor) -> torch.Tensor:
        self.eval()
        self.decoder.disable_ar = self.disable_ar

        b = x_noisy.size(0)
        x_patches = self.patch_embed(x_noisy)
        n = x_patches.size(1)

        feats = self.encoder(x_patches)
        h = self.fusion(feats, target_tokens=n)
        h_fused, _, _ = self._apply_vq(h)

        device = x_noisy.device
        y_prev = torch.zeros(b, n, self.patch_size, device=device)

        gamma_all = beta_all = None
        if self.enable_condition and self.decoder.cond_mode == "film":
            gamma_all = self.decoder.h_to_gamma(h_fused)
            beta_all = self.decoder.h_to_beta(h_fused)

        for t in range(n):
            if gamma_all is not None and beta_all is not None:
                gamma_t = gamma_all[:, : t + 1, :]
                beta_t = beta_all[:, : t + 1, :]
            else:
                gamma_t = beta_t = None

            y_hat_t = self.decoder(
                h_fused[:, : t + 1, :],
                x_patches[:, : t + 1, :],
                y_prev[:, : t + 1, :],
                return_hidden=False,
                gamma=gamma_t,
                beta=beta_t,
            )
            y_prev[:, t, :] = y_hat_t[:, t, :]

        x_hat = y_prev.reshape(b, 1, -1)
        if self.out_channels != 1:
            x_hat = x_hat.repeat(1, self.out_channels, 1)
        return x_hat

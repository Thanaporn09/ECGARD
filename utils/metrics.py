import torch

def denormalize(x, mu, sigma):
    return x * sigma + mu

def ssd_sum(pred, target):
    return torch.sum((pred - target) ** 2, dim=-1)

def mad_max(pred, target):
    return torch.max(torch.abs(pred - target), dim=-1).values

def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=-1) + 1e-12)

def prd(pred, target):
    num = torch.sum((pred - target) ** 2, dim=-1)
    den = torch.sum(target ** 2, dim=-1) + 1e-12
    return torch.sqrt(num / den) * 100.0

def snr_db(est, ref):
    num = torch.mean(ref ** 2, dim=-1) + 1e-12
    den = torch.mean((est - ref) ** 2, dim=-1) + 1e-12
    return 10.0 * torch.log10(num / den)

def snr_in_db(noisy, clean):
    return snr_db(noisy, clean)

def snr_out_db(pred, clean):
    return snr_db(pred, clean)

def delta_snr_db(noisy, clean, pred):
    return snr_out_db(pred, clean) - snr_in_db(noisy, clean)

def cross_corr(pred, target):
    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    targ_c = target - target.mean(dim=-1, keepdim=True)
    num = torch.mean(pred_c * targ_c, dim=-1)
    den = torch.sqrt(torch.mean(pred_c ** 2, dim=-1) * torch.mean(targ_c ** 2, dim=-1)) + 1e-12
    return num / den

def cosine_similarity(pred, target):
    dot = torch.sum(pred * target, dim=-1)
    den = torch.sqrt(torch.sum(pred ** 2, dim=-1) * torch.sum(target ** 2, dim=-1)) + 1e-12
    return dot / den

METRIC_FUNCS = {
    "SSD": ssd_sum,      # au^2
    "MAD": mad_max,      # au
    "PRD": prd,          # %
    "RMSE": rmse,        # au
    "CC": cross_corr,
    "CosSim": cosine_similarity,
}

def evaluate_metrics(pred, target, noisy=None, mu=None, sigma=None, selected=None):
    """
    Option A: Compute metrics in signal-domain "au".
    Inputs pred/target/noisy are expected to be normalized [B, 1, T].
    If mu/sigma are provided (per-window), denormalize to au before metrics.
    """
    # [B, 1, T] -> [B, T]
    pred = pred.squeeze(1)
    target = target.squeeze(1)
    if noisy is not None:
        noisy = noisy.squeeze(1)

    # Work in float64 for numerical stability
    pred = pred.double()
    target = target.double()
    if noisy is not None:
        noisy = noisy.double()

    # Denormalize to au if mu/sigma available
    if (mu is not None) and (sigma is not None):
        mu = mu.view(-1, 1).to(pred).double()
        sigma = sigma.view(-1, 1).to(pred).double()
        sigma = torch.clamp(sigma, min=1e-6)  # critical for stability

        pred = denormalize(pred, mu, sigma)
        target = denormalize(target, mu, sigma)
        if noisy is not None:
            noisy = denormalize(noisy, mu, sigma)

    selected = selected or ["SSD", "MAD", "PRD", "RMSE", "CC", "CosSim", "SNRin", "SNRout", "DeltaSNR"]
    metrics = {}

    # Standard metrics (mean across windows)
    for name in selected:
        if name in METRIC_FUNCS:
            metrics[name] = float(METRIC_FUNCS[name](pred, target).mean().item())

    # SNR metrics in dB
    if "SNRout" in selected:
        metrics["SNRout"] = float(snr_out_db(pred, target).mean().item())

    if ("SNRin" in selected) or ("DeltaSNR" in selected):
        if noisy is None:
            raise ValueError("SNRin/DeltaSNR requested but 'noisy' was not provided.")
        if "SNRin" in selected:
            metrics["SNRin"] = float(snr_in_db(noisy, target).mean().item())
        if "DeltaSNR" in selected:
            metrics["DeltaSNR"] = float(delta_snr_db(noisy, target, pred).mean().item())

    return metrics

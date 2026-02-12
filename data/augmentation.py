import numpy as np
import random
import scipy.signal as sg

# ----- Basic augmentation functions -----
def jitter(x, sigma=0.05):
    return x + np.random.normal(0, sigma, size=x.shape)

def scaling(x, sigma=0.1):
    factor = np.random.normal(1.0, sigma)
    return x * factor

def time_warp(x, sigma=0.2):
    orig_len = len(x)
    warp_factor = np.random.uniform(1 - sigma, 1 + sigma)
    warped = sg.resample(x, int(orig_len * warp_factor))
    if len(warped) > orig_len:
        warped = warped[:orig_len]
    elif len(warped) < orig_len:
        warped = np.pad(warped, (0, orig_len - len(warped)), mode="edge")
    return warped

def baseline_wander(x, fs=360, amp_range=(0.05, 0.2)):
    freq = np.random.uniform(0.1, 0.3)
    phase = np.random.uniform(0, 2*np.pi)
    amp = np.random.uniform(*amp_range)
    drift = amp * np.sin(2 * np.pi * freq * np.arange(len(x)) / fs + phase)
    return x + drift

def powerline_noise(x, fs=360):
    freq = 50 if random.random() < 0.5 else 60
    amp = np.random.uniform(0.02, 0.08)
    noise = amp * np.sin(2 * np.pi * freq * np.arange(len(x)) / fs)
    return x + noise


AUGMENT_FUNCS = {
    "jitter": jitter,
    "scaling": scaling,
    "time_warp": time_warp,
    "baseline_wander": baseline_wander,
    "powerline_noise": powerline_noise,
}


def apply_augmentation(x, fs, augment_cfg):
    """Flexible, probabilistic augmentation pipeline."""
    if not augment_cfg or not augment_cfg.get("enable", False):
        return x

    global_prob = augment_cfg.get("policy", {}).get("global_prob", 1.0)
    if random.random() > global_prob:
        return x  # skip augmentations entirely

    methods_cfg = augment_cfg.get("methods", {})
    mode = augment_cfg.get("policy", {}).get("mode", "sequential")
    n_apply = augment_cfg.get("policy", {}).get("n_apply", 1)

    available = list(methods_cfg.keys())
    if mode == "random_n":
        to_apply = random.sample(available, min(n_apply, len(available)))
    else:  # sequential mode
        to_apply = available

    for name in to_apply:
        if name not in AUGMENT_FUNCS:
            continue
        cfg = methods_cfg[name] or {}
        prob = cfg.get("prob", 1.0)
        if random.random() > prob:
            continue  # skip this one

        func = AUGMENT_FUNCS[name]
        params = cfg.copy()
        params.pop("prob", None)

        # Add sampling rate if needed
        if "fs" in func.__code__.co_varnames:
            params["fs"] = fs

        x = func(x, **params)
    return x

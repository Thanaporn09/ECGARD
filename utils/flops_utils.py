import torch
from thop import profile

def estimate_flops(model, input_shape=(1, 1, 512), device="cuda"):
    """
    Universal FLOPs estimator for ECG framework.
    Works with diffusion, GAN, and normal CNN models.
    """
    dummy_input = torch.randn(input_shape).to(device)

    # ✅ ensure we have an nn.Module, not a function
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model)}")

    try:
        # --- Custom dummy forward if available ---
        if hasattr(model, "forward_for_flops"):
            flops, params = profile(model, inputs=(dummy_input,), verbose=False,
                                    custom_ops={"forward": model.forward_for_flops})

        # --- Diffusion-like models (ConditionalModel / DeScoD_ECG) ---
        elif model.__class__.__name__ in ["ConditionalModel", "DeScoD_ECG"]:
            cond = dummy_input.clone()
            noise_scale = torch.ones(dummy_input.size(0), 1).to(device)
            flops, params = profile(model, inputs=(dummy_input, cond, noise_scale), verbose=False)

        # --- GAN generator ---
        elif hasattr(model, "generator"):
            flops, params = profile(model.generator, inputs=(dummy_input,), verbose=False)

        # --- Normal single-input model ---
        else:
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    except Exception as e:
        print(f"⚠️ FLOPs estimation skipped (error: {e})")
        flops = 0
        params = sum(p.numel() for p in model.parameters())

    return flops, params

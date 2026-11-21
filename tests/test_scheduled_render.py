from contextlib import contextmanager

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, ZImageTransformer2DModel

from embedded_zimage_shared.scheduled_render import render


def test_tiny_native_scheduler_uses_each_gain_and_preserves_constant_output():
    pytest.importorskip("transformers")
    from diffusers import ZImagePipeline

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        transformer = ZImageTransformer2DModel(
            all_patch_size=(2,), all_f_patch_size=(1,), in_channels=4,
            dim=32, n_layers=1, n_refiner_layers=1, n_heads=4, n_kv_heads=2,
            cap_feat_dim=16, axes_dims=[4, 2, 2], axes_lens=[128, 128, 128],
        ).eval().requires_grad_(False)
        latents = torch.randn(1, 4, 4, 4)
        embedding = torch.randn(3, 16)

    pipe = ZImagePipeline(
        vae=None, text_encoder=None, tokenizer=None, transformer=transformer,
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    pipe.set_progress_bar_config(disable=True)
    strength = 0.0
    observed = []

    @contextmanager
    def scaled(value):
        nonlocal strength
        previous, strength = strength, value
        observed.append(value)
        try:
            yield
        finally:
            strength = previous

    def add_delta(_module, _args, output):
        return ([value + strength * 0.125 for value in output[0]],)

    handle = transformer.register_forward_hook(add_delta)
    inputs = dict(
        prompt_embeds=[embedding], guidance_scale=0.0, height=32, width=32,
        num_inference_steps=3, output_type="latent", latents=latents,
    )
    try:
        with scaled(0.5):
            original = pipe(**inputs)
        observed.clear()
        constant, trace = render(pipe, scale_adapter=scaled, strength=0.5, step_gains=[1, 1, 1], **inputs)
        assert torch.equal(constant.images, original.images)
        assert trace["effective_strengths"] == observed == [0.5] * 3
        assert trace["native_timesteps"] == pipe.scheduler.timesteps.tolist()
        observed.clear()
        varied, trace = render(pipe, scale_adapter=scaled, strength=0.5, step_gains=[2, 1, 0], **inputs)
        assert trace["effective_strengths"] == observed == [1.0, 0.5, 0.0]
        # The schedule has a different sigma-weighted dose even with equal step-count sums.
        assert not torch.equal(varied.images, original.images)
        assert strength == 0.0
    finally:
        handle.remove()

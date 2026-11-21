from hashlib import sha256
from itertools import pairwise
import os
from zipfile import ZipFile

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, ZImageTransformer2DModel

from embedded_zimage_shared.activation_gate import CORE_MODULE, LatentGatePlan
from embedded_zimage_shared.graphs import ZImageLoRA
from embedded_zimage_shared.scheduled_render import render


@pytest.fixture
def core_wheel():
    path = os.environ.get("ZIMAGE_STEERING_CORE_WHEEL")
    if not path:
        pytest.skip("Set ZIMAGE_STEERING_CORE_WHEEL to the wheel built from shared core PR #269")
    return path


@pytest.fixture
def native():
    pytest.importorskip("transformers")
    from diffusers import ZImagePipeline

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        transformer = ZImageTransformer2DModel(
            all_patch_size=(2,), all_f_patch_size=(1,), in_channels=4,
            dim=32, n_layers=1, n_refiner_layers=1, n_heads=4, n_kv_heads=2,
            cap_feat_dim=16, axes_dims=[4, 2, 2], axes_lens=[128, 128, 128],
        ).eval().requires_grad_(False)
        network = ZImageLoRA(transformer)
        with torch.no_grad():
            for projection in network.projections.values():
                projection.graph["up"].weight.normal_(std=0.02)
        inputs = dict(
            prompt_embeds=[torch.randn(3, 16)], negative_prompt_embeds=[torch.randn(3, 16)],
            guidance_scale=2.0, height=32, width=32, num_inference_steps=3,
            output_type="latent", latents=torch.randn(1, 4, 4, 4),
        )
    pipeline = ZImagePipeline(
        vae=None, text_encoder=None, tokenizer=None, transformer=transformer,
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    pipeline.set_progress_bar_config(disable=True)
    try:
        yield pipeline, network, inputs
    finally:
        network.close()


def _plan(wheel, method, parameter, **options):
    return LatentGatePlan(core_wheel=wheel, method=method, parameter=parameter,
                          calibration_sha256=sha256(b"tiny native fixture").hexdigest(), **options)


def test_cfp_reads_live_conditional_trajectory_and_replays(native, core_wheel):
    pipeline, network, inputs = native
    observed = []
    direction = [1.0, 0.0, 0.0, 0.0]

    def record(_module, args):
        # The first copy belongs to the positive branch and both branches
        # share a native Euler state before the guided update.
        latents = args[0]
        assert len(latents) == 2 and torch.equal(latents[0], latents[1])
        observed.append(float(latents[0][0].float().mean()))

    handle = pipeline.transformer.register_forward_pre_hook(record)
    try:
        with network.scaled(0.0):
            pipeline(**inputs)
        threshold = (min(observed) + max(observed)) / 2
        observed.clear()
        plan = _plan(core_wheel, "cfp", threshold, direction=direction)
        gated, trace = render(pipeline, scale_adapter=network.scaled, gate_plan=plan,
                              strength=0.5, **inputs)
        decisions = trace["decisions"]
        assert len(decisions) == trace["eligible"] == 3
        assert [d["projection"] for d in decisions] == observed
        assert [d["fired"] for d in decisions] == [x <= threshold for x in observed]
        assert all(d["cfg"] for d in decisions)
        assert {d["fired"] for d in decisions} == {True, False}
    finally:
        handle.remove()
    values = trace["effective_strengths"]
    assert trace["native_timesteps"] == pipeline.scheduler.timesteps.tolist()
    assert trace["sigmas"] == pipeline.scheduler.sigmas.tolist()
    assert trace["sigma_weighted_strength"] == sum(
        value * (start - end) for value, (start, end) in zip(values, pairwise(trace["sigmas"]), strict=True)
    )
    fixed, fixed_trace = render(pipeline, scale_adapter=network.scaled, strength=0.5,
                                step_gains=[value / 0.5 for value in values], **inputs)
    assert fixed_trace == {"effective_strengths": values, "native_timesteps": trace["native_timesteps"]}
    assert torch.equal(fixed.images, gated.images)
    assert network.scale == 0.0


def test_gate_recreates_clabs_and_native_off_fixed_endpoints(native, core_wheel):
    pipeline, network, inputs = native
    inputs.update(guidance_scale=0.0, negative_prompt_embeds=None)
    for probability in (0.0, 1.0):
        plan = _plan(core_wheel, "random", probability, seed=103)
        gated, trace = render(pipeline, scale_adapter=network.scaled, gate_plan=plan,
                              strength=-0.5, **inputs)
        with network.scaled(-0.5 * probability):
            ordinary = pipeline(**inputs)
        assert torch.equal(ordinary.images, gated.images)
        assert trace["gate_rate"] == probability
        assert all(d["projection"] is None and not d["cfg"] for d in trace["decisions"])

    plan = _plan(core_wheel, "clabs", 0.5, seed=103, direction=[1.0, 0.0, 0.0, 0.0],
                 clabs=dict(mean=0.0, std=1.0, epsilon=1e-6, setpoint=0.0,
                            proportional_gain=1.0, integral_gain=0.5, beta=0.5))
    initial, first = render(pipeline, scale_adapter=network.scaled, gate_plan=plan, **inputs)
    repeated, second = render(pipeline, scale_adapter=network.scaled, gate_plan=plan, **inputs)
    assert torch.equal(initial.images, repeated.images)
    assert first == second
    assert len({d["probability"] for d in first["decisions"]}) > 1


def test_native_callback_failure_restores_strength_and_hooks(native, core_wheel):
    pipeline, network, inputs = native
    plan = _plan(core_wheel, "random", 1.0, seed=103)

    def stop(*_):
        assert network.scale == 0.125
        raise LookupError("fixture stopped")

    handles = (len(pipeline.transformer._forward_pre_hooks), len(pipeline.transformer._forward_hooks))
    with network.scaled(0.125):
        with pytest.raises(LookupError, match="fixture stopped"):
            render(pipeline, scale_adapter=network.scaled, gate_plan=plan,
                   callback_on_step_end=stop, **inputs)
        assert network.scale == 0.125
    assert network.scale == 0.0
    assert handles == (len(pipeline.transformer._forward_pre_hooks), len(pipeline.transformer._forward_hooks))


def test_only_exact_shared_module_and_single_native_trajectory(native, core_wheel, tmp_path):
    pipeline, network, inputs = native
    altered = tmp_path / "altered.whl"
    with ZipFile(altered, "w") as archive:
        archive.writestr(CORE_MODULE, "raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="exact shared steering"):
        _plan(altered, "random", 1.0, seed=103)
    plan = _plan(core_wheel, "random", 1.0, seed=103)
    inputs["prompt_embeds"] *= 2
    with pytest.raises(ValueError, match="one native Z-Image trajectory"):
        render(pipeline, scale_adapter=network.scaled, gate_plan=plan, **inputs)

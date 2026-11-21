"""Per-step strength on a caller-owned native denoising pipeline."""

from contextlib import ExitStack
from itertools import pairwise
from math import isfinite
from sys import float_info

def _fixed_strengths(step_gains, strength, steps):
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != steps:
        raise ValueError("one gain per requested native denoising step is required")
    if any(type(gain) not in (float, int) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("step gains must be finite and nonnegative")
    strengths = [float(strength * gain) for gain in step_gains]
    if any(not isfinite(value) for value in strengths):
        raise ValueError("effective strengths must be finite")
    return strengths


def render(pipeline, *, scale_adapter, step_gains=None, gate_plan=None, strength=1.0,
           callback_on_step_end=None, **inputs):
    """Sample with an attached RAW adapter and return the output and step trace."""
    if type(strength) not in (int, float) or not -float_info.max <= strength <= float_info.max:
        raise ValueError("nominal strength must be finite")
    steps = inputs.get("num_inference_steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("one gain per requested native denoising step is required")
    if gate_plan is None:
        strengths = _fixed_strengths(step_gains, strength, steps)
    else:
        from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline
        from embedded_zimage_shared.activation_gate import LatentGatePlan

        if step_gains is not None or not isinstance(gate_plan, LatentGatePlan):
            raise ValueError("Choose fixed step gains or one calibrated Z-Image LatentGatePlan")
        if strength == 0:
            raise ValueError("Gated Z-Image needs nonzero nominal strength; use fixed strength zero for OFF")
        if not isinstance(pipeline, ZImagePipeline) or not isinstance(
            pipeline.scheduler, FlowMatchEulerDiscreteScheduler
        ):
            raise ValueError("Gated Z-Image requires its native image pipeline and Euler scheduler")
        if (inputs.get("num_images_per_prompt", 1) != 1
                or any(isinstance(inputs.get(key), (tuple, list)) and len(inputs[key]) != 1
                       for key in ("prompt", "prompt_embeds"))):
            raise ValueError("Gate calibration requires one native Z-Image trajectory per request")
        gate = gate_plan._new_gate()
        scheduler = pipeline.scheduler
        strengths = []

    active = ExitStack()
    native_timesteps = []
    decisions = []
    sigmas = direction = None
    evaluations = 0

    def before(_module, args):
        nonlocal evaluations, sigmas, direction
        if evaluations != len(native_timesteps) or evaluations >= steps:
            raise RuntimeError("native denoiser order differs from the requested schedule")
        if gate_plan is not None:
            expected_index = evaluations if evaluations else None
            if (pipeline.scheduler is not scheduler or scheduler.begin_index != 0
                    or scheduler.step_index != expected_index):
                raise RuntimeError("Expected one Z-Image forward per native Euler interval")
            # ZImagePipeline passes a list of [C, F, H, W] latents. If CFG is
            # active it places the conditional copy first in the same forward.
            latents = args[0]
            if (not isinstance(latents, list) or not 1 <= len(latents) <= 2
                    or (len(latents) == 2 and not pipeline.do_classifier_free_guidance)
                    or latents[0].ndim != 4):
                raise RuntimeError("Expected one native Z-Image latent and its optional CFG copy")
            latent = latents[0]
            if len(latents) == 2 and not latents[1].equal(latent):
                raise RuntimeError("Gate calibration requires one Z-Image latent trajectory per request")
            if sigmas is None:
                sigmas = scheduler.sigmas.detach().cpu().tolist()
                if len(sigmas) != steps + 1 or any(
                    not isfinite(start) or not isfinite(end) or start <= end
                    for start, end in pairwise(sigmas)
                ):
                    raise RuntimeError("Z-Image native sigma grid differs from the requested step count")
                direction = gate_plan._direction_for(latent)
            score = gate_plan._score(latent, direction, strength)
            decision = gate.step(score)
            effective = float(strength) if decision.fired else 0.0
            strengths.append(effective)
            decisions.append(dict(interval=evaluations, projection=score, probability=decision.probability,
                                  fired=decision.fired, effective_strength=effective, cfg=len(latents) == 2))
        active.enter_context(scale_adapter(strengths[evaluations]))
        evaluations += 1

    def after(_module, _args, _output):
        active.close()

    def step_end(owner, index, timestep, values):
        if owner is not pipeline or index != len(native_timesteps) or evaluations != index + 1:
            raise RuntimeError("native solver order differs from the requested schedule")
        if gate_plan is not None and (pipeline.scheduler is not scheduler or scheduler.step_index != index + 1
                                      or timestep != scheduler.timesteps[index]):
            raise RuntimeError("Z-Image did not complete the expected native Euler interval")
        native_timesteps.append(float(timestep))
        if callback_on_step_end is not None:
            return callback_on_step_end(owner, index, timestep, values)
        return values

    with ExitStack() as request:
        request.callback(pipeline.transformer.register_forward_pre_hook(before).remove)
        request.callback(pipeline.transformer.register_forward_hook(after, always_call=True).remove)
        request.callback(active.close)
        result = pipeline(callback_on_step_end=step_end, **inputs)
    if len(native_timesteps) != steps or evaluations != steps:
        raise RuntimeError("native denoising did not complete the requested schedule")
    trace = {"native_timesteps": native_timesteps, "effective_strengths": strengths}
    if gate_plan is not None:
        usage = gate.usage
        trace.update(activation_gate=gate_plan._receipt(), decisions=decisions, sigmas=sigmas,
                     fired=usage.fired, eligible=usage.eligible, gate_rate=usage.rate,
                     sigma_weighted_strength=sum(value * (start - end) for value, (start, end)
                                                 in zip(strengths, pairwise(sigmas), strict=True)))
    return result, trace

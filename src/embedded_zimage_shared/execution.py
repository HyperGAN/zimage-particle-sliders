"""Explicit CUDA execution identity; validation never initializes or seeds CUDA."""
import os
from pathlib import Path
import re
from uuid import UUID

import torch

CUDA_PROFILE = 'native-cuda-fp32-math-v1'
EXECUTION_FILE = Path(__file__)


def visible_uuid():
    value = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value):
        raise ValueError('CUDA profile requires one physical GPU UUID in CUDA_VISIBLE_DEVICES before startup')
    return 'GPU-'+str(UUID(value.removeprefix('GPU-')))


def numerical_flags():
    return dict(fp32_matmul_precision=torch.get_float32_matmul_precision(),
        nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),
        torch_tf32_cublas_override=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


REQUIRED_FLAGS = dict(fp32_matmul_precision='highest', nvidia_tf32_override=None,
    torch_tf32_cublas_override=None, deterministic=True, warn_only=False, matmul_tf32=False,
    cudnn_tf32=False, cudnn_benchmark=False, cudnn_deterministic=True,
    flash_sdp=False, efficient_sdp=False, cudnn_sdp=False,
    math_sdp=True, cublas_workspace=':4096:8')


def execution_policy(options):
    if (options.get('runtime_profile') != CUDA_PROFILE or options.get('device') != 'cuda:0'
            or options.get('base_precision', 'float32') != 'float32'):
        raise ValueError('select the explicit native CUDA FP32 math profile and device=cuda:0')
    physical_uuid = visible_uuid()
    flags = numerical_flags()
    if flags != REQUIRED_FLAGS:
        raise ValueError('CUDA profile requires caller-set strict deterministic/math flags')
    if type(options.get('gradient_checkpointing', False)) is not bool:
        raise ValueError('gradient_checkpointing must be an explicit boolean')
    return dict(profile=CUDA_PROFILE, device='cuda:0', physical_uuid=physical_uuid,
        base_precision='float32', torch=str(torch.__version__), cuda=torch.version.cuda,
        flags=flags, gradient_checkpointing=options.get('gradient_checkpointing', False))


def configure_execution_profile():
    """Explicit CLI startup action. API callers may set the same flags themselves."""
    visible_uuid()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting the CUDA command')
    if any(name in os.environ for name in ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')):
        raise ValueError('unset TF32 force-on environment overrides before CUDA startup')
    if torch.cuda.is_initialized():
        raise ValueError('configure the CUDA profile before CUDA initialization')
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

"""Separate source-bound CUDA FP32 response cohort. Production quality is unqualified.

The historical CPU backend/rows/checkpoints keep their original immutable bytes.
Callers own GPU admission, physical locks, memory budget and numerical startup.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from uuid import UUID

import torch
from diffusers import ZImageTransformer2DModel
from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor
from particle_sliders.field_critic import seed_initialization_rng
from particle_sliders.fibo_hndl import HNDLFieldCritic
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Binding, native_cleanup, _restore_all
from particle_sliders.pg094 import ConditionalPG094
from particle_sliders.pg09_backends import ResponseRow
from particle_sliders.pg09_images import ImageResponseAdapter, image_condition_features
from particle_sliders.response import ResponseBackend, ResponseCapabilities
from . import backend as cpu
from .execution import CUDA_PROFILE, EXECUTION_FILE, execution_policy, visible_uuid
from .graphs import ZImageLoRA, targets
from .learning_rates import recipe_overrides

NAME = 'embedded-zimage-cuda-fp32'
COHORT = 'embedded-zimage-adapter-gan-pg094-hg-hndl010-cuda-v1'
PROTOCOL = 'embedded-zimage-native-4d-raw-velocity-cuda-v1'
BACKEND_FILE = Path(__file__)
file_sha256 = cpu.file_sha256
PRODUCT_HELPERS = ('backend.py', 'graphs.py', 'learning_rates.py', 'source_profile.json')


def helper_sources():
    return {name:file_sha256(BACKEND_FILE.with_name(name)) for name in PRODUCT_HELPERS}


def bundle_identity(directory, *, options):
    policy=execution_policy(options)
    return dict(cpu.bundle_identity(directory), execution_policy=policy,
        execution_source_sha256=file_sha256(EXECUTION_FILE),
        cuda_backend_source_sha256=file_sha256(BACKEND_FILE),
        product_helper_sources=helper_sources(),
        qualification='explicit native CUDA FP32 source/numerical profile; pretrained quality/resources unqualified')


def initialization_identity(seed, policy):
    if type(seed) is not int or not 0 <= seed < 2**63 - 100:
        raise ValueError('invalid initialization seed')
    return dict(policy='scoped-cpu-owned-cuda-initialization-streams-v1', seed=seed,
        device=policy['device'],
        streams=dict(adapter=seed, critic=seed+1, native_loader=seed+2, preparation=seed+3))


def validate_initialization(identity, policy):
    if not isinstance(identity, dict) or identity != initialization_identity(identity.get('seed'), policy):
        raise ValueError('CUDA initialization policy/seed/streams/owned device differ')


def admit_owned_device(policy):
    # CUDAVISIBLE is exactly one physical UUID; logical0 cannot address another GPU.
    # No manual_seed_all/current-device discovery or unrelated generator snapshots.
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('CUDA profile requires exactly its one visible admitted GPU')
    properties = torch.cuda.get_device_properties(0)
    actual = getattr(properties, 'uuid', None)
    try:
        actual_uuid = UUID(str(actual).removeprefix('GPU-'))
        expected_uuid = UUID(policy['physical_uuid'].removeprefix('GPU-'))
        if expected_uuid != UUID(visible_uuid().removeprefix('GPU-')):
            raise ValueError('visible physical CUDA UUID differs from bound execution identity')
    except (ValueError, AttributeError, TypeError, KeyError) as exc:
        raise ValueError('invalid physical CUDA UUID reported by admitted device') from exc
    if actual_uuid != expected_uuid:
        raise ValueError('visible physical CUDA UUID differs from execution identity')


@contextmanager
def initialization_scope(identity, component, policy):
    validate_initialization(identity, policy)
    if component not in identity['streams']:
        raise ValueError('unknown CUDA initialization stream')
    with torch.random.fork_rng(devices=[0]):
        seed_initialization_rng(identity['streams'][component], [0])
        yield


def capabilities(options):
    return ResponseCapabilities('adapter_gan', 'image', 'raw_flow_velocity', 'continuous_field',
        cpu._shape(options.get('response_shape', [1,4,4,4])), '0.9.4', 'release-0.9.4',
        base_precisions=('float32',), qualification='separate CUDA FP32 execution profile; pretrained quality/resources unqualified')


def validate(options):
    allowed = {'model_dir','model_identity','rows','rows_sha256','response_shape','steps','batch','seed','preset',
        'ema_decay','base_precision','pg_version','pg_source_profile','penalty_input_domain','dummy',
        'lr','d_lr_mult','prior_lr_mult','runtime_profile','device','gradient_checkpointing'}
    if set(options)-allowed:
        raise ValueError('unknown CUDA Z-Image option')
    recipe_overrides(**{name:options.get(name) for name in ('lr','d_lr_mult','prior_lr_mult')})
    if 'dummy' in options and type(options['dummy']) is not bool:
        raise ValueError('dummy must be an actual boolean fixture marker')
    execution_policy(options)  # No model/file/CUDA allocation before explicit policy.
    cpu._core_gate()
    shape = cpu._shape(options.get('response_shape'))
    if options.get('preset', 'bispf_optimized') not in ('bispf','bispf_optimized'):
        raise ValueError('choose an existing shared BISPF preset')
    actual = bundle_identity(options['model_dir'], options=options)
    if actual != options.get('model_identity') or shape[1] != actual['config']['in_channels']:
        raise ValueError('CUDA native model/source/geometry identity differs before attachment')
    digest = options.get('rows_sha256')
    if (not isinstance(options.get('rows'), str) or not isinstance(digest, str) or len(digest)!=64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('explicit frozen rows and complete rows_sha256 required')


def read_rows(options):
    with Path(options['rows']).open('rb') as stream:
        if cpu._stream_sha256(stream) != options.get('rows_sha256'):
            raise ValueError('complete frozen CUDA rows file hash differs before attachment')
        stream.seek(0); raw = torch.load(stream, map_location='cpu', weights_only=True)
        stream.seek(0)
        if cpu._stream_sha256(stream) != options['rows_sha256']:
            raise ValueError('frozen CUDA rows changed while reading')
    if (not isinstance(raw, dict) or set(raw)!={'schema','model_identity','rows'} or raw['schema']!=PROTOCOL
            or raw['model_identity']!=options['model_identity']):
        raise ValueError('frozen CUDA rows model/protocol/execution identity differs; no CPU cohort rebinding')
    if not isinstance(raw['rows'], list) or len(raw['rows'])<2:
        raise ValueError('at least two frozen native CUDA rows required')
    result=[]
    for item in raw['rows']:
        if not isinstance(item, dict) or set(item)!={'neutral','positive','negative','latent','timestep'}:
            raise ValueError('rows require three native encodings, latent and raw timestep')
        row=ResponseRow(**item, guidance=1.)
        for condition in (row.neutral,row.positive,row.negative):
            cpu.check_request(condition,row.latent,row.timestep,options['response_shape'],options['model_identity'])
        result.append(row)
    return result


def _model_gate(model, identity, policy):
    if type(model) is not ZImageTransformer2DModel:
        raise ValueError('borrowed native CUDA model type differs')
    for module in model.modules():
        if ('forward' in vars(module) or getattr(module,'_compiled_call_impl',None) is not None
                or '_hf_hook' in vars(module) or '_old_forward' in vars(module)):
            raise ValueError('native instance forward/compiled/offload modifications are unowned')
        if (module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
                or module._backward_pre_hooks or module._is_full_backward_hook is not None):
            raise ValueError('native model already has hooks; overlapping ownership refused')
    if any(str(p.device)!=policy['device'] or p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('native model must already occupy the explicit CUDA FP32 device')
    if cpu.digest_state(model.state_dict())!=identity['state_sha256']:
        raise ValueError('borrowed CUDA model tensor identity differs')
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('native CUDA model must contain finite FP32 weights')
    a={k:v for k,v in model.config.items() if not k.startswith('_')}
    b={k:v for k,v in identity['config'].items() if not k.startswith('_')}
    if json.loads(json.dumps(a))!=b:
        raise ValueError('borrowed CUDA model configuration differs')
    if model.is_gradient_checkpointing:
        raise ValueError('borrow an initially uncheckpointed native model; profile owns enablement/cleanup')
    for module in model.modules():
        processor=getattr(module,'processor',None)
        if processor is not None and (type(processor) is not ZSingleStreamAttnProcessor
                or processor._attention_backend is not None or processor._parallel_config is not None):
            raise ValueError('only the exact native attention processor under the math execution profile is admitted')
    targets(model)


def preflight(options, request, staged):
    read_rows(options)
    if staged.get('model') is not None:
        _model_gate(staged['model'],options['model_identity'],execution_policy(options))


class CUDAResponseAdapter(ImageResponseAdapter):
    def __init__(self,model,network,*,identity,options):
        self.execution_options={key:options[key] for key in ('runtime_profile','device')}
        self.execution_options['base_precision']='float32'
        self.execution_options['gradient_checkpointing']=options.get('gradient_checkpointing',False)
        super().__init__('zimage',model,network,model_identity=identity)

    def guard_policy(self):
        policy=execution_policy(self.execution_options)
        if bool(self.transformer.is_gradient_checkpointing)!=policy['gradient_checkpointing']:
            raise ValueError('native checkpoint mode differs from the CUDA execution profile')
        if policy!=self.model_identity['execution_policy']:
            raise ValueError('CUDA execution policy changed before native operation')

    def velocity(self,condition,latent,timestep,*,scale=1.):
        self.guard_policy()
        return super().velocity(condition,latent,timestep,scale=scale)


def observe(adapter):
    native_observe=cpu._observe(adapter)
    @contextmanager
    def context():
        adapter.guard_policy()
        try:
            with native_observe():yield
        except BaseException as primary:
            _restore_all([adapter.guard_policy],primary=primary)
            raise
        else:adapter.guard_policy()
    return context


def _native(options, identity, model=None, *, initialization):
    policy=execution_policy(options)
    validate_initialization(initialization,policy)
    admit_owned_device(policy)
    if model is None:
        with initialization_scope(initialization,'native_loader',policy):
            model=ZImageTransformer2DModel.from_pretrained(options['model_dir'],local_files_only=True,
                torch_dtype=torch.float32,low_cpu_mem_usage=False).to(policy['device'])
    if any(file_sha256(Path(options['model_dir'])/name)!=digest for name,digest in identity['files'].items()):
        raise ValueError('native file identity changed before CUDA attachment')
    _model_gate(model,identity,policy)
    restore=native_cleanup(model)
    checkpoint_state=[(m,{k:(k in vars(m),vars(m).get(k)) for k in ('gradient_checkpointing','_gradient_checkpointing_func')}) for m in model.modules()]
    def restore_checkpointing():
        for module,values in checkpoint_state:
            for key,(owned,value) in values.items():
                if owned:setattr(module,key,value)
                else:vars(module).pop(key,None)
    network=None
    def close():
        _restore_all(([network.close] if network is not None else [])+[restore_checkpointing,restore])
    try:
        model.requires_grad_(False).eval()
        if policy['gradient_checkpointing']:model.enable_gradient_checkpointing()
        with initialization_scope(initialization,'adapter',policy):
            network=ZImageLoRA(model).to(policy['device'])
        adapter=CUDAResponseAdapter(model,network,identity=identity,options=options)
        return adapter,close
    except BaseException as primary:
        _restore_all([close],primary=primary);raise


def stage_row(row,device):
    def condition(values):return {k:v.to(device) for k,v in values.items()}
    return replace(row,latent=row.latent.to(device),timestep=row.timestep.to(device),
        neutral=condition(row.neutral),positive=condition(row.positive),negative=condition(row.negative))


def prepare(options, *, model=None):
    recipe_overrides(**{name:options.get(name) for name in ('lr','d_lr_mult','prior_lr_mult')})
    policy=execution_policy(options)
    initialization=initialization_identity(options.get('seed',7),policy)
    admit_owned_device(policy)
    with initialization_scope(initialization,'preparation',policy):
        frozen=read_rows(options)
        adapter,close=_native(options,options['model_identity'],model,initialization=initialization)
        try:
            rows=[adapter.prepare(stage_row(row,policy['device'])) for row in frozen]
            features=image_condition_features(rows)
            with initialization_scope(initialization,'critic',policy):
                critic=HNDLFieldCritic(options['response_shape'][1],features.shape[1],tuple(options['response_shape'][2:])).to(policy['device'])
            game=ConditionalPG094(adapter,rows,critic,conditions=features,preset=options.get('preset','bispf_optimized'),
                total_steps=options.get('steps',4),batch_size=options.get('batch',2),seed=options.get('seed',7),
                response_domain='continuous_field',recipe_overrides=recipe_overrides(**{n:options.get(n) for n in ('lr','d_lr_mult','prior_lr_mult')}))
            configuration=dict(model_identity=options['model_identity'],response_shape=options['response_shape'],
                initialization=initialization,runtime_profile=CUDA_PROFILE,device='cuda:0',base_precision='float32',
                gradient_checkpointing=policy['gradient_checkpointing'])
            spec=INFERENCE_BACKEND.specification(adapter,model_identity=options['model_identity'],protocol=PROTOCOL,configuration=configuration)
            return Binding(game,close,inference=spec,before_update=lambda state:adapter.guard_policy(),observation=observe(adapter))
        except BaseException as primary:
            _restore_all([close],primary=primary);raise


def validate_inference(options,spec):
    if set(options)!={'model_dir'}:
        raise ValueError('CUDA native-only inference accepts just a staged model_dir')
    cpu._core_gate()
    config=spec.configuration
    policy=execution_policy(config)
    identity=config['model_identity']
    if (identity.get('execution_source_sha256')!=file_sha256(EXECUTION_FILE)
            or identity.get('cuda_backend_source_sha256')!=file_sha256(BACKEND_FILE)
            or identity.get('product_helper_sources')!=helper_sources()):
        raise ValueError('CUDA execution source differs before allocation')
    if (spec.identity['protocol']!=PROTOCOL or config['model_identity']!=bundle_identity(options['model_dir'],options=config)):
        raise ValueError('CUDA inference source/model/execution identity differs before allocation; no cohort rebinding')
    cpu._shape(config['response_shape'])
    validate_initialization(config.get('initialization'),policy)
    if spec.identity['sources'].get('embedded_zimage_shared.graphs')!=file_sha256(Path(__file__).with_name('graphs.py')):
        raise ValueError('saved native CUDA graph source differs before attachment')


def prepare_inference(options,spec):
    config=spec.configuration;policy=execution_policy(config)
    initialization=config['initialization'];admit_owned_device(policy)
    with initialization_scope(initialization,'preparation',policy):
        adapter,close=_native(dict(options,**{k:config[k] for k in ('runtime_profile','device','base_precision','gradient_checkpointing')}),config['model_identity'],initialization=initialization)
        try:
            def response(*,condition,latent,timestep,strength,seed):
                execution_policy(config)
                cpu.check_request(condition,latent,timestep,config['response_shape'],config['model_identity'])
                return adapter.velocity({k:v.to(policy['device']) for k,v in condition.items()},latent.to(policy['device']),timestep.to(policy['device']),scale=strength).detach().clone()
            return InferenceBinding(adapter.network,native_identity(adapter,model_identity=config['model_identity'],protocol=PROTOCOL),
                close,response=response,observation=observe(adapter))
        except BaseException as primary:
            _restore_all([close],primary=primary);raise


BACKEND=ResponseBackend(NAME,capabilities({'response_shape':[1,4,4,4]}),prepare,validate,
    configure_capabilities=capabilities,preflight=preflight)
INFERENCE_BACKEND=InferenceBackend(NAME,capabilities({'response_shape':[1,4,4,4]}),prepare_inference,
    validate_inference,configure_capabilities=capabilities)

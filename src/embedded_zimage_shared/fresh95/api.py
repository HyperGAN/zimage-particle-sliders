"""Additional explicit cached fresh095 owner; original entrypoints stay exact."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib,torch
from particle_sliders.pg095 import ConditionalPG095
from particle_sliders.pg09_images import ImageResponseAdapter
from particle_sliders.fibo_hndl import HNDLFieldCritic
from embedded_zimage_shared import backend as cpu,cuda_backend as gpu
from embedded_zimage_shared.learning_rates import recipe_overrides
from . import PROFILE,ownership,policy
from .cache import load_cache,to_device
from .util import canonical,equal,new_directory,read_packet,save_packet,write_json

FULL_SCHEMA='ZImage-explicit-cached-fresh95-v1-FULL'
RAW_SCHEMA='ZImage-explicit-cached-fresh95-v1-RAW'

def configuration(value):
    if not isinstance(value,dict):raise ValueError('explicit new-profile configuration required')
    allowed={'runtime','model_dir','model_identity','response_shape','steps','batch','seed','preset','ema_decay',
        'base_precision','pg_version','pg_source_profile','penalty_input_domain','lr','d_lr_mult','prior_lr_mult',
        'runtime_profile','device','gradient_checkpointing'}
    if set(value)-allowed or value.get('runtime')not in('cpu-native-fp32','cuda-native-fp32'):
        raise ValueError('explicit original CPU or CUDA lane; old entrypoints/configs are not migrated')
    if (value.get('ema_decay','missing')is not None or value.get('base_precision')!='float32'
            or tuple(value.get(k)for k in('pg_version','pg_source_profile','penalty_input_domain'))!=('0.9.5','release-0.9.5','continuous')):
        raise ValueError('explicit RAW/noEMA/native-FP32/current095 continuous recipe required')
    for key,low,high in(('steps',1,10000),('batch',2,512),('seed',0,2**63-101)):
        if type(value.get(key))is not int or not low<=value[key]<=high:raise ValueError('invalid '+key)
    if value.get('preset')not in('bispf','bispf_optimized')or type(value.get('gradient_checkpointing'))is not bool:
        raise ValueError('explicit original preset/checkpoint choice required')
    cpu._core_gate();shape=cpu._shape(value.get('response_shape'))
    recipe_overrides(**{k:value.get(k)for k in('lr','d_lr_mult','prior_lr_mult')})
    cuda=value['runtime']=='cuda-native-fp32'
    if cuda:
        gpu.execution_policy(value)  # Caller-set numerical/physical policy, no library startup mutation.
        actual=gpu.bundle_identity(value['model_dir'],options=value)
    else:
        if value.get('device')!='cpu'or 'runtime_profile'in value or value['gradient_checkpointing']:
            raise ValueError('original CPU lane stays CPU/no native checkpoint enablement')
        actual=cpu.bundle_identity(value['model_dir'])
    if actual!=value.get('model_identity')or shape[1]!=actual['config']['in_channels']:
        raise ValueError('whole original native bundle/source/geometry identity differs')
    canonical(value);return deepcopy(value)

def intent(mode,resume,cohort,ema_decay):
    if mode not in('fresh','resume')or not isinstance(cohort,str)or not cohort.strip():raise ValueError('explicit fresh/resume and cohort required')
    if(mode=='fresh'and resume is not None)or(mode=='resume'and resume is None):raise ValueError('fresh refuses state; resume requires exact new FULL')
    if ema_decay is not None:raise ValueError('weight EMA refused; RAW only')

def constructor_identity(config):
    return(gpu.initialization_identity(config['seed'],gpu.execution_policy(config))
        if config['runtime']=='cuda-native-fp32'else cpu.initialization_identity(config['seed']))

def constructor_scope(config,identity,component):
    return(gpu.initialization_scope(identity,component,gpu.execution_policy(config))
        if config['runtime']=='cuda-native-fp32'else cpu.initialization_scope(identity,component))

def model_gate(model,config):
    if config['runtime']=='cuda-native-fp32':
        execution=gpu.execution_policy(config);gpu.admit_owned_device(execution)
        gpu._model_gate(model,config['model_identity'],execution)
    else:
        cpu._model_gate(model,config['model_identity'])
        if any(getattr(module,'gradient_checkpointing',False)for module in model.modules()):
            raise ValueError('new CPU lane requires original uncheckpointed native model')
    if any('forward'in vars(module)or getattr(module,'_compiled_call_impl',None)is not None
            or '_hf_hook'in vars(module)or '_old_forward'in vars(module)for module in model.modules()):
        raise ValueError('unowned native instance forward/compile/offload modification refused')

class ReceiptBackend:
    def __init__(self,native,receipt,sources,cache_identity,constructor):
        self.backend=native;self.network=native.network
        self.receipt=deepcopy(receipt);self.sources=deepcopy(sources);self.cache=deepcopy(cache_identity)
        self.constructor=deepcopy(constructor)
    def __getattr__(self,name):return getattr(self.backend,name)
    def metadata(self):
        return dict(self.backend.metadata(),fresh_initialization=dict(profile=PROFILE,receipt=deepcopy(self.receipt),sources=deepcopy(self.sources)),
            original_constructor_streams=deepcopy(self.constructor),completed_native_cache=deepcopy(self.cache),
            particle_bank=None,old_request_only_prepare_or_teacher_collector_called=False)

class NativeOwner(ownership.Owner):
    @contextmanager
    def observation(self,seed):
        from particle_sliders.lifecycle import _network_observation
        self.require_open()
        devices=[0]if self.cuda else[]
        native_observe=gpu.observe(self.native)if self.cuda else cpu._observe(self.native)
        with _network_observation(self.native.network,seed,devices=devices),native_observe():yield
    def response(self,*,condition,latent,timestep,strength=1.,seed=7):
        (gpu.capabilities(self.config)if self.cuda else cpu.capabilities(self.config)).check('response',strength=strength,precision='float32')
        cpu.check_request(condition,latent,timestep,self.config['response_shape'],self.config['model_identity'])
        with self.observation(seed),torch.no_grad():
            device=self.config['device']
            return self.native.velocity({k:v.to(device)for k,v in condition.items()},latent.to(device),timestep.to(device),scale=strength).detach().clone()
    def _components(self,request,component_factory):
        self.model=component_factory(deepcopy(request));model_gate(self.model,self.config)
        self.attachment=ownership.Attachment(self.model)
        self.network=self.attachment.attach(self.constructor,self.config,self.cuda)
        try:
            self.native=(gpu.CUDAResponseAdapter(self.model,self.network,identity=self.config['model_identity'],options=self.config)
                if self.cuda else ImageResponseAdapter('zimage',self.model,self.network,model_identity=self.config['model_identity']))
        finally:self.attachment._capture_owned_checkpoint_state()

class Training(NativeOwner):
    def __init__(self,config,cache,directory,*,mode,cohort_id,component_factory,owner_admission,resume=None,ema_decay=None):
        super().__init__();intent(mode,resume,cohort_id,ema_decay)
        if not callable(component_factory)or not callable(owner_admission):raise ValueError('actual owner factory/admission required')
        self.directory=Path(directory).absolute()
        if self.directory.exists():raise FileExistsError('new immutable owner directory required')
        self.config=configuration(config);self.runtime_identity=ownership.guard();self.sources=ownership.sources()
        self.cuda=self.config['runtime']=='cuda-native-fp32';self.cohort_id=cohort_id
        self.constructor=constructor_identity(self.config)
        self.options=dict(config=self.config,original_constructor_streams=self.constructor)
        self.construction=policy.Construction(mode,cohort_id,hashlib.sha256(canonical(self.options).encode()).hexdigest())
        packet=read_packet(resume)if mode=='resume'else None
        if packet is not None:
            keys={'schema','profile','cohort_id','options','sources','initialization','cache_identity','game','ema_decay'}
            if (not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=FULL_SCHEMA
                    or packet['profile']!=PROFILE or packet['cohort_id']!=cohort_id
                    or packet['options']!=self.options or packet['sources']!=self.sources or packet['ema_decay']is not None):
                raise ValueError('exact typed new-profile FULL required; no old cohort reset/migration')
            policy.validate_receipt(packet['initialization'],self.construction,self.options,self.sources)
        prepared,conditions,self.cache_identity=load_cache(cache,config=self.config)
        if packet is not None and packet['cache_identity']!=self.cache_identity:raise ValueError('original completed cache differs')
        request=dict(operation=mode,profile=PROFILE,cohort_id=cohort_id,options=deepcopy(self.options),
            completed_cache=deepcopy(self.cache_identity),new_teacher_or_trajectory_calls=0,
            fresh_PG_initialization_calls=int(mode=='fresh'),particle_bank=None,ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('owner did not admit/authenticate new operation')
        try:
            with constructor_scope(self.config,self.constructor,'preparation'):
                self._components(request,component_factory)
                with constructor_scope(self.config,self.constructor,'critic'):
                    self.critic=HNDLFieldCritic(self.config['response_shape'][1],conditions.shape[1],tuple(self.config['response_shape'][2:])).to(self.config['device'])
                self.initialization=(policy.initialize(self.network,self.critic,self.construction,self.options,self.sources)
                    if mode=='fresh'else deepcopy(packet['initialization']))
                self.backend=ReceiptBackend(self.native,self.initialization,self.sources,self.cache_identity,self.constructor)
                # The scoped public PG receipt exists before the first optimizer.
                self.game=ConditionalPG095(self.backend,to_device(prepared,self.config['device']),self.critic,
                    conditions=conditions.to(self.config['device']),particles=None,response_scale=1.0,
                    preset=self.config['preset'],total_steps=self.config['steps'],batch_size=self.config['batch'],seed=self.config['seed'],
                    response_domain='continuous_field',recipe_overrides=recipe_overrides(**{k:self.config.get(k)for k in('lr','d_lr_mult','prior_lr_mult')}))
                if self.game.prior is not None:raise RuntimeError('ZImage remains adapter-only, no bank')
                if packet is not None:
                    self.game.load_state_dict(packet['game'])
                    if not equal(self.game.state_dict(),packet['game']):raise RuntimeError('whole game/optimizers/cursors/RNG restore differs')
            self.inherited_steps=self.game.completed_steps
            new_directory(self.directory);write_json(self.directory/'initialization.json',self.initialization)
            self.entry_FULL=self.save(self.directory/f'FULL-{self.inherited_steps:06d}.pt')
            self.entry_RAW=self.export(self.directory/f'RAW-{self.inherited_steps:06d}.pt')
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('ZImage entry and cleanup failed',[primary,cleanup])
            raise
    def update(self):self.require_open();return self.game.step()
    def save(self,path):
        self.require_open()
        return save_packet(path,dict(schema=FULL_SCHEMA,profile=PROFILE,cohort_id=self.cohort_id,
            options=deepcopy(self.options),sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),
            cache_identity=deepcopy(self.cache_identity),game=self.game.state_dict(),ema_decay=None))
    def export(self,path):
        self.require_open();state=self.game.export_state_dict()
        return save_packet(path,dict(schema=RAW_SCHEMA,profile=PROFILE,cohort_id=self.cohort_id,
            options=deepcopy(self.options),sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),
            cache_identity=deepcopy(self.cache_identity),model_identity=deepcopy(self.config['model_identity']),
            network=state['network'],completed_steps=self.game.completed_steps,selection='raw',ema_decay=None))

class Inference(NativeOwner):
    def __init__(self,raw,*,component_factory,owner_admission):
        super().__init__();packet=read_packet(raw)
        keys={'schema','profile','cohort_id','options','sources','initialization','cache_identity','model_identity','network','completed_steps','selection','ema_decay'}
        if (not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=RAW_SCHEMA or packet['profile']!=PROFILE
                or packet['selection']!='raw'or packet['ema_decay']is not None or type(packet['completed_steps'])is not int
                or packet['completed_steps']<0 or not callable(component_factory)or not callable(owner_admission)):
            raise ValueError('exact typed new RAW and owner required; no old migration')
        self.config=configuration(packet['options']['config']);self.runtime_identity=ownership.guard();self.sources=ownership.sources()
        self.cuda=self.config['runtime']=='cuda-native-fp32';self.constructor=constructor_identity(self.config)
        self.options=dict(config=self.config,original_constructor_streams=self.constructor)
        if (packet['options']!=self.options or packet['sources']!=self.sources
                or packet['model_identity']!=self.config['model_identity']
                or packet['cache_identity']['teacher_identity']['model_identity']!=self.config['model_identity']
                or packet['completed_steps']>self.config['steps']):raise ValueError('exact native/configuration/Source/update identity required')
        construction=policy.Construction('inference',packet['cohort_id'],hashlib.sha256(canonical(self.options).encode()).hexdigest())
        policy.validate_receipt(packet['initialization'],construction,self.options,self.sources)
        self.initialization=deepcopy(packet['initialization'])
        request=dict(operation='inference',profile=PROFILE,cohort_id=packet['cohort_id'],options=deepcopy(self.options),
            completed_cache_metadata_only=deepcopy(packet['cache_identity']),new_teacher_or_trajectory_calls=0,
            fresh_PG_initialization_calls=0,particle_bank=None,ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('owner did not admit native RAW inference')
        try:
            with constructor_scope(self.config,self.constructor,'preparation'):
                self._components(request,component_factory)
                self.network.load_state_dict(packet['network'],strict=True)
                if not equal(self.network.state_dict(),packet['network']):raise RuntimeError('strict native RAW restore differs')
            self.completed_steps=packet['completed_steps']
            # No cache load, critic/game/optimizer, teacher or public PG initialization.
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('ZImage inference and cleanup failed',[primary,cleanup])
            raise

"""CPU source/policy/artifact proofs. These never claim native CUDA qualification."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from diffusers import ZImageTransformer2DModel

from embedded_zimage_shared import backend as cpu
from embedded_zimage_shared import cuda_backend as b
from embedded_zimage_shared import execution as e
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference

UUID='GPU-72c1b506-891d-b8bc-b353-e020585e1c47'


def strict_metadata(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',UUID)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setattr(e,'numerical_flags',lambda:dict(e.REQUIRED_FLAGS))
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('CPU policy initialized CUDA'))
    return dict(runtime_profile=e.CUDA_PROFILE,device='cuda:0',base_precision='float32')


def test_policy_checks_caller_flags_and_physical_uuid_without_cuda_init(monkeypatch):
    options=strict_metadata(monkeypatch)
    result=e.execution_policy(options)
    assert result['device']=='cuda:0' and result['flags']==e.REQUIRED_FLAGS
    assert 'physical_uuid' not in result  # Admission/audit UUID is not a portable weight key.
    monkeypatch.setattr(e,'numerical_flags',lambda:dict(e.REQUIRED_FLAGS,flash_sdp=True))
    with pytest.raises(ValueError,match='caller-set strict'):e.execution_policy(options)


@pytest.mark.parametrize('visible',['','0','0,1',UUID+',GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'])
def test_ambiguous_visible_devices_refuse_without_cuda(monkeypatch,visible):
    options=strict_metadata(monkeypatch);monkeypatch.setenv('CUDA_VISIBLE_DEVICES',visible)
    with pytest.raises(ValueError,match='one physical GPU UUID'):e.execution_policy(options)


@pytest.mark.parametrize('change',[{'runtime_profile':'cpu-fp32-v1'},{'device':'cuda:1'},
    {'base_precision':'bfloat16'},{'gradient_checkpointing':1}])
def test_invalid_execution_options_refuse_before_file_or_model_allocation(monkeypatch,change):
    options=strict_metadata(monkeypatch);options.update(change)
    monkeypatch.setattr(cpu,'bundle_identity',lambda *args:pytest.fail('read base before policy refusal'))
    with pytest.raises(ValueError):b.bundle_identity('missing-model',options=options)


def test_prior_rate_rejected_before_cuda_or_native_paths(monkeypatch):
    monkeypatch.setattr(b,'execution_policy',lambda *args:pytest.fail('CUDA policy touched for invalid prior rate'))
    with pytest.raises(ValueError,match='no particle bank'):b.prepare({'prior_lr_mult':1})


def test_only_owned_visible_generator_is_forked_and_seeded(monkeypatch):
    calls=[]
    @contextmanager
    def fork(*,devices):calls.append(('fork',devices));yield
    monkeypatch.setattr(torch.random,'fork_rng',fork)
    monkeypatch.setattr(b,'seed_initialization_rng',lambda seed,devices:calls.append(('seed',seed,devices)))
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda *args:pytest.fail('seeded unrelated GPUs'))
    monkeypatch.setattr(torch.cuda,'current_device',lambda:pytest.fail('discovered a different current GPU'))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:pytest.fail('enumerated devices during initialization scope'))
    policy={'device':'cuda:0'};identity=b.initialization_identity(7,policy)
    with b.initialization_scope(identity,'critic',policy):pass
    assert calls==[('fork',[0]),('seed',8,[0])]
    with pytest.raises(ValueError,match='unknown'): 
        with b.initialization_scope(identity,'unowned',policy):pass


def test_admission_verifies_one_visible_device_and_actual_uuid(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',UUID)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda index:SimpleNamespace(uuid=UUID))
    b.admit_owned_device({'device':'cuda:0'})
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda index:SimpleNamespace(uuid='GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'))
    with pytest.raises(ValueError,match='physical CUDA UUID differs'):b.admit_owned_device({'device':'cuda:0'})


def test_explicit_cli_profile_configuration_does_not_initialize_cuda(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',UUID);monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('configuration initialized CUDA'))
    methods=[]
    monkeypatch.setattr(torch,'use_deterministic_algorithms',lambda value:methods.append(('deterministic',value)))
    for name in ('enable_flash_sdp','enable_mem_efficient_sdp','enable_cudnn_sdp','enable_math_sdp'):
        monkeypatch.setattr(torch.backends.cuda,name,lambda value,name=name:methods.append((name,value)))
    # Restore process flags after testing the explicit configurator.
    old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
    try:e.configure_execution_profile()
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic=old
    assert methods==[('deterministic',True),('enable_flash_sdp',False),('enable_mem_efficient_sdp',False),('enable_cudnn_sdp',False),('enable_math_sdp',True)]


@pytest.fixture
def cpu_software_cuda_path(tmp_path,monkeypatch):
    # Exercise real native blocks/LoRA/HNDL/session artifacts on CPU by replacing
    # only hardware admission/placement and CUDA RNG scopes. No GPU claim.
    torch.set_num_threads(1)
    policy=dict(profile=e.CUDA_PROFILE,device='cpu',base_precision='float32',
        gradient_checkpointing=False,qualification='CPU software fixture only')
    def current(options):return dict(policy,gradient_checkpointing=options.get('gradient_checkpointing',False))
    monkeypatch.setattr(b,'execution_policy',current)
    monkeypatch.setattr(b,'admit_owned_device',lambda policy:None)
    import particle_sliders.lifecycle as lifecycle
    import particle_sliders.inference as inference
    monkeypatch.setattr(lifecycle,'_rng_devices',lambda *values:[])
    monkeypatch.setattr(inference,'_rng_devices',lambda *values:[])
    @contextmanager
    def scope(identity,component,policy):
        b.validate_initialization(identity,policy)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(identity['streams'][component]);yield
    monkeypatch.setattr(b,'initialization_scope',scope)
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('CPU software fixture initialized CUDA'))
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(23)
        native=ZImageTransformer2DModel(all_patch_size=(2,),all_f_patch_size=(1,),in_channels=4,
            dim=32,n_layers=1,n_refiner_layers=1,n_heads=4,n_kv_heads=2,norm_eps=1e-5,qk_norm=True,
            cap_feat_dim=16,siglip_feat_dim=None,rope_theta=256.,t_scale=1000.,axes_dims=[4,2,2],axes_lens=[128,128,128])
        assert not native.config.get('_use_default_values',[])
        native.save_pretrained(tmp_path/'model')
        options=dict(runtime_profile=e.CUDA_PROFILE,device='cuda:0',base_precision='float32')
        identity=b.bundle_identity(tmp_path/'model',options=options)
        rows=[dict(neutral={'embeds':torch.randn(3,16)},positive={'embeds':torch.randn(4,16)},
            negative={'embeds':torch.randn(2,16)},latent=torch.randn(1,4,4,4),timestep=torch.tensor(300.+100*i)) for i in range(3)]
    torch.save(dict(schema=b.PROTOCOL,model_identity=identity,rows=rows),tmp_path/'rows.pt')
    config=SessionConfig(b.NAME,b.COHORT,model_dir=str(tmp_path/'model'),model_identity=identity,
        rows=str(tmp_path/'rows.pt'),rows_sha256=b.file_sha256(tmp_path/'rows.pt'),response_shape=[1,4,4,4],
        steps=4,batch=2,seed=7,lr=1e-4,d_lr_mult=2.,**options)
    return native,rows,config


def test_new_factory_real_updates_exact_resume_and_raw_inference_without_data(cpu_software_cuda_path,tmp_path,monkeypatch):
    native,rows,config=cpu_software_cuda_path
    with open_response(config) as full:
        initial=deepcopy(full.game.network.state_dict());full.train();expected=full.checkpoint()
        raw=full.export_inference(selection='raw');outputs={s:full.evaluate(strength=s).cpu() for s in (0.,.5,1.)}
        assert full.game.prior is None
        assert [g['lr'] for g in full.game.opt_g.param_groups]==[1e-4]
        assert all(g['lr']==2e-4 for g in full.game.opt_d.param_groups)
        assert any(not torch.equal(v,full.game.network.state_dict()[k]) for k,v in initial.items())
    with open_response(config) as split:split.train(stop_after=2);split.checkpoint(tmp_path/'split.pt')
    with open_response(config,resume=tmp_path/'split.pt') as replay:
        replay.train()
        from particle_sliders.pg09 import _hash
        assert _hash(expected)==_hash(replay.checkpoint())
    (tmp_path/'rows.pt').unlink()
    monkeypatch.setattr(b,'HNDLFieldCritic',lambda *args:pytest.fail('constructed training critic during inference'))
    request={k:rows[0][k] for k in ('latent','timestep')};request['condition']=rows[0]['neutral']
    with open_inference(raw,options={'model_dir':str(tmp_path/'model')},operation='response') as inference:
        assert not hasattr(inference,'game')
        for strength in (0.,.5,1.):assert torch.equal(inference.response(**request,strength=strength).cpu(),outputs[strength])
    assert not torch.equal(outputs[.5],outputs[1.])
    assert not torch.cuda.is_initialized()


def test_changed_execution_source_rejects_before_native_allocation(cpu_software_cuda_path,tmp_path,monkeypatch):
    _,_,config=cpu_software_cuda_path
    with open_response(config) as session:artifact=session.export_inference()
    source=tmp_path/'changed-execution.py';source.write_bytes(e.EXECUTION_FILE.read_bytes()+b'\n# different source\n')
    monkeypatch.setattr(b,'EXECUTION_FILE',source)
    monkeypatch.setattr(b,'_native',lambda *args,**kwargs:pytest.fail('allocated changed execution profile'))
    with pytest.raises(ValueError,match='execution source differs'):open_inference(artifact,options={'model_dir':config.options['model_dir']},operation='response')


def test_changed_helper_source_rejects_before_native_allocation(cpu_software_cuda_path,monkeypatch):
    _,_,config=cpu_software_cuda_path
    with open_response(config) as session:artifact=session.export_inference()
    monkeypatch.setattr(b,'helper_sources',lambda:{'backend.py':'different helper source'})
    monkeypatch.setattr(b,'_native',lambda *args,**kwargs:pytest.fail('allocated changed helper closure'))
    with pytest.raises(ValueError,match='execution source differs'):
        open_inference(artifact,options={'model_dir':config.options['model_dir']},operation='response')


def test_cpu_rows_are_not_rebound_to_cuda_protocol(cpu_software_cuda_path,tmp_path):
    _,rows,config=cpu_software_cuda_path
    path=tmp_path/'old-cpu-rows.pt';torch.save(dict(schema=cpu.PROTOCOL,model_identity=config.options['model_identity'],rows=rows),path)
    options=dict(config.options,rows=str(path),rows_sha256=b.file_sha256(path))
    with pytest.raises(ValueError,match='no CPU cohort rebinding'):b.read_rows(options)


def test_checkpoint_profile_preserves_scale_gradients_and_borrowed_cleanup(cpu_software_cuda_path):
    native,rows,config=cpu_software_cuda_path
    original=native._gradient_checkpointing_func
    outputs=[];grads=[]
    for enabled in (False,True):
        options=dict(config.options,gradient_checkpointing=enabled)
        options['model_identity']=b.bundle_identity(options['model_dir'],options=options)
        policy=b.execution_policy(options);initialization=b.initialization_identity(7,policy)
        adapter,close=b._native(options,options['model_identity'],native,initialization=initialization)
        try:
            with torch.no_grad():
                for p in adapter.network.parameters():p.add_(.01)
            values=[adapter.velocity(row['neutral'],row['latent'],row['timestep'],scale=s) for row,s in zip(rows,(.5,-1.))]
            leaves=list(adapter.network.parameters());loss=sum(v.square().mean() for v in values)
            first=torch.autograd.grad(loss,leaves,retain_graph=True)
            second=torch.autograd.grad(loss,leaves)
            assert all(torch.equal(a,b) for a,b in zip(first,second))
            outputs.append([v.detach() for v in values]);grads.append([g.detach() for g in second])
        finally:close()
        assert not native.is_gradient_checkpointing and native._gradient_checkpointing_func is original
        assert not any(m._forward_hooks for m in native.modules())
    assert all(torch.equal(a,b) for a,b in zip(outputs[0]+grads[0],outputs[1]+grads[1]))


def test_cuda_cli_keeps_raw_and_no_ema_defaults(monkeypatch,tmp_path):
    from embedded_zimage_shared import cuda_cli as cli
    captured={}
    class Session:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def train(self,**kwargs):return []
        def checkpoint(self,path):pass
        def export_inference(self,path,**kwargs):captured['export']=kwargs
    def opened(config,**kwargs):captured['config']=config;return Session()
    monkeypatch.setattr(cli,'configure_execution_profile',lambda:captured.update(configured=True))
    monkeypatch.setattr(cli,'bundle_identity',lambda *args,**kwargs:{'fixture':'no native allocation'})
    monkeypatch.setattr(cli,'file_sha256',lambda *args:'a'*64)
    monkeypatch.setattr(cli,'open_response',opened)
    cli.main(['train','--model-dir',str(tmp_path/'model'),'--rows',str(tmp_path/'rows'),
        '--shape','1','4','4','4','--checkpoint',str(tmp_path/'checkpoint'),'--export',str(tmp_path/'raw'),
        '--lr','0.0001','--d-lr-mult','2'])
    assert captured['config'].options['ema_decay'] is None
    assert captured['export']=={'selection':'raw'}
    assert captured['config'].options['runtime_profile']==e.CUDA_PROFILE
    assert captured['config'].options['lr']==1e-4 and captured['config'].options['d_lr_mult']==2.
    assert captured['config'].options['gradient_checkpointing'] is False


def test_cuda_cli_refuses_implicit_ema_before_profile_or_paths(monkeypatch):
    from embedded_zimage_shared import cuda_cli as cli
    monkeypatch.setattr(cli,'configure_execution_profile',lambda:pytest.fail('configured invalid EMA request'))
    with pytest.raises(SystemExit):
        cli.main(['train','--model-dir','missing','--rows','missing','--shape','1','4','4','4',
            '--checkpoint','missing','--export','missing','--selection','ema'])


def test_changed_profile_refuses_before_native_forward(cpu_software_cuda_path,monkeypatch):
    native,rows,config=cpu_software_cuda_path;options=config.options
    policy=b.execution_policy(options);initialization=b.initialization_identity(7,policy)
    adapter,close=b._native(options,options['model_identity'],native,initialization=initialization)
    try:
        monkeypatch.setattr(b,'execution_policy',lambda options:dict(policy,changed_numerical_flag=True))
        monkeypatch.setattr(native,'forward',lambda *args,**kwargs:pytest.fail('forward ran under changed numerical profile'))
        with pytest.raises(ValueError,match='execution policy changed'):
            adapter.velocity(rows[0]['neutral'],rows[0]['latent'],rows[0]['timestep'])
    finally:close()


def test_fresh_process_cpu_software_replay_under_different_hash_seeds(tmp_path):
    import os,subprocess,sys,json
    from pathlib import Path
    worker=Path(__file__).with_name('cuda_cpu_worker.py')
    for mode,hash_seed in [('full','123'),('replay','987')]:
        result=subprocess.run([sys.executable,str(worker),str(tmp_path/'cohort'),mode],cwd=tmp_path,
            env=dict(os.environ,PYTHONHASHSEED=hash_seed,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=120)
        assert result.returncode==0,result.stdout+result.stderr
    receipt=json.loads((tmp_path/'cohort/result.json').read_text())
    assert all(receipt[k] for k in ('initial_exact','resume_exact','metrics_exact','cuda_uninitialized'))

"""Four NEW authored/UNRUN native tiny software cases; ONE total update.

Original core/Git/native/HNDL guards and genuine owning distribution RECORD
stay active. Synthetic completed fields are data, never a teacher rerun.
"""
from copy import deepcopy
import math
import pytest,torch
from diffusers import ZImageTransformer2DModel
from particle_sliders.pg09_backends import PreparedResponse,ResponseRow,FrozenResponseAdapter
from particle_sliders.lifecycle import UnsupportedOperation
from embedded_zimage_shared import backend as original,cuda_backend as original_cuda
from embedded_zimage_shared.fresh95 import open_training,open_inference,save_prepared_cache,file_pin,Checkpoint
from embedded_zimage_shared.fresh95 import api,cache,policy,ownership
from embedded_zimage_shared.fresh95.util import equal

def tiny_model():
    # Original native tiny constructor dimensions; no old test is imported/run.
    return ZImageTransformer2DModel(all_patch_size=(2,),all_f_patch_size=(1,),in_channels=4,
        dim=32,n_layers=1,n_refiner_layers=1,n_heads=4,n_kv_heads=2,cap_feat_dim=16,
        axes_dims=[4,2,2],axes_lens=[128,128,128])

def component_factory(request):
    assert request['operation']in('fresh','resume','inference')and request['particle_bank']is None
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        return tiny_model()

@pytest.fixture
def staged(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('no old request preparation, native loader or teacher in new software cases')
    for module in(original,original_cuda):
        monkeypatch.setattr(module,'prepare',forbidden)
        monkeypatch.setattr(module,'read_rows',forbidden)
    monkeypatch.setattr(FrozenResponseAdapter,'prepare',forbidden)
    monkeypatch.setattr(ZImageTransformer2DModel,'from_pretrained',classmethod(forbidden))
    with torch.random.fork_rng(devices=[]):
        base=component_factory(dict(operation='fresh',particle_bank=None));state=deepcopy(base.state_dict())
        base.save_pretrained(tmp_path/'model');identity=original.bundle_identity(tmp_path/'model')
        contexts=tuple({'embeds':torch.linspace(.1,.6,n*16).reshape(n,16)}for n in(3,4,2))
        prepared=[]
        for i in range(3):
            latent=torch.linspace(-.3,.3,64).reshape(1,4,4,4)+i*.01
            row=ResponseRow(*deepcopy(contexts),latent,timestep=torch.tensor(300.+i*100.),guidance=1.)
            prepared.append(PreparedResponse(row,torch.full_like(latent,.1+.03*i),torch.zeros_like(latent)))
    config=dict(runtime='cpu-native-fp32',device='cpu',model_dir=str(tmp_path/'model'),model_identity=identity,
        response_shape=[1,4,4,4],steps=3,batch=2,seed=7,preset='bispf_optimized',ema_decay=None,
        base_precision='float32',pg_version='0.9.5',pg_source_profile='release-0.9.5',
        penalty_input_domain='continuous',lr=.001,d_lr_mult=18.,gradient_checkpointing=False)
    original_output=tmp_path/'synthetic-completed-native-fields.pt'
    torch.save([cache.dump(item)for item in prepared],original_output)
    whole=file_pin(original_output);source=file_pin(__file__)
    origins=[dict(scope='synthetic-software-only',complete_prepared_output=True,synthetic_row=i,
        whole_output_files=[dict(path=whole.path,bytes=whole.bytes,sha256=whole.sha256)],
        source_files=[dict(path=source.path,bytes=source.bytes,sha256=source.sha256)],real_teacher_calls=0)for i in range(3)]
    frozen=save_prepared_cache(tmp_path/'completed-cache.pt',config=config,prepared=prepared,
        teacher_origins=origins,cache_id='NEW-ZImage-three-completed-synthetic-fields',
        scope='synthetic-software-only',owner_admission=lambda request:True)
    return dict(config=config,cache=frozen,cohort_id='NEW-ZImage-cached-fresh95-software-v1',
        component_factory=component_factory,owner_admission=lambda request:True),tmp_path,state

def opened(staged,name='fresh'):
    options,root,_=staged
    return open_training(directory=root/name,mode='fresh',**options)

def test_new_before_optimizer_same_HNDL_Parameters_zeroUP_critic_KEEP_no_bank(staged,monkeypatch):
    initialize=policy.initialize;game=api.ConditionalPG095;seen=[]
    def initialize_spy(adapter,critic,*args):
        ids=[id(p)for p in adapter.parameters()]+[id(p)for p in critic.parameters()]
        before=deepcopy(critic.state_dict());receipt=initialize(adapter,critic,*args)
        assert ids==[id(p)for p in adapter.parameters()]+[id(p)for p in critic.parameters()]
        assert equal(before,critic.state_dict())
        assert all(torch.count_nonzero(p.graph['up'].weight)==0 for p in adapter.projections.values())
        seen.append(receipt);return receipt
    def construct(backend,*args,**kwargs):
        assert seen and backend.metadata()['fresh_initialization']['receipt']==seen[-1]
        assert kwargs['particles']is None and kwargs['response_scale']==1.0
        return game(backend,*args,**kwargs)
    monkeypatch.setattr(policy,'initialize',initialize_spy);monkeypatch.setattr(api,'ConditionalPG095',construct)
    with opened(staged)as owner:
        assert owner.game.completed_steps==0 and owner.game.prior is None
        assert owner.game.state_dict()['config']['backend']['fresh_initialization']['receipt']==seen[0]
        assert (owner.directory/'FULL-000000.pt').is_file()and(owner.directory/'RAW-000000.pt').is_file()
        assert {id(p)for p in owner.native.network.parameters()}=={id(p)for p in owner.network.parameters()}
        assert owner.constructor==original.initialization_identity(7)
        assert owner.cache_identity['game_response_scale']==1.0
        assert owner.cache_identity['target_delta_RMS_diagnostic']!=1.0

def test_new_a1_QR_Gram_deterministic_KEEP_caller_RNG_and_frozen_base(staged):
    before=torch.get_rng_state().clone();_,_,base=staged
    with opened(staged,'a')as owner:
        first=deepcopy(owner.network.state_dict());receipt=deepcopy(owner.initialization)
        model=owner.model;flags=owner.attachment.flags
        for projection in owner.network.projections.values():
            leaf=projection.graph['down'];down=leaf.weight.detach();rows,cols=down.shape;variance=1/leaf.in_features
            gram=down@down.T if rows<=cols else down.T@down
            assert torch.allclose(gram,torch.eye(min(rows,cols))*variance*max(rows,cols),rtol=1e-5,atol=1e-6)
            assert math.isclose(float(down.square().mean()),variance,rel_tol=1e-5,abs_tol=1e-7)
            assert torch.count_nonzero(projection.graph['up'].weight)==0
        assert equal(base,model.state_dict())
    assert all(p.requires_grad==flag for p,flag in flags)
    assert not hasattr(model,ownership.MARKER)and not any(m._forward_hooks for m in model.modules())
    assert torch.equal(before,torch.get_rng_state())
    with opened(staged,'b')as again:
        assert equal(first,again.network.state_dict())and receipt==again.initialization
        assert all(x['before_sha256']==x['after_sha256']for x in receipt['parameters']if x['role']!='host-a1-down-QR')
    assert torch.equal(before,torch.get_rng_state())

def test_new_ONE_update_strict_whole_FULL_opts_RNG_and_RAW_native_skip(staged,monkeypatch):
    options,root,base=staged
    with opened(staged)as trained:
        trained.update()  # The ONE total new tiny random native software update.
        full=trained.save(root/'saved-FULL1.pt');raw=trained.export(root/'saved-RAW1.pt')
        state=deepcopy(trained.game.state_dict());network=deepcopy(trained.network.state_dict());row=trained.game.rows[0].row
        model=trained.model
    assert equal(base,model.state_dict())
    def forbidden(*args,**kwargs):raise AssertionError('resume/RAW cannot initialize or recollect')
    monkeypatch.setattr(policy,'initialize',forbidden)
    with open_training(directory=root/'resume',mode='resume',resume=full,**options)as resumed:
        assert resumed.inherited_steps==1 and equal(state,resumed.game.state_dict())
    monkeypatch.setattr(api,'load_cache',forbidden);monkeypatch.setattr(api,'ConditionalPG095',forbidden)
    with open_inference(raw,component_factory=component_factory,owner_admission=lambda request:True)as inferred:
        assert not hasattr(inferred,'game')and not hasattr(inferred,'critic')
        assert equal(network,inferred.network.state_dict())
        before=torch.get_rng_state().clone()
        result=inferred.response(condition=row.neutral,latent=row.latent,timestep=row.timestep,strength=1.,seed=17)
        assert result.shape==(1,4,4,4)and torch.isfinite(result).all()and torch.equal(before,torch.get_rng_state())
        assert equal(base,inferred.model.state_dict())

def test_new_early_mode_FULL_EMA_refusal_and_retry_only_pending_cleanup(staged,monkeypatch):
    options,root,_=staged;calls=[]
    def forbidden(request):calls.append(request);raise AssertionError('bad intent reached native model factory')
    bad=dict(options,component_factory=forbidden)
    for mode,resume,ema in(('fresh',Checkpoint('unused',1,'0'*64),None),('resume',None,None),('fresh',None,.9)):
        with pytest.raises(ValueError):open_training(directory=root/'bad',mode=mode,resume=resume,ema_decay=ema,**bad)
    legacy=root/'old-FULL.pt';torch.save(dict(schema='old-ZImage-FULL'),legacy)
    with pytest.raises(ValueError):open_training(directory=root/'legacy',mode='resume',resume=file_pin(legacy),**bad)
    with pytest.raises(ValueError):
        open_training(directory=root/'denied',mode='fresh',**dict(options,component_factory=forbidden,owner_admission=lambda request:False))
    assert calls==[]
    bounded=opened(staged,'bounded');row=bounded.game.rows[0].row
    for strength in(-1.,2.,float('nan')):
        with pytest.raises(UnsupportedOperation):bounded.response(condition=row.neutral,latent=row.latent,timestep=row.timestep,strength=strength)
    bounded.close()
    retained=opened(staged,'marker-retained');attachment=retained.attachment
    parameter=attachment.pending_flags[0][0];requires_grad=parameter.requires_grad_;restores=[]
    def fail_restore_once(value):
        restores.append(value)
        if len(restores)==1:raise RuntimeError('deliberate fail-once flag restoration')
        return requires_grad(value)
    monkeypatch.setattr(parameter,'requires_grad_',fail_restore_once)
    with pytest.raises(BaseExceptionGroup,match='cleanup pending'):retained.close()
    assert getattr(retained.model,ownership.MARKER)is attachment and attachment.pending_flags==[(parameter,True)]
    with pytest.raises(ValueError,match='owner exists'):ownership.Attachment(retained.model)
    foreign_marker=object();setattr(retained.model,ownership.MARKER,foreign_marker)
    with pytest.raises(BaseExceptionGroup,match='cleanup pending'):retained.close()
    assert len(restores)==1 and getattr(retained.model,ownership.MARKER)is foreign_marker
    setattr(retained.model,ownership.MARKER,attachment);retained.close()
    assert retained.closed and len(restores)==2 and not hasattr(retained.model,ownership.MARKER)
    owner=opened(staged,'cleanup');handle=owner.attachment.pending_hooks[0];remove=handle.remove;attempts=[]
    def once():
        attempts.append(1)
        if len(attempts)==1:raise RuntimeError('deliberate fail-once owned removal')
        return remove()
    monkeypatch.setattr(handle,'remove',once)
    with pytest.raises(BaseExceptionGroup,match='cleanup pending'):owner.close()
    assert owner.closing and not owner.closed and owner.attachment.pending_hooks==[handle]
    with pytest.raises(RuntimeError,match='cleanup'):owner.update()
    with pytest.raises(RuntimeError,match='cleanup'):owner.require_open()
    owner.close();assert owner.closed and len(attempts)==2
    foreign_owner=opened(staged,'foreign');model=foreign_owner.model;attachment=foreign_owner.attachment
    # Synthetic owned checkpoint state exercises pending restoration on CPU;
    # no checkpoint forward, CUDA call, native enablement or update is added.
    key='_gradient_checkpointing_func';had=key in vars(model);original_checkpoint=vars(model).get(key)
    def owned_checkpoint(*args,**kwargs):raise AssertionError('synthetic cleanup attribute must never execute')
    setattr(model,key,owned_checkpoint)
    attachment.pending_checkpoint.append((model,key,had,original_checkpoint,True,owned_checkpoint))
    def borrowed_state():
        return dict(hooks=tuple((id(module),id(module._forward_hooks),tuple((k,id(v))for k,v in module._forward_hooks.items()),
                id(module._forward_pre_hooks),tuple((k,id(v))for k,v in module._forward_pre_hooks.items()))for module in model.modules()),
            handles=(id(foreign_owner.network._handles),tuple(id(handle)for handle in foreign_owner.network._handles)),
            checkpoints=tuple((id(module),name,name in vars(module),id(vars(module).get(name)))for module in model.modules()
                for name in('gradient_checkpointing','_gradient_checkpointing_func')),
            flags=tuple((id(parameter),parameter.requires_grad)for parameter in model.parameters()),
            modes=tuple((id(module),module.training)for module in model.modules()),
            pending_hooks=tuple(id(handle)for handle in attachment.pending_hooks),
            pending_checkpoint=tuple((id(module),name,existed,id(old),exists,id(owned))
                for module,name,existed,old,exists,owned in attachment.pending_checkpoint),
            pending_container=attachment.pending_container,pending_marker=attachment.pending_marker)
    before_foreign=borrowed_state();foreign=object();setattr(model,ownership.MARKER,foreign)
    with pytest.raises(BaseExceptionGroup,match='cleanup pending'):foreign_owner.close()
    assert borrowed_state()==before_foreign  # No borrowed mutation before exclusive-marker refusal.
    assert getattr(model,ownership.MARKER)is foreign and not foreign_owner.closed
    setattr(model,ownership.MARKER,attachment);foreign_owner.close();assert foreign_owner.closed
    assert (key in vars(model))is had and vars(model).get(key)is original_checkpoint
    flag_owner=opened(staged,'foreign-gradient-flag');attachment=flag_owner.attachment
    parameter,original_flag=attachment.flags[0];owned_flag=attachment.owned_flags[id(parameter)]
    assert owned_flag is False and original_flag is True
    parameter.requires_grad_(True)
    with pytest.raises(BaseExceptionGroup,match='cleanup pending'):flag_owner.close()
    assert parameter.requires_grad is True and not flag_owner.closed
    assert attachment.pending_flags==[(parameter,original_flag)]and getattr(flag_owner.model,ownership.MARKER)is attachment
    parameter.requires_grad_(owned_flag);flag_owner.close()
    assert flag_owner.closed and parameter.requires_grad is original_flag and not hasattr(flag_owner.model,ownership.MARKER)
    mode_owner=opened(staged,'foreign-training-mode');attachment=mode_owner.attachment
    module,original_mode=attachment.modes[0];owned_mode=attachment.owned_modes[id(module)]
    assert owned_mode is False and original_mode is True
    module.training=True
    with pytest.raises(BaseExceptionGroup,match='cleanup pending'):mode_owner.close()
    assert module.training is True and not mode_owner.closed
    assert attachment.pending_modes==[(module,original_mode)]and getattr(mode_owner.model,ownership.MARKER)is attachment
    module.training=owned_mode;mode_owner.close()
    assert mode_owner.closed and module.training is original_mode and not hasattr(mode_owner.model,ownership.MARKER)
    hook_owner=opened(staged,'foreign-hook');attachment=hook_owner.attachment
    handle=attachment.pending_hooks[0];hooks=handle.hooks_dict_ref();owned=hooks[handle.id]
    foreign=lambda *args,**kwargs:None;hooks[handle.id]=foreign
    with pytest.raises(BaseExceptionGroup,match='cleanup pending'):hook_owner.close()
    assert hooks[handle.id]is foreign and not hook_owner.closed
    hooks[handle.id]=owned;hook_owner.close();assert hook_owner.closed
    container_owner=opened(staged,'foreign-container');attachment=container_owner.attachment
    original_handles=attachment.handle_container
    replacement=list(original_handles);container_owner.network._handles=replacement
    with pytest.raises(BaseExceptionGroup,match='cleanup pending'):container_owner.close()
    assert container_owner.network._handles is replacement and replacement==original_handles
    container_owner.network._handles=original_handles
    appended=object();original_handles.append(appended)
    with pytest.raises(BaseExceptionGroup,match='cleanup pending'):container_owner.close()
    assert original_handles[-1]is appended and not container_owner.closed
    original_handles.pop();container_owner.close();assert container_owner.closed

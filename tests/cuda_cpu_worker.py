"""Fresh-process CPU software replay of CUDA factory. Not CUDA qualification."""
from contextlib import contextmanager
from pathlib import Path
import json,sys,hashlib
import torch
from diffusers import ZImageTransformer2DModel
from embedded_zimage_shared import cuda_backend as b
from embedded_zimage_shared import execution as e
import particle_sliders.lifecycle as lifecycle
import particle_sliders.inference as inference
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response
from particle_sliders.pg09 import _hash

torch.set_num_threads(1)
b.execution_policy=lambda options:dict(profile=e.CUDA_PROFILE,device='cpu',base_precision='float32',gradient_checkpointing=False,qualification='CPU software fixture only')
b.admit_owned_device=lambda policy:None
lifecycle._rng_devices=lambda *values:[]
inference._rng_devices=lambda *values:[]
@contextmanager
def scope(identity,component,policy):
    b.validate_initialization(identity,policy)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(identity['streams'][component]);yield
b.initialization_scope=scope
root=Path(sys.argv[1]);mode=sys.argv[2]
options=dict(runtime_profile=e.CUDA_PROFILE,device='cuda:0',base_precision='float32')
if mode=='full':
    root.mkdir()
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(23)
        native=ZImageTransformer2DModel(all_patch_size=(2,),all_f_patch_size=(1,),in_channels=4,
            dim=32,n_layers=1,n_refiner_layers=1,n_heads=4,n_kv_heads=2,norm_eps=1e-5,qk_norm=True,
            cap_feat_dim=16,siglip_feat_dim=None,rope_theta=256.,t_scale=1000.,axes_dims=[4,2,2],axes_lens=[128,128,128])
        assert not native.config.get('_use_default_values',[])
        native.save_pretrained(root/'model')
        identity=b.bundle_identity(root/'model',options=options)
        rows=[dict(neutral={'embeds':torch.randn(3,16)},positive={'embeds':torch.randn(4,16)},negative={'embeds':torch.randn(2,16)},latent=torch.randn(1,4,4,4),timestep=torch.tensor(300.+100*i)) for i in range(3)]
    torch.save(dict(schema=b.PROTOCOL,model_identity=identity,rows=rows),root/'rows.pt')
else:identity=b.bundle_identity(root/'model',options=options)
config=SessionConfig(b.NAME,b.COHORT,model_dir=str(root/'model'),model_identity=identity,
    rows=str(root/'rows.pt'),rows_sha256=b.file_sha256(root/'rows.pt'),response_shape=[1,4,4,4],
    steps=2,batch=2,seed=7,lr=1e-4,d_lr_mult=2.,**options)
if mode=='full':
    with open_response(config) as session:
        initial_hash=_hash(session.game.state_dict())
        session.step();session.checkpoint(root/'midpoint.pt');metrics=session.step();packet=session.checkpoint()
        torch.save(packet,root/'uninterrupted.pt')
        (root/'expected.json').write_text(json.dumps(dict(initial_hash=initial_hash,metrics=metrics,final_hash=_hash(packet))))
else:
    expected=json.loads((root/'expected.json').read_text())
    with open_response(config) as initial:initial_hash=_hash(initial.game.state_dict())
    with open_response(config,resume=root/'midpoint.pt') as replay:
        metrics=replay.step();packet=replay.checkpoint();actual_hash=_hash(packet)
        torch.save(packet,root/'replayed.pt')
    result=dict(initial_exact=initial_hash==expected['initial_hash'],resume_exact=actual_hash==expected['final_hash'],metrics_exact=metrics==expected['metrics'],cuda_uninitialized=not torch.cuda.is_initialized(),qualification='Actual tiny native blocks/HNDL on CPU with hardware scopes mocked; no CUDA claim.',worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),initial_hash=initial_hash,final_hash=actual_hash)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['initial_exact'] and result['resume_exact'] and result['metrics_exact'] and result['cuda_uninitialized'],result

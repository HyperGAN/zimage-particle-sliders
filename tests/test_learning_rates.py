"""API and both CLI paths must retain the same explicit optimizer law."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from particle_sliders.lifecycle import SessionConfig,CheckpointError
from particle_sliders.response import open_response
from embedded_zimage_shared import backend as product
from embedded_zimage_shared import cli
from .test_native_cohort import staged,equal


def configuration(staged,**rates):
    _,_,config=staged
    return SessionConfig(config.backend,config.cohort,**dict(config.options,**rates))


def test_rates_reach_real_optimizers_and_exact_resume(staged):
    config=SessionConfig.from_dict(json.loads(json.dumps(configuration(staged,lr=.001,d_lr_mult=2.).as_dict())))
    with open_response(config) as current:
        assert current.game.prior is None
        assert current.game.base_lrs[0] and set(current.game.base_lrs[0])=={.001}
        assert current.game.base_lrs[1] and set(current.game.base_lrs[1])=={.002}
        current.train(stop_after=2)
        packet=current.checkpoint()
        expected=current.train()
        final=current.checkpoint()
    with open_response(config) as restored:
        restored.restore(packet)
        equal(restored.train(),expected)
        equal(restored.checkpoint(),final)
    with open_response(configuration(staged,lr=.002,d_lr_mult=2.)) as changed:
        with pytest.raises(CheckpointError):
            changed.restore(packet)


def test_product_cli_reaches_identical_rates(staged,tmp_path,capsys):
    _,_,base=staged
    checkpoint=tmp_path/'training.pt'
    cli.main(['train','--model-dir',base.options['model_dir'],'--rows',base.options['rows'],
        '--shape','1','4','4','4','--steps','2','--lr','.001','--d-lr-mult','2',
        '--checkpoint',str(checkpoint),'--export',str(tmp_path/'adapter.pt')])
    import torch
    packet=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert packet['payload']['game']['config']['recipe']['lr']==.001
    assert packet['payload']['game']['config']['recipe']['d_lr_mult']==2.


@pytest.mark.parametrize('rates',[dict(lr=0),dict(lr=True),dict(lr=float('nan')),
    dict(d_lr_mult=-1),dict(d_lr_mult=float('inf')),dict(prior_lr_mult=1.)])
def test_invalid_rates_refuse_before_native_gate(rates,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('native preparation must not be reached')
    monkeypatch.setattr(product,'_core_gate',forbidden)
    with pytest.raises(ValueError,match='positive|particle bank'):
        product.validate(rates)


def test_product_cli_rejects_bank_before_native_paths(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('native paths must not be read')
    monkeypatch.setattr(cli,'bundle_identity',forbidden)
    with pytest.raises(SystemExit):
        cli.main(['train','--model-dir','missing','--rows','missing','--shape','1','4','4','4',
            '--checkpoint','missing','--export','missing','--prior-lr-mult','1'])

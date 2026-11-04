"""Bounded tiny random native API fixture dispatch; no pretrained admission."""


def fixture_profile():
    return dict(schema='particle-slider-profile-v1', name='zimage-fixture',
        product='zimage-particle-sliders', config_kind='session', status='fixture',
        backend='embedded-zimage-shared', interface='response', runtime={},
        capabilities=dict(resume=True, periodic_checkpoint=True, check=True),
        default_checkpoint_every=1, target_steps_path='options.steps',
        reason='Tiny random FP32 CPU native API fixtures only; pretrained/GPU quality unqualified.')

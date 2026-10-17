import pytest
import os
torch = pytest.importorskip("torch")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
pytest.importorskip("verl")
from swe_smith.probes.policy_objective_contract import direction_contract,native_objective
from pathlib import Path
import json
import math

def actor():
    return OmegaConf.create(dict(clip_ratio=.2,clip_ratio_low=.2,clip_ratio_high=.2,clip_ratio_c=3.,
        global_batch_info={},policy_loss={'loss_mode':'vanilla'},loss_agg_mode='token-mean',
        use_kl_loss=True,kl_loss_type='low_var_kl',kl_loss_coef=.001,grad_clip=1.,optim={'weight_decay':.01}))

def test_native_objective_direction_mask_and_optimizer_contract():
    result=direction_contract(actor(),steps=8)
    assert result['positive_adv_logprob_delta']>0
    assert result['negative_adv_logprob_delta']<0
    assert result['trace'][-1]['parameter_rms_delta']>0
    assert not result['real_model_loaded'] and not result['real_trajectories_fitted']

def test_unfrozen_or_bad_token_fields_rejected():
    old=torch.zeros((1,2),requires_grad=True)
    with pytest.raises(ValueError):native_objective(actor(),old,old,old,torch.ones_like(old),old)
    x=old.detach()
    with pytest.raises(ValueError):native_objective(actor(),x,x,x,torch.zeros_like(x),x)
    with pytest.raises(ValueError):native_objective(actor(),x,x,x,torch.ones((1,1)),x)

def test_resolved_formal_rollout_mean_and_ppo_clip_boundary():
    configured=os.environ.get('POLICY_CONTRACT_ACTOR_CONFIG')
    if not configured:pytest.skip('requires verified formal settings export')
    path=Path(configured)
    config=OmegaConf.create(json.loads(path.read_text()))
    result=direction_contract(config,steps=8)
    assert result['formal_per_rollout_normalization']
    assert result['policy_loss_mode']=='per_rollout_mean'
    old=torch.tensor([[-1.]],dtype=torch.float64)
    logprob=(old+math.log(1+float(config.clip_ratio_high))+.1).requires_grad_()
    loss,*_=native_objective(config,old,logprob,torch.ones_like(old)/4,
        torch.ones_like(old),logprob.detach(),rollout_normalized=True)
    loss.backward()
    assert logprob.grad.abs().max().item()==0

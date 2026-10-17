"""CPU contract preparation, NOT a real-model frozen-trajectory experiment.

Calls installed verl's selected policy loss, KL and aggregation functions. The
small normalized-logits fixture checks sign/mask/optimizer instrumentation only.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import torch
from omegaconf import OmegaConf
from verl.trainer.ppo.core_algos import get_policy_loss_fn,kl_penalty,agg_loss

def load_formal_loss(actor):
    if str(actor.policy_loss.loss_mode)!='per_rollout_mean':return None
    source=Path(str(actor.verified_custom_loss_source))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==actor.verified_custom_loss_sha256
    spec=importlib.util.spec_from_file_location('frozen_contract_formal_loss',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Register loss only; never call worker hook.
    return module

def native_objective(actor,old,logprob,advantages,mask,reference,*,rollout_normalized=False):
    if old.requires_grad or advantages.requires_grad or reference.requires_grad:
        raise ValueError('old policy, advantages and reference must be frozen')
    if not (old.shape==logprob.shape==advantages.shape==mask.shape==reference.shape):
        raise ValueError('objective fields have inconsistent token shapes')
    if not mask.bool().any():
        raise ValueError('diagnostic requires real action tokens')
    cfg=OmegaConf.create(dict(clip_ratio=float(actor.clip_ratio),clip_ratio_low=actor.clip_ratio_low,
         clip_ratio_high=actor.clip_ratio_high,clip_ratio_c=float(actor.get('clip_ratio_c',3.0)),global_batch_info={}))
    mode=str(actor.policy_loss.loss_mode)
    if mode=='per_rollout_mean' and not rollout_normalized:
        raise ValueError('per-rollout objective requires fixed normalized advantages')
    pg,metrics=get_policy_loss_fn(mode)(old_log_prob=old,log_prob=logprob,advantages=advantages,
        response_mask=mask,loss_agg_mode=str(actor.loss_agg_mode),config=cfg)
    kl=agg_loss(kl_penalty(logprob,reference,str(actor.kl_loss_type)),mask,str(actor.loss_agg_mode))
    total=pg+float(actor.kl_loss_coef)*kl if actor.use_kl_loss else pg
    return total,pg,kl,metrics

def direction_contract(actor,steps=32,lr=1e-7):
    assert lr>0 and steps>0
    logits=torch.nn.Parameter(torch.zeros((2,4,3),dtype=torch.float64,device='cpu'))
    actions=torch.tensor([[0]*4,[1]*4],dtype=torch.long)
    mask=torch.tensor([[1,1,0,0],[1,1,0,0]],dtype=torch.float64)
    advantages=torch.tensor([[1.]*4,[-1.]*4],dtype=torch.float64)
    custom=load_formal_loss(actor)
    if custom:
        advantages=custom.normalize_advantages_by_rollout(advantages,mask,['positive','negative'],num_trained_rows=2)
    frozen_adv=advantages.clone()
    def logprob():
        return logits.log_softmax(-1).gather(-1,actions.unsqueeze(-1)).squeeze(-1)
    old=logprob().detach().clone()
    reference=old.clone()
    initial=logits.detach().clone()
    before=old.clone()
    optimizer=torch.optim.AdamW([logits],lr=lr,weight_decay=float(actor.optim.weight_decay))
    trace=[]
    for index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        lp=logprob()
        lp.retain_grad()
        loss,pg,kl,_=native_objective(actor,old,lp,advantages,mask,reference,rollout_normalized=custom is not None)
        loss.backward()
        assert lp.grad[mask==0].abs().max().item()==0
        assert torch.isfinite(logits.grad).all()
        before_clip=torch.linalg.vector_norm(logits.grad).item()
        torch.nn.utils.clip_grad_norm_([logits],float(actor.grad_clip))
        after_clip=torch.linalg.vector_norm(logits.grad).item()
        optimizer.step()
        with torch.no_grad():
            after=logprob()
            trace.append(dict(iteration=index+1,loss=loss.item(),policy_loss=pg.item(),kl_loss=kl.item(),
                 gradient_norm_before_clip=before_clip,gradient_norm_after_clip=after_clip,
                 positive_adv_action_logprob=after[(advantages>0)&mask.bool()].mean().item(),
                 negative_adv_action_logprob=after[(advantages<0)&mask.bool()].mean().item(),
                 parameter_rms_delta=(logits-initial).square().mean().sqrt().item(),masked_logprob_gradient_max=0.0))
    assert torch.equal(old,before)
    assert torch.equal(advantages,frozen_adv) and torch.equal(reference,before)
    pos=(after-before)[(advantages>0)&mask.bool()].mean().item()
    neg=(after-before)[(advantages<0)&mask.bool()].mean().item()
    assert pos>0 and neg<0 and trace[-1]['loss']<trace[0]['loss']
    assert all(math.isfinite(row[k]) for row in trace for k in row)
    return dict(mode='synthetic_CPU_objective_contract_only',real_model_loaded=False,
        real_trajectories_fitted=False,quality_conclusion=False,policy_loss_mode=str(actor.policy_loss.loss_mode),
        loss_aggregation=str(actor.loss_agg_mode),kl_coefficient=float(actor.kl_loss_coef),
        diagnostic_lr=lr,steps=steps,old_logprob_and_advantages_frozen=True,
        formal_per_rollout_normalization=custom is not None,
        positive_adv_logprob_delta=pos,negative_adv_logprob_delta=neg,trace=trace)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--actor-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--synthetic-contract-only',action='store_true',required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    result=direction_contract(OmegaConf.create(json.loads(args.actor_config.read_text())))
    with args.output.open("x",encoding="utf-8") as out:
        json.dump(result,out,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='trace'}))

if __name__=='__main__':main()

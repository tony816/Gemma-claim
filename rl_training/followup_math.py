"""Online rewards and a frozen-v2 reference; no reference answers or labels."""
import torch
import torch.nn.functional as F
from .core import parse_json, advantages
from tools.rl_judge import score
from rl_materials.contracts import validate_output
from serving.rl_client import evidence_in_policy


def rewards_for(policy, texts, judgment):
    cards, _ = evidence_in_policy(policy)
    supplied = {c['card_id']: {**c, 'review_status': 'approved'} for c in cards}
    rewards = []; errors = []
    for raw, assessed in zip(texts, judgment['candidates']):
        reward = score(assessed, policy['task_type'])
        problems = []
        if policy['task_type'] == 'claim_set':
            try:
                parsed = raw if isinstance(raw, dict) else parse_json(raw)
                problems = validate_output(parsed, supplied)
                if problems: reward = min(reward, .25)
            except (ValueError, TypeError, AttributeError):
                problems = ['invalid_json']; reward = 0.
        rewards.append(reward); errors.append(problems)
    pref = judgment['preference']
    if len(rewards) == 2 and rewards[0] == rewards[1] and rewards[0] > 0 and pref in {'A', 'B'}:
        loser = 1 if pref == 'A' else 0
        rewards[loser] = max(0., rewards[loser] - .01)
    return rewards, errors


def weighted_advantages(rewards):
    return advantages(rewards) * min(1., (max(rewards) - min(rewards)) / .2)


def token_logps(logits, tokens):
    assert logits.shape[1] == tokens.shape[1] + 1
    return -F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                            tokens.reshape(-1), reduction='none')


def objective(logits, tokens, advantage, reference_logps, beta):
    policy_logps = token_logps(logits, tokens)
    delta = (reference_logps.detach() - policy_logps).clamp(-10, 10)
    kl = (delta.exp() - delta - 1).mean()
    return -float(advantage) * policy_logps.mean() + beta * kl, kl


def activate(model, name):
    model.set_adapter(name)
    # PEFT set_adapter can toggle requires_grad. Never train the reference.
    for key, parameter in model.named_parameters():
        parameter.requires_grad_('.default.' in key and name == 'default')

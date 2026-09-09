"""Frozen RL examples and verified source resolution for the local tester.

Only TRAIN policy inputs and source cards are exposed. No reward references or
held-out evaluation answers are loaded by the UI.
"""
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_materials.contracts import resolve_annotations

RELEASE = ROOT/'data/case_rl/releases/rl-pilot-27fc35d39288134b'


def read_verified(relative):
    receipt = json.loads((ROOT/'.superloopy/evidence/frozen-release.json').read_text())
    manifest_bytes = (RELEASE/'manifest.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != receipt['manifest_sha256']:
        raise ValueError('release_manifest_mismatch')
    manifest = json.loads(manifest_bytes)
    path = (RELEASE/relative).resolve()
    if relative not in manifest['files'] or not path.is_relative_to(RELEASE.resolve()):
        raise ValueError('unmanifested_release_path')
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest['files'][relative]:
        raise ValueError('release_file_mismatch')
    return payload


def policies():
    return {p['episode_id']: p for p in map(json.loads,
        read_verified('policy/train.jsonl').decode('utf-8').splitlines())}


def evidence_in_policy(policy):
    cards, excerpts = [], []
    for message in policy['messages']:
        text = message['content']
        start = text.find('{')
        if start < 0:
            continue
        try:
            payload = json.loads(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            cards.extend(payload.get('case_cards', []))
            excerpts.extend(payload.get('evidence', []))
    return cards, excerpts


def build_messages(policy):
    from claim_client import encode_image
    messages = [{'role': m['role'], 'content': [{'type': 'text', 'text': m['content']}]}
                for m in policy['messages']]
    if policy['images']:
        images = []
        for relative in policy['images']:
            read_verified(relative)
            images.append({'type': 'image_url', 'image_url': {'url': encode_image(str(RELEASE/relative))}})
        user = next(m for m in messages if m['role'] == 'user')
        user['content'] = images + user['content']
    return messages


def resolve_response(raw, policy, allow_code_fence=False, source_split='train'):
    # UI may unwrap exactly one Markdown JSON fence, with an explicit notice.
    # Strict evaluation remains the default; never repair the JSON or its content.
    if allow_code_fence:
        fenced = re.fullmatch(r'\s*```(?:json)?\s*\n([\s\S]*?)\n```\s*', raw)
        if fenced:
            raw = fenced.group(1)
    if source_split not in {'train', 'calibration', 'test'}:
        raise ValueError('invalid_source_split')
    output = json.loads(raw)
    supplied, excerpts = evidence_in_policy(policy)
    if policy['task_type'] == 'claim_set':
        ids = {x['card_id'] for x in supplied}
        # Drawing episodes in held-out splits explicitly reuse approved train
        # cards. Resolve only supplied IDs, while the UI still reads TRAIN only.
        source_splits = ['train'] if source_split == 'train' else ['train', source_split]
        store = {}
        for split in source_splits:
            for card in map(json.loads, read_verified(f'evidence/cards.{split}.jsonl').decode('utf-8').splitlines()):
                previous = store.get(card['card_id'])
                if previous is not None and previous != card:
                    raise ValueError('conflicting_verified_card')
                store[card['card_id']] = card
        if not ids.issubset(store):
            raise ValueError('supplied_card_missing_from_verified_store')
        return output, resolve_annotations(output, {cid: store[cid] for cid in ids})
    if not isinstance(output, dict) or set(output) != {'answer', 'citations', 'limitations'}:
        raise ValueError('advisory_output_keys')
    if not isinstance(output['answer'], str) or not output['answer'].strip():
        raise ValueError('advisory_answer_missing')
    if not isinstance(output['limitations'], list) or any(not isinstance(x, str) for x in output['limitations']):
        raise ValueError('advisory_limitations_invalid')
    citations = output['citations']
    if not isinstance(citations, list) or any(not isinstance(x, str) for x in citations):
        raise ValueError('advisory_citation_list_invalid')
    store = {x['evidence_id']: x for x in excerpts}
    if any(cid not in store for cid in citations):
        raise ValueError('unsupplied_advisory_citation')
    keys = ['evidence_id', 'jurisdiction', 'docket', 'court', 'decision_date',
            'pdf_page', 'speaker', 'quote', 'source_sha256']
    return output, [{k: store[cid].get(k) for k in keys} for cid in citations]

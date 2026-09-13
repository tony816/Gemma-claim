"""Describe held-out outputs without modifying model selection or judge scores."""
from collections import Counter
import json
import re

from .core import OUT, rows, parse_json, write
from serving.rl_client import evidence_in_policy, resolve_response


def summarize(label, split):
    policies = {p['episode_id']:p for p in rows(f'policy/{split}.jsonl')}
    path = OUT/f'pod/evaluation/{label}-{split}.jsonl'
    outputs = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert len(outputs) == len(policies) == 20
    result = {}
    for task in ('all','claim_set','patent_advisory','claims_with_cards','KR_advisory','US_advisory'):
        selected = [r for r in outputs if task == 'all' or r['task_type'] == task or
            (task == 'claims_with_cards' and r['task_type']=='claim_set' and
             evidence_in_policy(policies[r['episode_id']])[0]) or
            (task in {'KR_advisory','US_advisory'} and r['task_type']=='patent_advisory'
             and r['episode_id'].startswith(task[:2]+'-'))]
        counts = Counter(); dimensions = Counter(); cases = []
        for row in selected:
            contract_error = None
            try:
                resolve_response(row['raw'], policies[row['episode_id']], allow_code_fence=True, source_split=split)
                resolved_valid = True
            except (ValueError, TypeError) as exc:
                contract_error = str(exc)
                if contract_error in {'supplied_card_missing_from_verified_store', 'conflicting_verified_card',
                        'release_manifest_mismatch', 'release_file_mismatch', 'unmanifested_release_path'}:
                    raise RuntimeError('evaluation_source_lookup_failure') from exc
                resolved_valid = False
            counts['n'] += 1
            counts['severe_errors'] += row['judgment']['severe_error']
            counts['original_pipeline_resolved_contract'] += row['structure']['valid']
            counts['resolved_contract'] += resolved_valid
            counts['reward_sum'] += row['reward']
            dimensions.update(row['judgment']['dimensions'])
            try: json.loads(row['raw']); counts['strict_json'] += 1
            except ValueError: pass
            try: parsed = parse_json(row['raw'])
            except ValueError:
                cases.append({'episode_id':row['episode_id'],'parse_failure':True})
                continue
            if not isinstance(parsed,dict):
                cases.append({'episode_id':row['episode_id'],'object_failure':True})
                continue
            if row['task_type']=='claim_set':
                claims = parsed.get('claims',[])
                if not isinstance(claims,list):claims=[]
                claims=[c for c in claims if isinstance(c,dict)]
                independent = sum(c.get('kind')=='independent' for c in claims)
                dependent = sum(c.get('kind')=='dependent' for c in claims)
                annotations = parsed.get('annotations',[])
                if not isinstance(annotations,list):annotations=[]
                counts['one_independent'] += independent == 1
                counts['with_dependent_claims'] += dependent > 0
                counts['dependent_claims_total'] += dependent
                counts['nonempty_annotations'] += bool(annotations)
                counts['annotations_total'] += len(annotations)
                texts=[' '.join(c.get('text','').split()).casefold() for c in claims if isinstance(c.get('text'),str)]
                counts['duplicate_claim_text_cases'] += len(texts)!=len(set(texts))
                words=re.findall(r'\w+',row['raw'].casefold())
                grams=Counter(tuple(words[i:i+5]) for i in range(max(0,len(words)-4)))
                counts['repeated_five_word_sequence_8_times'] += max(grams.values(),default=0)>=8
                cases.append({'episode_id':row['episode_id'],'independent':independent,
                    'dependent':dependent,'annotations':len(annotations),
                    'resolved_contract':resolved_valid,'reward':row['reward'],
                    'severe_error':row['judgment']['severe_error'],'contract_error':contract_error})
            else:
                counts['citations_total'] += len(parsed.get('citations',[]))
                counts['nonempty_citations'] += bool(parsed.get('citations'))
        count = counts.pop('n',0)
        reward_sum = counts.pop('reward_sum',0)
        result[task] = {'n':count,'mean_semantic_reward':reward_sum/count if count else None,
            'counts':dict(counts),'mean_dimensions':{k:v/count for k,v in dimensions.items()},
            'claim_details':cases}
    return result


def run():
    selection = json.loads((OUT/'pod/selection.json').read_text())
    values = {f'{label}-{split}':summarize(label,split)
        for split in ('calibration','test') for label in ('v2','rl')}
    before = values['v2-calibration']['all']
    after = values['rl-calibration']['all']
    corrected_gate = {
        'passed': after['mean_semantic_reward'] >= before['mean_semantic_reward'] and
            after['counts']['severe_errors'] <= before['counts']['severe_errors'] and
            after['counts']['resolved_contract'] >= before['counts']['resolved_contract'],
        'baseline_contract': before['counts']['resolved_contract'],
        'rl_contract': after['counts']['resolved_contract'],
        'selected_step_unchanged': selection['step'],
        'method': 'Re-resolve unchanged generated outputs against explicitly supplied, hash-verified train plus current-split cards. No regeneration or reward change.'}
    write('comparison.json', {'selection':selection,'results':values,
        'corrected_calibration_gate':corrected_gate,
        'notes':['Automatic semantic rubric scores are not legal accuracy percentages.',
                 'Structural counts are corrected CPU re-resolution; original pipeline counts are preserved because its split-only card lookup was defective.',
                 'Resolved contract includes only IDs from the supplied cards/excerpts.',
                 'Nonempty annotation counts do not alone prove substantive application quality.',
                 'Repeated five-word sequences are a diagnostic heuristic, not a semantic error rate.',
                 'Strict JSON and explicit single-fence display parsing are reported separately.',
                 'Both policies use greedy generation with max_new_tokens 768.']})
    print(json.dumps(values,ensure_ascii=False,indent=2))


if __name__ == '__main__':run()

import json
import torch
from transformers import AutoProcessor
from .core import CONFIG,write
from .judge import response_schema, prepare_calibration, messages_for
from .formatting import tokenizer_data,prefix_function


def run():
    p=AutoProcessor.from_pretrained(CONFIG['judge_model'],revision=CONFIG['judge_revision'],
        min_pixels=CONFIG['judge_image_min_pixels'],max_pixels=CONFIG['judge_image_max_pixels'])
    data=tokenizer_data(p.tokenizer)
    sample={'dimension_reasons':{k:'The supplied evidence supports this limitation.' for k in ['source_annotation','image_support','dependency_scope','uncertainty']},
            'dimensions':{'image_support':4,'dependency_scope':2,'source_annotation':1,'uncertainty':1},
            'format_valid':True,'severe_error':False,'severe_reasons':[],'reasons':['Image 1 has a visible slot.']}
    outputs=[{'candidates':[sample,sample],'preference':'tie'},
             {'candidates':[sample],'preference':None}]
    checked=0
    for output in outputs:
        fn=prefix_function(data,response_schema('claim_set',len(output['candidates'])))
        prefix=p.tokenizer.encode('arbitrary prompt',add_special_tokens=False)
        text=json.dumps(output)
        # Conservative grammar filters can reject a merged punctuation token
        # while allowing its equivalent individual-character representation.
        ids=[token for char in text for token in p.tokenizer.encode(char,add_special_tokens=False)]
        assert p.tokenizer.decode(ids)==text
        assert p.tokenizer.eos_token_id not in fn(0,torch.tensor(prefix))
        for token in ids:
            assert token in fn(0,torch.tensor(prefix)),('valid_token_rejected',token,p.tokenizer.decode([token]))
            prefix.append(token);checked+=1
        assert p.tokenizer.eos_token_id in fn(0,torch.tensor(prefix))
    # An incomplete candidate array cannot terminate, and input-count is enforced.
    fn=prefix_function(data,response_schema('claim_set',2))
    prefix=p.tokenizer.encode('prompt',add_special_tokens=False)
    invalid=json.dumps({'candidates':[sample]})
    rejected=False
    for token in p.tokenizer.encode(invalid,add_special_tokens=False):
        allowed=fn(0,torch.tensor(prefix))
        if token not in allowed:rejected=True;break
        prefix.append(token)
    assert rejected
    packets,_=prepare_calibration();lengths=[]
    for item in packets:
        packet=dict(item['policy'],candidates=item['candidates'])
        enc=p.apply_chat_template(messages_for(packet),tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True)
        lengths.append(enc['input_ids'].shape[-1])
    assert max(lengths)<16000
    write('cpu_judge_encoding.json',{'passed':24,'token_lengths':lengths,'gpu_inference':'not_run'})
    write('cpu_schema.json',{'passed':True,'valid_tokens_checked':checked,
        'rejects_early_eos':True,'rejects_missing_second_candidate':True,
        'scope':'actual Qwen tokenizer and LMFE public API; no model quality measurement'})
    print('Schema CPU gate passed',checked,flush=True)


if __name__=='__main__':run()

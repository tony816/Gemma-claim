"""LM Format Enforcer's public API bridge for Transformers 5 tokenizers.

Algorithm follows the library's MIT-licensed Transformers integration, whose
legacy tokenization_utils import does not support this Transformers release.
https://github.com/noamgat/lm-format-enforcer/blob/main/lmformatenforcer/integrations/transformers.py
"""
from functools import partial
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData


def decode(tokenizer,tokens):
    return tokenizer.decode(tokens).rstrip('\ufffd')


def tokenizer_data(tokenizer):
    zero=tokenizer.encode('0',add_special_tokens=False)[-1]
    special=set(tokenizer.all_special_ids);regular=[]
    for index in range(len(tokenizer)):
        if index in special:continue
        after=tokenizer.decode([zero,index])[1:]
        alone=tokenizer.decode([index])
        # Judge output is English JSON. Restrict only GENERATED vocabulary;
        # multilingual prompt encoding remains the full, unchanged tokenizer.
        # This keeps LMFE's fast free-text path without a costly regex per token.
        if not all(32<=ord(char)<127 or char in '\n\r\t' for char in after):continue
        regular.append((index,after,len(after)>len(alone)))
    return TokenEnforcerTokenizerData(regular,partial(decode,tokenizer),tokenizer.eos_token_id,
                                     False,len(tokenizer))


def prefix_function(data,schema):
    enforcer=TokenEnforcer(data,JsonSchemaParser(schema,
        config=CharacterLevelParserConfig(force_json_field_order=True)))
    def allowed(batch_id,sequence):
        return enforcer.get_allowed_tokens(sequence.tolist()).allowed_tokens
    return allowed

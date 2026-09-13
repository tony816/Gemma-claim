"""Decode judge JSON, preserving literal string controls from tokenizer output."""
import json
import re


def parse_judge_json(text):
    text = text.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```', text)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        # strict=False changes ONLY acceptance of U+0000..U+001F inside
        # JSON strings. It neither fills fields nor repairs JSON structure.
        value = json.loads(text, strict=False)
        assert json.loads(json.dumps(value)) == value
        return value, True

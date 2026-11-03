"""Validated gateway-owned JSON for native engines without grammar support.

This is never applied to caller-supplied response_format. Only the gateway's
checkpoint and visual-note producers use it, followed by strict validation and
their existing bounded retry. Unsupported schema keywords fail closed.
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


def prepare_internal_json(body: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return body
    result=deepcopy(body)
    schema=result.pop('response_format')['json_schema']['schema']
    instruction=(
        '\n\n[INTERNAL_JSON_CONTRACT]\nReturn exactly one JSON object matching the '
        'following schema. No Markdown fences, explanations or trailing text. '
        'This contract is checked before the result can be used.\n'
        +json.dumps(schema,ensure_ascii=False,separators=(',',':'))
    )
    message=result['messages'][0]
    if message.get('role')!='system' or not isinstance(message.get('content'),str):
        raise ValueError('Internal JSON requires an owned text system message')
    message['content']+=instruction
    return result


def validate_internal_json(raw: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete owned schema subset; never accepts partial JSON."""
    def invalid_constant(value):
        raise ValueError('Non-finite JSON number: '+value)
    value=json.loads(raw,parse_constant=invalid_constant)
    def check(item,rule,path):
        allowed={'type','properties','required','additionalProperties','items','maxItems','maxLength','enum','const'}
        if set(rule)-allowed:
            raise ValueError('Unsupported internal schema keyword at '+path)
        kind=rule.get('type')
        types={'object':dict,'array':list,'string':str,'integer':int,'number':(int,float),'boolean':bool,'null':type(None)}
        if kind not in types or not isinstance(item,types[kind]) or kind in ('integer','number') and isinstance(item,bool):
            raise ValueError('Internal JSON type mismatch at '+path)
        if 'const' in rule and (type(item) is not type(rule['const']) or item!=rule['const']):
            raise ValueError('Internal JSON constant mismatch at '+path)
        if 'enum' in rule and not any(type(item) is type(option) and item==option for option in rule['enum']):
            raise ValueError('Internal JSON enum mismatch at '+path)
        if kind=='object':
            properties=rule.get('properties',{})
            if not set(rule.get('required',[]))<=set(item):
                raise ValueError('Internal JSON missing field at '+path)
            if rule.get('additionalProperties') is False and set(item)-set(properties):
                raise ValueError('Internal JSON unknown field at '+path)
            for name,child in item.items():
                if name in properties:check(child,properties[name],path+'.'+name)
        elif kind=='array':
            if len(item)>rule.get('maxItems',len(item)):
                raise ValueError('Internal JSON array too long at '+path)
            for index,child in enumerate(item):check(child,rule['items'],path+'['+str(index)+']')
        elif kind=='string' and len(item)>rule.get('maxLength',len(item)):
            raise ValueError('Internal JSON string too long at '+path)
    check(value,schema,'$')
    if not isinstance(value,dict):raise ValueError('Internal JSON must be an object')
    return value

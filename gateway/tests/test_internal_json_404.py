"""Exercise real internal producers against a backend rejecting grammar fields."""
import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from laguna_gateway.chat_images import NOTE_SCHEMA
from laguna_gateway.config import Settings
from laguna_gateway.engine import GatewayEngine
from laguna_gateway.internal_json import prepare_internal_json, validate_internal_json
from laguna_gateway.metrics import GatewayMetrics
from laguna_gateway.responses_checkpoint import CheckpointMaterial, checkpoint_response_format


def material():
    return CheckpointMaterial('task','source',1,'{}',('evidence',),('e1',),(),(),(),'',True,0,'key')


def valid_value(schema):
    if 'const' in schema:return schema['const']
    if 'enum' in schema:return schema['enum'][0]
    kind=schema['type']
    if kind=='object':return {k:valid_value(v) for k,v in schema['properties'].items()}
    if kind=='array':return []
    if kind=='string':return 'Observed evidence'
    if kind=='integer':return 1
    raise AssertionError(kind)


def test_owned_schema_preserved_and_original_request_immutable():
    schema=checkpoint_response_format(material())
    body={'messages':[{'role':'system','content':'Owned policy'},{'role':'user','content':'Evidence'}],
          'response_format':schema,'temperature':0,'max_tokens':100}
    original=deepcopy(body)
    adapted=prepare_internal_json(body,enabled=True)
    assert body==original and 'response_format' not in adapted
    assert json.loads(adapted['messages'][0]['content'].split('\n')[-1])==schema['json_schema']['schema']
    assert adapted['max_tokens']==100 and adapted['temperature']==0
    assert prepare_internal_json(body,enabled=False) is body


@pytest.mark.parametrize('mutate',[
    lambda x:x.pop('source_digest'),lambda x:x.update(extra='not allowed'),
    lambda x:x.update(source_item_count=True),lambda x:x.update(source_digest='wrong'),
    lambda x:x['goal'].update(evidence_ids=['invented']),
    lambda x:x['goal'].update(text='x'*385),lambda x:x.update(exact_literals=['invented']),
])
def test_invalid_evidence_never_passes(mutate):
    schema=checkpoint_response_format(material())['json_schema']['schema'];value=valid_value(schema)
    validate_internal_json(json.dumps(value),schema);mutate(value)
    with pytest.raises(ValueError):validate_internal_json(json.dumps(value),schema)


@pytest.mark.parametrize('raw',['{} trailing','```json\n{}\n```','{"x":NaN}','[]'])
def test_partial_or_non_json_output_rejected(raw):
    with pytest.raises(ValueError):validate_internal_json(raw,NOTE_SCHEMA)


def test_real_checkpoint_and_image_producers_no_longer_send_unsupported_format():
    async def run():
        bodies=[];schema=checkpoint_response_format(material())['json_schema']['schema']
        async def backend(request):
            body=json.loads(request.content) if request.content else {}
            if request.url.path=='/tokenize':return httpx.Response(200,json={'count':100})
            assert request.url.path=='/v1/chat/completions'
            bodies.append(body)
            if 'response_format' in body:return httpx.Response(400,json={'error':{'message':'structured output (response_format and the guided_* fields) is not supported by this engine yet'}})
            output=valid_value(NOTE_SCHEMA if len(bodies)==2 else schema)
            return httpx.Response(200,json={'choices':[{'message':{'role':'assistant','content':json.dumps(output)},'finish_reason':'stop'}],'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}})
        settings=Settings(tensorfold_internal_json=True,enforce_czech=False,vision_upstream_base_url='http://native.invalid',max_admitted_tokens=1000000)
        async with httpx.AsyncClient(base_url='http://native.invalid',transport=httpx.MockTransport(backend)) as client:
            engine=GatewayEngine(settings=settings,client=client,vision_client=client,metrics=GatewayMetrics())
            try:
                output=await engine._responses_checkpoint_completion(material=material(),system_prompt='Extract evidence.',user_prompt='source',purpose='responses_checkpoint_extract')
                validate_internal_json(output,schema)
                note=await engine._describe_chat_image({'type':'image_url','image_url':{'url':'data:image/png;base64,owned-test'}})
                validate_internal_json(note,NOTE_SCHEMA)
                assert len(bodies)==2 and all('response_format' not in b for b in bodies)
                # Direct/client structured requests retain their contract and error.
                response=await engine._post_completion({'messages':[{'role':'user','content':'X'}],'response_format':{'type':'json_object'}},purpose='client_passthrough')
                assert response.status_code==400 and bodies[-1]['response_format']=={'type':'json_object'}
            finally:await engine.close_background_tasks()
    asyncio.run(run())

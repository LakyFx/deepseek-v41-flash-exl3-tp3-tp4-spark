"""Fast synthetic reproductions of the 1.2.9 TensorFold format rejection.

Original failed prompts were not logged; none of these fixtures contain them.
"""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from laguna_gateway.app import create_app
from laguna_gateway.config import Settings
from laguna_gateway.client_json import (
    ACTIVE, META, Contract, JsonContractError, Validator, adapt, contract_from,
    documents, frames, run,
)


SCHEMA = {
    'type': 'object', 'required': ['status', 'count'], 'additionalProperties': False,
    'properties': {'status': {'enum': ['ok']}, 'count': {'type': 'integer', 'minimum': 1}},
}
FORMATS = [
    {'response_format': {'type': 'json_object'}},
    {'response_format': {'type': 'json_schema', 'json_schema': {'name': 'result', 'strict': True, 'schema': SCHEMA}}},
    {'guided_json': SCHEMA},
]


def completion(text='{"status":"ok","count":2}', *, finish='stop', calls=None, used=10):
    message = {'role': 'assistant', 'content': text}
    if calls:
        message['tool_calls'] = calls
    return {'id': 'chatcmpl-json', 'object': 'chat.completion', 'model': 'deepseek', 'created': 1,
        'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': used, 'total_tokens': 100 + used}}


async def exchange(body, *, api='chat_completions', outputs=None, enabled=True, check=None, jobs=None, enforce_czech=False):
    sent = []
    outputs = list(outputs or [completion()])
    async def backend(request):
        data = json.loads(request.content) if request.content else {}
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'count': 100})
        if request.url.path == '/v1/chat/completions':
            sent.append(data)
            if 'response_format' in data or any(k.startswith('guided_') for k in data):
                return httpx.Response(400, json={'error': {'message': 'structured output (response_format and the guided_* fields) is not supported by this engine yet'}})
            assert META not in json.dumps({k:v for k,v in data.items() if k != 'messages'})
            if check:
                check(data)
            return httpx.Response(200, json=outputs[min(len(sent)-1, len(outputs)-1)])
        raise AssertionError(request.url.path)
    settings = Settings(tensorfold_client_json=enabled, tensorfold_internal_json=True,
        backend_family='qwen38_vllm', deepseek_v41_vllm=True, qwen_native_vllm=True,
        served_model='deepseek', enforce_czech=enforce_czech, measure_reasoning_tokens=False,
        apply_sampling_defaults=False, api_key='offline-test', stream_heartbeat_seconds=.02)
    app = create_app(settings=settings, upstream_transport=httpx.MockTransport(backend))
    async with app.router.lifespan_context(app):
        if jobs is not None:
            app.state.vision_jobs = jobs
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gateway.test') as client:
            response = await client.post('/v1/responses' if api == 'responses' else '/v1/chat/completions',
                json=body, headers={'Authorization': 'Bearer offline-test'})
        app.state.vision_jobs = None
        assert not ACTIVE.get()
        assert app.state.engine.gate.active == 0
    return response, sent


def body_for(fmt, *, api='chat_completions', stream=False):
    body = {'model': 'deepseek', 'max_tokens': 256, 'stream': stream,
        'messages': [{'role': 'user', 'content': 'Return status ok and count 2.'}], **deepcopy(fmt)}
    if api == 'responses':
        body['input'] = body.pop('messages')
        body['max_output_tokens'] = body.pop('max_tokens')
        if 'response_format' in body:
            value = body.pop('response_format')
            if value['type'] == 'json_schema':
                value = {'type': 'json_schema', **value['json_schema']}
            body['text'] = {'format': value}
    return body


@pytest.mark.parametrize('fmt', FORMATS)
@pytest.mark.parametrize('api', ['chat_completions', 'responses'])
@pytest.mark.parametrize('stream', [False, True])
def test_client_formats_no_longer_reach_rejecting_engine(fmt, api, stream):
    response, sent = asyncio.run(exchange(body_for(fmt, api=api, stream=stream), api=api))
    assert response.status_code == 200, response.text
    assert len(sent) == 1
    assert 'Output contract:' in sent[0]['messages'][-1]['content']
    if stream:
        assert response.text.startswith(': dgx-gateway-heartbeat')
        assert '"count\\":2' in response.text or '\\"count\\":2' in response.text
        assert 'data: [DONE]' in response.text
        if api == 'responses':
            assert 'response.completed' in response.text
    else:
        assert response.headers['X-DGX-JSON-Validation'] in {'schema', 'object'}


@pytest.mark.parametrize('bad', ['{} trailing', '```json\n{}\n```', '{"count":NaN}',
    '{"count":Infinity}', '{"count":1e999}', '{"count":1,"count":2}', '[]', '{', '', '{"status":"ok","count":0}',
    '{"status":"ok","count":true}', '{"status":"bad","count":2}', '{"status":"ok","count":2,"extra":1}'])
def test_invalid_output_gets_one_bounded_repair(bad):
    response, sent = asyncio.run(exchange(body_for(FORMATS[1]), outputs=[completion(bad), completion()]))
    assert response.status_code == 200, response.text
    assert len(sent) == 2
    assert response.headers['X-DGX-JSON-Repair'] == '1'
    assert sent[1]['max_tokens'] == 246
    assert response.json()['usage']['completion_tokens'] == 20
    assert response.json()['usage']['prompt_tokens'] == 200


def test_no_invalid_success_after_failed_repair():
    response, sent = asyncio.run(exchange(body_for(FORMATS[1]), outputs=[completion('{')]))
    assert response.status_code == 502 and len(sent) == 2
    assert 'choices' not in response.json()


def test_truncation_cannot_exceed_total_output_budget():
    response, sent = asyncio.run(exchange(body_for(FORMATS[0]), outputs=[completion('{}', finish='length', used=256)]))
    assert response.status_code == 502 and len(sent) == 1
    assert response.json()['error']['code'] == 'json_output_budget_exhausted'


@pytest.mark.parametrize('fmt', [
    {'response_format': {'type': 'xml'}}, {'guided_regex': '.*'},
    {'response_format': {'type': 'json_schema', 'json_schema': {'schema': {'type': 'not-a-type'}}}},
    {'guided_json': {'$ref': 'https://example.invalid/schema'}},
    {'guided_json': {'$schema': 'https://example.invalid/dialect'}},
    {'guided_json': {'$ref': '#/$defs/missing'}},
    {'guided_json': SCHEMA, 'response_format': {'type': 'json_object'}},
])
def test_bad_or_unsupported_contract_rejected_before_inference(fmt):
    response, sent = asyncio.run(exchange(body_for(fmt)))
    assert response.status_code == 400 and not sent, response.text


@pytest.mark.parametrize('fmt', [{}, {'response_format': None}, {'response_format': {'type': 'text'}}])
def test_plain_text_not_parsed_or_regenerated(fmt):
    response, sent = asyncio.run(exchange(body_for(fmt), outputs=[completion('Normal text')]))
    assert response.status_code == 200 and len(sent) == 1
    assert sent[0]['messages'][-1]['content'] == 'Return status ok and count 2.'


def test_disabled_flag_retains_129_backend_behaviour():
    response, sent = asyncio.run(exchange(body_for(FORMATS[0]), enabled=False))
    assert response.status_code == 400 and len(sent) == 1


@pytest.mark.parametrize('api', ['chat_completions', 'responses'])
def test_tools_are_preserved_and_not_executed_or_schema_parsed(api):
    call = {'id': 'call_test', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"id":1}'}}
    body = body_for(FORMATS[1], api=api)
    body['tools'] = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}] if api != 'responses' else [{'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object'}}]
    response, sent = asyncio.run(exchange(body, api=api, outputs=[completion(None, finish='tool_calls', calls=[call])]))
    assert response.status_code == 200 and len(sent) == 1, response.text
    assert response.headers['X-DGX-JSON-Validation'] == 'tool_calls'
    assert sent[0]['tools'][0]['function']['name'] == 'lookup'


def test_long_history_sampling_images_and_input_immutable():
    body = body_for(FORMATS[1])
    body.update(temperature=.6, top_p=.95, seed=7)
    body['messages'] = [{'role': 'system', 'content': 'Stable policy'},
        {'role': 'user', 'content': 'large-context-' * 50000},
        {'role': 'assistant', 'content': 'Previous answer'},
        {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,fixture'}}, {'type': 'text', 'text': 'Inspect'}]}]
    original = deepcopy(body)
    adapted = adapt(body, contract_from(body))
    assert body == original
    assert adapted['messages'][:-1] == original['messages']
    assert all(adapted[k] == original[k] for k in ['temperature', 'top_p', 'seed', 'max_tokens'])


def test_full_schema_features_and_local_references():
    async def check():
        validator = Validator()
        schema = {'$defs': {'n': {'type': 'integer', 'minimum': 2}}, 'type': 'object',
            'required': ['kind', 'items'], 'properties': {'kind': {'const': 'a'},
                'items': {'type': 'array', 'minItems': 1, 'uniqueItems': True, 'items': {'$ref': '#/$defs/n'}}},
            'if': {'properties': {'kind': {'const': 'a'}}}, 'then': {'required': ['items']},
            'additionalProperties': False}
        contract = contract_from({'guided_json': schema})
        await validator.check(contract)
        await validator.check(contract, ['{"kind":"a","items":[2,3]}'])
        for text in ['{"kind":"a","items":[1]}', '{"kind":"a","items":[2,2]}', '{"kind":"b","items":[2]}']:
            with pytest.raises(JsonContractError):
                await validator.check(contract, [text])
        # Instance data containing a URL is not a remote schema reference.
        contract_from({'guided_json': {'const': {'$ref': 'https://example.invalid/data'}}})
    asyncio.run(check())


def test_validator_timeout_kills_process_and_releases_slot(monkeypatch):
    async def check():
        class Hung:
            returncode = None
            killed = False
            async def communicate(self, _):
                await asyncio.Future()
            def kill(self):
                self.killed = True
            async def wait(self):
                self.returncode = -1
        process = Hung()
        async def spawn(*args, **kwargs):
            return process
        monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
        validator = Validator(timeout=.01)
        with pytest.raises(JsonContractError, match='json_validation_timeout'):
            await validator.check(Contract('json_object', {'type':'object'}), ['{}'])
        assert process.killed and validator.slots._value == 2
    asyncio.run(check())


def test_cancellation_stops_generation_and_leaves_no_context():
    async def check():
        entered, cancelled = asyncio.Event(), asyncio.Event()
        async def dispatch(*_):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        task = asyncio.create_task(run(body_for(FORMATS[0]), Contract('json_object', {'type':'object'}),
            api='chat_completions', dispatch=dispatch, validator=Validator(), max_tokens=256, deadline=2))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set() and not ACTIVE.get()
    asyncio.run(check())


@pytest.mark.parametrize('api', ['chat_completions', 'responses'])
@pytest.mark.parametrize('valid', [True, False])
@pytest.mark.parametrize('stream', [True, False])
def test_durable_vision_validates_before_pollable_success(tmp_path, monkeypatch, api, valid, stream):
    from fastapi.testclient import TestClient
    from test_vision_jobs import Fake, IMAGE
    monkeypatch.setenv('DGX_VISION_JOBS_DB', str(tmp_path / 'jobs.sqlite3'))
    monkeypatch.setenv('DGX_MEDIA_SERIAL_REQUESTS', '1')
    monkeypatch.delenv('DGX_VISION_CONTROL_ENABLED', raising=False)
    fake = Fake()
    fake.content = '{"status":"ok","count":2}' if valid else '{"status":"bad"}'
    settings = Settings(tensorfold_client_json=True, upstream_base_url='http://fake', vision_upstream_base_url='http://fake',
        served_model='deepseek', backend_family='qwen38_vllm', deepseek_v41_vllm=True, qwen_native_vllm=True,
        chat_image_overflow_mode='off', enforce_czech=False, measure_reasoning_tokens=False)
    app = create_app(settings, upstream_transport=httpx.MockTransport(fake), vision_transport=httpx.MockTransport(fake))
    body = body_for(FORMATS[1], api=api, stream=stream)
    if api == 'responses':
        body['input'] = [{'role':'user','content':[{'type':'input_text','text':'Read status'}, {'type':'input_image','image_url':IMAGE['image_url']['url']}]}]
    else:
        body['messages'][0]['content'] = [{'type':'text','text':'Read status'}, deepcopy(IMAGE)]
    with TestClient(app) as client:
        response = client.post('/v1/responses' if api == 'responses' else '/v1/chat/completions', json=body,
            headers={'Idempotency-Key':'json-vision-test'})
        assert len(fake.calls) == 1
        assert 'response_format' not in fake.calls[0]
        assert 'metadata' not in fake.calls[0]
        if stream:
            assert ('client_json_schema_mismatch' in response.text) != valid
            if api == 'chat_completions':
                assert 'Private reasoning' not in response.text
        else:
            assert response.status_code == (200 if valid else 502), response.text
            if api == 'chat_completions':
                assert 'Private reasoning' not in response.text
            status = client.get(response.headers['location']).json()
            assert status['status'] == ('succeeded' if valid else 'failed')
            if not valid:
                assert 'choices' not in status['result']['data']


def test_declared_130_runtime_delta_only():
    from pathlib import Path
    import subprocess
    root = Path(__file__).resolve().parents[1]
    if not (root / 'vision-api').is_dir():
        pytest.skip('Repository-only source-scope gate; artifact hashes are verified separately')
    changed = set(subprocess.check_output(['git', 'diff', '--name-only', '8415ddf062a6a957e75b90d6c951b0adc5b8e229', '--', 'vision-api/laguna_gateway'], cwd=root, text=True).splitlines())
    allowed = {'app.py', 'config.py', 'engine.py', 'metrics.py', 'vision_jobs.py', 'vision_result.py', 'client_json.py', 'client_json_worker.py'}
    assert changed <= {'vision-api/laguna_gateway/' + name for name in allowed}


def test_invalid_json_never_leaks_before_stream_validation():
    response, sent = asyncio.run(exchange(body_for(FORMATS[1], stream=True), outputs=[completion('INVALID-FIRST'), completion()]))
    assert response.status_code == 200 and len(sent) == 2
    assert 'INVALID-FIRST' not in response.text
    assert 'data: [DONE]' in response.text


def test_refusal_is_explicit_and_never_regenerated():
    output = completion('')
    output['choices'][0]['message']['refusal'] = 'Cannot comply'
    response, sent = asyncio.run(exchange(body_for(FORMATS[1]), outputs=[output]))
    assert response.status_code == 422 and len(sent) == 1
    assert response.json()['error']['code'] == 'json_output_refusal'


def test_schema_array_is_allowed_unlike_json_object():
    body = body_for({'guided_json': {'type':'array','items':{'type':'integer'},'minItems':2}})
    response, sent = asyncio.run(exchange(body, outputs=[completion('[1,2]')]))
    assert response.status_code == 200 and len(sent) == 1


def test_oneof_and_unevaluated_properties_are_not_ignored():
    async def check():
        contract = Contract('json_schema', {'type':'object', 'oneOf':[
            {'required':['a'],'properties':{'a':{'type':'integer'}}},
            {'required':['b'],'properties':{'b':{'type':'string'}}}], 'unevaluatedProperties':False})
        validator = Validator()
        await validator.check(contract, ['{"a":1}'])
        for text in ['{"a":1,"b":"x"}', '{"a":1,"unknown":true}', '{"a":"bad"}']:
            with pytest.raises(JsonContractError):
                await validator.check(contract, [text])
    asyncio.run(check())


def test_partial_tool_call_is_never_regenerated_as_json():
    call = {'id':'call_partial','type':'function','function':{'name':'lookup','arguments':'{'}}
    response, sent = asyncio.run(exchange(body_for(FORMATS[0]), outputs=[completion(None, finish='length', calls=[call])]))
    assert response.status_code == 502 and len(sent) == 1


def test_czech_prose_repair_does_not_run_inside_json_contract():
    text = 'Here is the result. The following response is in English and describes what the system has found. ' * 8
    response, sent = asyncio.run(exchange(body_for(FORMATS[1]), outputs=[completion(text), completion()], enforce_czech=True))
    assert response.status_code == 200 and len(sent) == 2
    assert all('Output contract:' in item['messages'][-1]['content'] for item in sent)

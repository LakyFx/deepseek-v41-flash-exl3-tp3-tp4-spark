"""TensorFold client JSON compatibility, independent of internal checkpoint JSON.

Only validated final content is published. Ordinary traffic never enters this
path. Validation runs in a killable CPU subprocess with no external references.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import json
import sys
import time
from pathlib import Path

from .client_json_worker import strict_loads

ACTIVE = ContextVar('tensorfold_client_json', default=False)
META = '_tensorfold_client_json_v1'
MAX_SCHEMA_BYTES = 64 * 1024
MAX_RESULT_BYTES = 4 * 1024 * 1024
DIALECTS = {
    'https://json-schema.org/draft/2020-12/schema',
    'https://json-schema.org/draft/2019-09/schema',
    'http://json-schema.org/draft-07/schema',
    'http://json-schema.org/draft-06/schema',
    'http://json-schema.org/draft-04/schema',
}


class JsonContractError(ValueError):
    def __init__(self, reason, *, status=400):
        self.reason, self.status = reason, status
        super().__init__(reason)


@dataclass(frozen=True)
class Contract:
    mode: str
    schema: dict | bool

    def serialise(self):
        return {'mode': self.mode, 'schema': self.schema}


def contract_from(body, *, api='chat_completions'):
    """Return an explicit contract or None; never silently weaken constraints."""
    fmt = body.get('response_format')
    text = body.get('text')
    if api == 'responses' and isinstance(text, dict) and text.get('format') is not None:
        if fmt is not None:
            raise JsonContractError('conflicting_json_formats')
        fmt = text['format']
    guided = [k for k in body if k.startswith('guided_') and body[k] is not None]
    unsupported = set(guided) - {'guided_json'}
    if unsupported:
        raise JsonContractError('unsupported_guided_constraint')
    if body.get('guided_json') is not None:
        if fmt is not None:
            raise JsonContractError('conflicting_json_formats')
        schema = body['guided_json']
        try:
            if isinstance(schema, str):
                schema = strict_loads(schema)
        except ValueError as exc:
            raise JsonContractError('invalid_json_schema') from exc
        contract = Contract('json_schema', schema)
    elif fmt is None:
        return None
    elif not isinstance(fmt, dict):
        raise JsonContractError('invalid_response_format')
    elif fmt.get('type') == 'text':
        if set(fmt) != {'type'}:
            raise JsonContractError('unsupported_text_format_options')
        return None
    elif fmt.get('type') == 'json_object':
        if set(fmt) != {'type'}:
            raise JsonContractError('unsupported_json_object_options')
        contract = Contract('json_object', {'type': 'object'})
    elif fmt.get('type') == 'json_schema':
        if api != 'responses' and set(fmt) - {'type', 'json_schema'}:
            raise JsonContractError('unsupported_json_schema_options')
        spec = fmt.get('json_schema') if 'json_schema' in fmt else fmt if api == 'responses' else None
        if not isinstance(spec, dict) or 'schema' not in spec:
            raise JsonContractError('missing_json_schema')
        if set(spec) - {'type', 'name', 'description', 'strict', 'schema'}:
            raise JsonContractError('unsupported_json_schema_options')
        if 'strict' in spec and not isinstance(spec['strict'], bool):
            raise JsonContractError('invalid_schema_strict')
        contract = Contract('json_schema', spec['schema'])
    else:
        raise JsonContractError('unsupported_response_format')
    if not isinstance(contract.schema, (dict, bool)):
        raise JsonContractError('invalid_json_schema')
    if contract.schema is False:
        raise JsonContractError('unsatisfiable_json_schema')
    try:
        encoded = json.dumps(contract.schema, ensure_ascii=False, allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError) as exc:
        raise JsonContractError('invalid_json_schema') from exc
    if len(encoded) > MAX_SCHEMA_BYTES:
        raise JsonContractError('json_schema_too_large')
    # Traverse schema locations, not instance-valued const/enum/default/examples.
    def visit(value, depth=0):
        if depth > 64:
            raise JsonContractError('json_schema_too_deep')
        if not isinstance(value, dict):
            return
        dialect = value.get('$schema')
        if dialect is not None and str(dialect).rstrip('#') not in DIALECTS:
            raise JsonContractError('unsupported_json_schema_dialect')
        for key in ('$ref', '$dynamicRef', '$recursiveRef'):
            if key in value and (not isinstance(value[key], str) or not value[key].startswith('#')):
                raise JsonContractError('external_schema_reference_not_supported')
        if '$id' in value or 'id' in value:
            # Resource scopes change fragment semantics; only local-root schemas.
            raise JsonContractError('schema_resource_ids_not_supported')
        for key in ('properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas'):
            child = value.get(key)
            if isinstance(child, dict):
                for sub in child.values():
                    visit(sub, depth + 1)
        for key in ('allOf', 'anyOf', 'oneOf', 'prefixItems'):
            child = value.get(key)
            if isinstance(child, list):
                for sub in child:
                    visit(sub, depth + 1)
        for key in ('items', 'additionalItems', 'additionalProperties', 'unevaluatedProperties',
                    'unevaluatedItems', 'contains', 'propertyNames', 'not', 'if', 'then', 'else'):
            child = value.get(key)
            if isinstance(child, list):
                for sub in child:
                    visit(sub, depth + 1)
            else:
                visit(child, depth + 1)
    visit(contract.schema)
    return contract


def clean_format(body, *, api):
    body = deepcopy(body)
    body.pop('response_format', None)
    for key in list(body):
        if key.startswith('guided_'):
            body.pop(key)
    if api == 'responses' and isinstance(body.get('text'), dict):
        body['text'].pop('format', None)
        if not body['text']:
            body.pop('text')
    return body


def adapt(body, contract, *, api='chat_completions', repair=False):
    body = clean_format(body, api=api)
    instruction = (
        'Output contract: return ONLY one complete JSON value matching the following JSON Schema. '
        'No Markdown fences, comments, prose, duplicate keys, NaN or Infinity. '
        'Use the requested tools if needed; this contract applies to the final answer, not tool calls.\n'
        + json.dumps(contract.schema, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    )
    if repair:
        instruction += '\nThe previous generation did not satisfy this contract. Produce a complete corrected final answer.'
    # A late USER instruction preserves every original system/history message;
    # instruction coalescing cannot move it to the beginning of a 300K prefix.
    if api == 'responses':
        items = body.get('input', [])
        if isinstance(items, str):
            items = [{'role': 'user', 'content': items}]
        if not isinstance(items, list):
            raise JsonContractError('invalid_responses_input')
        body['input'] = [*items, {'role': 'user', 'content': instruction}]
    else:
        messages = body.get('messages')
        if not isinstance(messages, list):
            raise JsonContractError('invalid_messages')
        body['messages'] = [*messages, {'role': 'user', 'content': instruction}]
    metadata = body.setdefault('metadata', {})
    if not isinstance(metadata, dict):
        raise JsonContractError('invalid_metadata')
    metadata[META] = contract.serialise()
    body['stream'] = False
    body.pop('stream_options', None)
    return body


class Validator:
    def __init__(self, *, timeout=5.0):
        self.timeout = timeout
        self.slots = asyncio.Semaphore(2)

    async def check(self, contract, documents=None):
        payload = {'schema': contract.schema, 'object_only': contract.mode == 'json_object'}
        if documents is not None:
            payload['documents'] = documents
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        if len(raw) > 5 * 1024 * 1024:
            raise JsonContractError('json_result_too_large', status=502)
        process = None
        try:
            # Deadline covers waiting for a validator slot as well as execution.
            async with asyncio.timeout(self.timeout):
                async with self.slots:
                    spawn = asyncio.create_task(asyncio.create_subprocess_exec(
                        sys.executable, str(Path(__file__).with_name('client_json_worker.py')),
                        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    ))
                    try:
                        process = await asyncio.shield(spawn)
                    except asyncio.CancelledError:
                        # Cancellation may land after the OS created a process
                        # but before asyncio handed ownership to this coroutine.
                        import anyio
                        with anyio.CancelScope(shield=True):
                            process = await spawn
                        raise
                    output, _ = await process.communicate(raw)
                    if process.returncode or len(output) > 4096:
                        raise JsonContractError('json_validator_failed', status=503)
                    result = json.loads(output)
                    if not result.get('ok'):
                        raise JsonContractError(result.get('reason', 'invalid_json'), status=400 if documents is None else 502)
        except TimeoutError as exc:
            raise JsonContractError('json_validation_timeout', status=503) from exc
        finally:
            if process is not None and process.returncode is None:
                import anyio
                with anyio.CancelScope(shield=True):
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()


def documents(data, *, api):
    """Inspect final visible content, preserving intermediate tool/refusal protocol."""
    texts = []
    if not isinstance(data, dict):
        raise JsonContractError('invalid_json_envelope', status=502)
    if api == 'responses':
        if data.get('status') != 'completed' or data.get('error') or data.get('incomplete_details'):
            raise JsonContractError('json_output_incomplete', status=502)
        output = data.get('output')
        if not isinstance(output, list) or not output:
            raise JsonContractError('json_output_missing', status=502)
        calls, refusals = False, False
        for item in output:
            if not isinstance(item, dict):
                raise JsonContractError('invalid_json_envelope', status=502)
            calls |= item.get('type') in {'function_call', 'custom_tool_call'}
            if item.get('type') == 'message':
                parts = item.get('content') or []
                if not isinstance(parts, list) or any(not isinstance(p, dict) for p in parts):
                    raise JsonContractError('invalid_json_envelope', status=502)
                refusals |= any(p.get('type') == 'refusal' for p in parts)
                if any(not isinstance(p.get('text', ''), str) for p in parts if p.get('type') == 'output_text'):
                    raise JsonContractError('invalid_json_envelope', status=502)
                text = ''.join(p.get('text', '') for p in parts if p.get('type') == 'output_text')
                if text:
                    texts.append(text)
        if refusals:
            raise JsonContractError('json_output_refusal', status=422)
        if not texts and calls:
            return None
    else:
        choices = data.get('choices')
        if not isinstance(choices, list) or not choices:
            raise JsonContractError('json_output_missing', status=502)
        for choice in choices:
            if not isinstance(choice, dict) or not isinstance(choice.get('message'), dict):
                raise JsonContractError('invalid_json_envelope', status=502)
            message = choice.get('message') or {}
            if message.get('refusal'):
                raise JsonContractError('json_output_refusal', status=422)
            if choice.get('finish_reason') == 'tool_calls' and message.get('tool_calls'):
                continue
            if choice.get('finish_reason') != 'stop' or message.get('tool_calls'):
                raise JsonContractError('json_output_incomplete', status=502)
            texts.append(message.get('content'))
        if not texts:
            return None
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise JsonContractError('json_output_missing', status=502)
    try:
        size = sum(len(t.encode()) for t in texts)
    except UnicodeError as exc:
        raise JsonContractError('json_invalid_unicode', status=502) from exc
    if size > MAX_RESULT_BYTES:
        raise JsonContractError('json_result_too_large', status=502)
    return texts


def add_usage(data, previous, *, api):
    """Bill both real generations, including their repeated prompt tokens."""
    target = data.setdefault('usage', {})
    fields = ('input_tokens', 'output_tokens', 'total_tokens') if api == 'responses' else ('prompt_tokens', 'completion_tokens', 'total_tokens')
    for field in fields:
        target[field] = int(target.get(field) or 0) + int(previous.get(field) or 0)
    for key in ('completion_tokens_details', 'prompt_tokens_details', 'input_tokens_details', 'output_tokens_details'):
        if isinstance(previous.get(key), dict):
            details = target.setdefault(key, {})
            for field, value in previous[key].items():
                if isinstance(value, int) and not isinstance(value, bool):
                    details[field] = int(details.get(field) or 0) + value


async def run(body, contract, *, api, dispatch, validator, max_tokens, deadline, observe=lambda _: None):
    """Two generations maximum; each is re-admitted by the unchanged engine."""
    previous = {}
    token = ACTIVE.set(True)
    try:
        async with asyncio.timeout(deadline):
            for attempt in range(2):
                adapted = adapt(body, contract, api=api, repair=bool(attempt))
                if attempt:
                    cap = max_tokens - int(previous.get('output_tokens' if api == 'responses' else 'completion_tokens') or max_tokens)
                    if cap <= 0:
                        raise JsonContractError('json_output_budget_exhausted', status=502)
                    adapted.pop('max_completion_tokens', None)
                    adapted['max_output_tokens' if api == 'responses' else 'max_tokens'] = cap
                    observe('repair')
                result = await dispatch(adapted, attempt)
                if result.status_code >= 400:
                    return result  # Never regenerate transport/refusal/provider errors.
                if not isinstance(result.data, dict):
                    raise JsonContractError('invalid_json_envelope', status=502)
                usage = result.data.get('usage') or {}
                if not isinstance(usage, dict):
                    raise JsonContractError('invalid_json_envelope', status=502)
                try:
                    texts = documents(result.data, api=api)
                    if texts is not None:
                        await validator.check(contract, texts)
                except JsonContractError as exc:
                    observe('invalid')
                    calls = any(isinstance(c, dict) and isinstance(c.get('message'), dict) and c['message'].get('tool_calls') for c in result.data.get('choices') or [])
                    calls = calls or any(isinstance(item, dict) and item.get('type') in {'function_call', 'custom_tool_call'} for item in result.data.get('output') or [])
                    if attempt or calls or exc.status != 502 or exc.reason in {'json_result_too_large', 'invalid_json_envelope'}:
                        raise
                    previous = deepcopy(usage)
                    continue
                if attempt:
                    add_usage(result.data, previous, api=api)
                result.headers['X-DGX-JSON-Validation'] = 'tool_calls' if texts is None else 'schema' if contract.mode == 'json_schema' else 'object'
                result.headers['X-DGX-JSON-Repair'] = str(attempt)
                observe('tool_calls' if texts is None else 'valid')
                return result
    except TimeoutError as exc:
        raise JsonContractError('json_request_deadline', status=504) from exc
    finally:
        ACTIVE.reset(token)


async def frames(result, *, api, include_usage=True):
    """Emit a completed, validated result using the existing protocol vocabulary."""
    if result.status_code >= 400:
        from .app import _stream_error_event
        detail = result.data.get('error') or {}
        yield _stream_error_event(api=api, message=detail.get('message', 'JSON request failed'), code=detail.get('code', 'json_request_failed'), status_code=result.status_code)
        return
    data = result.data
    if api == 'responses':
        from .vision_protocol import response_events
        for frame in response_events(data):
            yield frame
    else:
        base = {'id': data.get('id', 'chatcmpl-json'), 'created': data.get('created', int(time.time())), 'model': data.get('model', ''), 'object': 'chat.completion.chunk'}
        choices = []
        for choice in data['choices']:
            delta = deepcopy(choice['message'])
            if delta.get('tool_calls'):
                delta['tool_calls'] = [{'index': i, **call} for i, call in enumerate(delta['tool_calls'])]
            choices.append({'index': choice.get('index', 0), 'delta': delta, 'finish_reason': None})
        yield ('data: ' + json.dumps({**base, 'choices': choices}, ensure_ascii=False) + '\n\n').encode()
        yield ('data: ' + json.dumps({**base, 'choices': [{'index': c.get('index', 0), 'delta': {}, 'finish_reason': c['finish_reason']} for c in data['choices']]}) + '\n\n').encode()
        if include_usage and data.get('usage'):
            yield ('data: ' + json.dumps({**base, 'choices': [], 'usage': data['usage']}) + '\n\n').encode()
    yield b'data: [DONE]\n\n'

"""Buffer one image inference; reasoning/partial loops never escape as an answer."""
import json
import re
import time
from copy import deepcopy

from .vision_jobs import error, repeated, standalone, API_FIELD
from .vision_diagnostics import Diagnostics, observe
from .vision_protocol import tool_names, valid_call


def final_text(text):
    text = re.sub(r'<(think|thinking|reasoning|analysis)>.*?</\1>', '', text, flags=re.I|re.S)
    return re.sub(r'<(?:think|thinking|reasoning|analysis)>.*$', '', text, flags=re.I|re.S).strip()


class Collector:
    def __init__(self, diagnostics=None, *, body=None):
        self.diagnostics = diagnostics
        self.pending = b''
        self.content = self.reasoning = ''
        self.finish = None
        self.done = False
        self.identity = None
        self.model = None
        self.usage = None
        self.repetitive = False
        self.agent = body is not None and not standalone(body)
        self.names = tool_names(body or {})
        self.calls = {}
        self.call_bytes = 0
        self.refusal = ''

    def feed(self, chunk):
        self.pending = (self.pending+chunk).replace(b'\r\n',b'\n')
        if len(self.pending) + len(self.content.encode()) + len(self.reasoning.encode()) + len(self.refusal.encode()) + self.call_bytes > 4*1024*1024:
            raise ValueError('Vision response exceeded retained result limit')
        while b'\n\n' in self.pending:
            block,self.pending = self.pending.split(b'\n\n',1)
            data = b'\n'.join(line[5:].lstrip() for line in block.splitlines() if line.startswith(b'data:'))
            if not data:
                continue
            if data == b'[DONE]':
                self.done = True
                continue
            payload = json.loads(data)
            if payload.get('error'):
                raise ValueError('Provider returned a streaming error')
            self.identity = payload.get('id') or self.identity
            self.model = payload.get('model') or self.model
            self.usage = payload.get('usage') or self.usage
            for choice in payload.get('choices') or []:
                delta = choice.get('delta') or {}
                if self.diagnostics is not None:
                    observe(self.diagnostics, 'observe', delta)
                if not self.repetitive:
                    self.content += delta.get('content') or ''
                    self.reasoning += delta.get('reasoning_content') or delta.get('reasoning') or ''
                    self.refusal += delta.get('refusal') or ''
                self.finish = choice.get('finish_reason') or self.finish
                if delta.get('tool_calls'):
                    if not self.names:
                        raise ValueError('Vision returned unrequested tool calls')
                    if not self.repetitive:
                        for call in delta['tool_calls']:
                            index = call.get('index')
                            if not isinstance(index,int) or isinstance(index,bool) or not 0 <= index < 128:
                                raise ValueError('Invalid vision tool-call index')
                            saved = self.calls.setdefault(index, {'id':'','type':'function','function':{'name':'','arguments':''}})
                            if call.get('type','function') != 'function':
                                raise ValueError('Invalid vision tool-call type')
                            for field in ('id',):
                                if call.get(field):
                                    if saved[field] and saved[field] != call[field]:
                                        raise ValueError('Conflicting vision tool-call identity')
                                    if not saved[field]: self.call_bytes += len(call[field].encode())
                                    saved[field] = call[field]
                            for field in ('name','arguments'):
                                value = (call.get('function') or {}).get(field,'')
                                if not isinstance(value,str): raise ValueError('Invalid tool-call fragment')
                                self.call_bytes += len(value.encode())
                                saved['function'][field] += value
            if not self.repetitive and (repeated(self.content) or repeated(self.reasoning) or repeated(self.refusal)):
                if self.diagnostics is not None:
                    observe(self.diagnostics, 'trigger', self.content, self.reasoning)
                # Reject the answer, but drain this SAME bounded generation.
                # Raising here closes its stream before a terminal receipt;
                # a backend without abort support then leaves every image job
                # blocked as UNKNOWN. Transport errors/deadlines still fail
                # closed through the existing engine and job worker.
                self.repetitive = True
                self.content = self.reasoning = ''
                self.refusal = ''
                self.calls.clear()

    def result(self, identity):
        if self.repetitive:
            if (not self.done or self.pending.strip()
                    or self.finish not in {'stop','length','content_filter','end_turn','completed','tool_calls'}):
                raise ValueError('Repetitive vision stream ended without confirmed termination')
            return {'status_code':422,'data':error('vision_output_repetitive',
                'Vision generation finished with repetitive output. Output was discarded; do not automatically regenerate.',
                identity),'headers':{}}
        content = final_text(self.content)
        calls = [self.calls[key] for key in sorted(self.calls)]
        valid_tools = bool(calls) and len({call['id'] for call in calls}) == len(calls) and all(
            valid_call(call['id'],call['function']['name'],call['function']['arguments'],self.names) for call in calls)
        if (not self.done or self.pending.strip() or self.finish not in {'stop','end_turn','completed','tool_calls'}
                or (calls and not valid_tools) or (self.finish == 'tool_calls' and not valid_tools)
                or not (content or valid_tools or self.refusal)):
            code = 'vision_output_truncated' if self.finish == 'length' else 'vision_output_incomplete'
            return {'status_code':422,'data':error(code,
                'Vision returned no complete final answer. Partial/reasoning content was discarded; do not automatically regenerate.',identity),'headers':{}}
        message = {'role':'assistant','content':content or None}
        if calls: message['tool_calls'] = calls
        if self.refusal: message['refusal'] = self.refusal
        if self.agent and self.reasoning: message['reasoning_content'] = self.reasoning
        data = {'id':self.identity or identity,'object':'chat.completion','created':int(time.time()),
                'model':self.model,'choices':[{'index':0,'message':message,'finish_reason':'tool_calls' if calls else 'stop'}]}
        if self.usage:
            data['usage'] = self.usage
        return {'status_code':200,'data':data,'headers':{}}


class ResponsesCollector(Collector):
    def __init__(self, diagnostics=None, *, body):
        super().__init__(diagnostics, body=body)
        self.response = None

    def feed(self, chunk):
        self.pending = (self.pending+chunk).replace(b'\r\n',b'\n')
        if len(self.pending)+len(self.content.encode())+len(self.reasoning.encode()) > 4*1024*1024:
            raise ValueError('Vision response exceeded retained result limit')
        while b'\n\n' in self.pending:
            block,self.pending = self.pending.split(b'\n\n',1)
            data = b'\n'.join(line[5:].lstrip() for line in block.splitlines() if line.startswith(b'data:'))
            if not data or data == b'[DONE]': continue
            event = json.loads(data)
            kind = event.get('type','')
            if kind in {'response.output_text.delta','response.reasoning_text.delta','response.reasoning_summary_text.delta'}:
                field = 'content' if kind == 'response.output_text.delta' else 'reasoning'
                value = event.get('delta') or ''
                if not isinstance(value,str): raise ValueError('Invalid Responses text delta')
                if not self.repetitive: setattr(self,field,getattr(self,field)+value)
                if self.diagnostics is not None:
                    observe(self.diagnostics, 'observe', {field if field=='content' else 'reasoning_content':value})
            if not self.repetitive and (repeated(self.content) or repeated(self.reasoning)):
                observe(self.diagnostics,'trigger',self.content,self.reasoning)
                self.repetitive=True
                self.content=self.reasoning=''
            if kind in {'response.completed','response.incomplete','response.failed'}:
                if self.done: raise ValueError('Duplicate terminal Responses event')
                self.done=True
                self.finish='completed' if kind=='response.completed' else 'length' if kind=='response.incomplete' else 'content_filter'
                self.response=event.get('response')

    def result(self, identity):
        if self.repetitive: return super().result(identity)
        data=deepcopy(self.response)
        valid = self.done and not self.pending.strip() and self.finish=='completed' and isinstance(data,dict)
        valid = valid and data.get('status')=='completed' and data.get('error') is None and isinstance(data.get('output'),list)
        useful=False; seen=set(); call_ids=set()
        if valid:
            for item in data['output']:
                if not isinstance(item,dict): valid=False; break
                kind=item.get('type')
                if kind=='reasoning':
                    text=' '.join(str(part.get('text','')) for key in ('content','summary') for part in item.get(key) or [] if isinstance(part,dict))
                    if repeated(text): valid=False; break
                    continue
                if not item.get('id') or item['id'] in seen: valid=False; break
                seen.add(item['id'])
                if kind=='message':
                    for part in item.get('content') or []:
                        field='refusal' if part.get('type')=='refusal' else 'text'
                        text=part.get(field) or ''
                        if not isinstance(text,str) or repeated(text): valid=False; break
                        part[field]=final_text(text)
                        useful=useful or bool(part[field])
                elif kind=='function_call':
                    valid=valid and valid_call(item.get('call_id'),item.get('name'),item.get('arguments'),self.names,item.get('namespace'))
                    valid=valid and item.get('call_id') not in call_ids
                    call_ids.add(item.get('call_id'))
                    useful=True
                elif kind=='custom_tool_call':
                    valid=valid and isinstance(item.get('call_id'),str) and bool(item['call_id']) and (item.get('namespace'),item.get('name')) in self.names and isinstance(item.get('input'),str)
                    valid=valid and item.get('call_id') not in call_ids
                    call_ids.add(item.get('call_id'))
                    useful=True
                else: valid=False
        if not valid or not useful:
            code='vision_output_truncated' if self.finish=='length' else 'vision_output_incomplete'
            return {'status_code':422,'data':error(code,'Vision returned no complete Responses answer or tool call.',identity),'headers':{}}
        return {'status_code':200,'data':data,'headers':{}}


async def collect(engine, row, on_backend, dispatch=None, on_hold=None):
    from .engine import GatewayError
    handle = None
    iterator = None
    outcome = 'vision_job_failed'
    diagnostics = None
    result = None
    result_status = None
    try:
        body = json.loads(row['body'])
        responses = body.pop(API_FIELD, None) == 'responses'
        body['stream'] = True
        body['stream_options'] = {'include_usage':True}
        begin = engine.begin_responses_stream if responses else engine.begin_stream
        close = engine.close_responses_stream if responses else engine.close_stream
        handle = await begin(body, **({'vision_dispatch':dispatch} if dispatch is not None else {}))
        if on_hold is not None:
            release=getattr(handle,'media_release',None)
            if release is not None:
                # Closing HTTP is not a physical termination proof. Transfer the
                # existing admission lease to this durable attempt until receipt.
                handle.media_release=None
                on_hold(release)
        on_backend(handle.upstream_request_id)
        if handle.response.status_code >= 400:
            await handle.response.aread()
            status=handle.response.status_code
            hints={400:'Check image format and request parameters.',401:'Operator must check backend authentication.',
                403:'Backend access was denied.',413:'Image/request exceeds the accepted size.',
                422:'Backend could not process this image/request.',429:'Backend admission is currently full.',
                503:'The vision backend is unavailable.'}
            data=error('vision_provider_rejected',hints.get(status,'The vision backend rejected this request.'),row['id'])
            data['error']['upstream_status']=status
            # Only machine codes, never raw messages (may contain URLs, credentials,
            # image data or internal paths). HTTP status remains available even if
            # the backend returned HTML or a malformed body.
            try:
                provider=handle.response.json().get('error',{})
                code=provider.get('code') if isinstance(provider,dict) else None
                if isinstance(code,str) and re.fullmatch(r'[a-zA-Z0-9_.-]{1,80}',code):
                    data['error']['upstream_code']=code
            except (ValueError,AttributeError): pass
            return {'status_code':422,'data':data,'headers':{}}
        diagnostics = Diagnostics.open(row['id'], handle.upstream_request_id, dispatch, attempt=row.get('attempt'))
        result = ResponsesCollector(diagnostics=diagnostics,body=body) if responses else Collector(diagnostics=diagnostics,body=body)
        iterator = engine.iter_responses_stream(handle) if responses else engine.iter_stream(handle)
        async for chunk in iterator:
            result.feed(chunk)
            if row.get('attempt') and result.repetitive and not result.done:
                from .vision_recovery import RecoveryRequired
                raise RecoveryRequired('vision_output_repetitive')
        if row.get('attempt') and (not result.done or result.pending.strip() or result.finish not in {'stop','length','content_filter','end_turn','completed','tool_calls'}):
            from .vision_recovery import RecoveryRequired
            raise RecoveryRequired('vision_output_incomplete')
        response = result.result(row['id'])
        if row.get('attempt'):
            response['_termination']={'kind':'terminal_stream','attempt':row['attempt']}
        result_status = response['status_code']
        outcome = 'success'
        headers = engine._responses_headers if responses else engine._response_headers
        response['headers'] = headers(handle.prepared,queue_wait_seconds=handle.queue_wait_seconds)
        return response
    except GatewayError as exc:
        if handle is None and exc.status_code in {400,401,403,404,413,422,429}:
            return {'status_code':422,'data':error('vision_input_rejected',str(exc),row['id']),'headers':{}}
        raise
    finally:
        if diagnostics is not None and result is not None:
            observe(diagnostics, 'finish', done=result.done, finish_reason=result.finish,
                repetitive=result.repetitive, usage=result.usage,
                result_status=result_status, outcome=outcome)
        if iterator is not None:
            await iterator.aclose()
        if handle is not None:
            await close(handle,outcome=outcome)

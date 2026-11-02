"""Actual gateway/engine paths for image-bearing agent turns, all offline."""
from copy import deepcopy
import json

import pytest

from test_vision_jobs import api, Fake, body, event, IMAGE
from test_vision_control import controlled
from test_vision_recovery import recovery, success_on_second
from laguna_gateway.vision_jobs import API_FIELD
from laguna_gateway.vision_recovery import repair_body
from laguna_gateway.vision_result import Collector, ResponsesCollector

TOOLS = [{'type':'function','function':{'name':'inspect','parameters':{'type':'object'}}}]


def agent(kind='tool_result', **kwargs):
    request=body(**{'tools':deepcopy(TOOLS),'temperature':.4,'top_p':.9,'seed':123,**kwargs})
    if kind=='none': request['tool_choice']='none'
    if kind in {'history','tool_result'}:
        request['messages'] += [{'role':'assistant','content':'Earlier observation'},
                                {'role':'user','content':'Inspect the next step'}]
    if kind=='tool_result':
        request['messages'] += [{'role':'assistant','content':None,'tool_calls':[
            {'id':'call_before','type':'function','function':{'name':'inspect','arguments':'{}'}}]},
            {'role':'tool','tool_call_id':'call_before','content':'prior tool result'}]
    return request


def responses_body(**kwargs):
    return {'model':'deepseek-v4.1-flash', 'instructions':'Original policy',
            'input':[{'role':'user','content':[{'type':'input_image','image_url':IMAGE['image_url']['url']},
                                              {'type':'input_text','text':'Inspect this'}]},
                     {'type':'function_call','call_id':'call_before','name':'inspect','arguments':'{}'},
                     {'type':'function_call_output','call_id':'call_before','output':'prior tool result'}],
            'tools':[{'type':'function','name':'inspect','parameters':{'type':'object'}}],
            'reasoning':{'effort':'medium'},'max_output_tokens':6144,'temperature':.4,'top_p':.9,'seed':123,
            **kwargs}


def tool_frame(identity='call_after', arguments='{"ok":true}', finish='tool_calls'):
    payload={'id':'chat-agent','model':'deepseek-v4.1-flash','choices':[{'index':0,
        'delta':{'tool_calls':[{'index':0,'id':identity,'type':'function',
                               'function':{'name':'inspect','arguments':arguments}}]},'finish_reason':finish}]}
    return ('data: '+json.dumps(payload)+'\n\ndata: [DONE]\n\n').encode()


def frames(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: {')]


@pytest.mark.parametrize('kind',['tools','history','tool_result','none'])
@pytest.mark.parametrize('stream',[False,True])
def test_agent_image_turn_uses_common_profile_and_preserves_history(recovery,kind,stream):
    client,fake,app,native=recovery
    request=agent(kind,stream=stream)
    original=deepcopy(request)
    result=client.post('/v1/chat/completions',json=request)
    assert result.status_code==200,result.text
    assert 'x-dgx-vision-job-id' in result.headers
    sent=native.posts[0]
    assert sent['messages']==original['messages'] and sent['tools']==original['tools']
    assert (sent['reasoning_effort'],sent['thinking_budget'],sent['max_tokens'])==('max',8192,40960)
    assert (sent['temperature'],sent['top_p'],sent['seed'])==(.4,.9,123)
    assert request==original and len(fake.calls)==1
    assert app.state.vision_jobs.ledger.counts()=={'succeeded':1}


@pytest.mark.parametrize('stream',[False,True])
def test_complete_tool_call_is_delivered_once_with_id_and_finish(recovery,stream):
    client,fake,_,native=recovery
    fake.frames=[tool_frame()]
    result=client.post('/v1/chat/completions',json=agent(stream=stream))
    assert result.status_code==200,result.text
    if stream:
        chunks=frames(result)
        message=chunks[0]['choices'][0]['delta']
        assert chunks[1]['choices'][0]['finish_reason']=='tool_calls'
    else:
        message=result.json()['choices'][0]['message']
        assert result.json()['choices'][0]['finish_reason']=='tool_calls'
    assert message['tool_calls']==[{**({'index':0} if stream else {}),'id':'call_after','type':'function',
        'function':{'name':'inspect','arguments':'{"ok":true}'}}]
    assert len(native.posts)==1
    again=client.post('/v1/chat/completions',json=agent(stream=stream))
    assert again.headers['x-dgx-vision-job-id']==result.headers['x-dgx-vision-job-id']
    assert len(native.posts)==1


@pytest.mark.parametrize('endpoint',['chat','responses'])
@pytest.mark.parametrize('stream',[False,True])
def test_partial_tool_call_never_escapes_repair_and_original_input_survives(recovery,monkeypatch,endpoint,stream):
    client,fake,app,native=recovery
    fake.frames=[tool_frame('discarded-call','{"unfinished":','length')]
    original_call=Fake.__call__
    async def serve(self,req):
        if req.url.path=='/v1/chat/completions' and fake.calls:
            fake.frames=[tool_frame('accepted-call')]
        return await original_call(self,req)
    monkeypatch.setattr(Fake,'__call__',serve)
    request=agent(stream=stream) if endpoint=='chat' else responses_body(stream=stream)
    original=deepcopy(request)
    path='/v1/chat/completions' if endpoint=='chat' else '/v1/responses'
    result=client.post(path,json=request)
    assert result.status_code==200,result.text
    assert 'discarded-call' not in result.text and 'unfinished' not in result.text
    assert len(native.posts)==2
    assert request==original
    attempts=app.state.vision_jobs.ledger.attempts(result.headers['x-dgx-vision-job-id'])
    assert [row['state'] for row in attempts]==['failed','succeeded']
    assert attempts[0]['deadline']==attempts[1]['deadline']
    repaired=json.loads(attempts[1]['body'])
    if endpoint=='chat':
        assert repaired['messages'][:-1]==request['messages']
        assert repaired['messages'][-1]['role']=='system'
    else:
        assert repaired['input']==request['input']
        assert repaired['instructions'].startswith('Original policy')
    for sent in native.posts:
        assert (sent['reasoning_effort'],sent['thinking_budget'],sent['max_tokens'])==('max',8192,40960)
        assert API_FIELD not in sent


@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('use_tool',[False,True])
def test_responses_image_preserves_protocol_and_common_vision_budget(recovery,stream,use_tool):
    client,fake,app,native=recovery
    if use_tool: fake.frames=[tool_frame()]
    request=responses_body(stream=stream)
    result=client.post('/v1/responses',json=request)
    assert result.status_code==200,result.text
    data=next(chunk['response'] for chunk in frames(result) if chunk['type']=='response.completed') if stream else result.json()
    assert data['object']=='response' and data['status']=='completed'
    assert any(item['type']==('function_call' if use_tool else 'message') for item in data['output'])
    if use_tool:
        call=next(item for item in data['output'] if item['type']=='function_call')
        assert call['name']=='inspect' and json.loads(call['arguments'])=={'ok':True}
        if stream:
            events=frames(result)
            done=next(event['item'] for event in events if event['type']=='response.output_item.done' and event['item']['type']=='function_call')
            assert done['call_id']==call['call_id']
    sent=native.posts[0]
    assert sent['messages'][-1]['role']=='tool' and sent['messages'][-1]['tool_call_id']=='call_before'
    assert IMAGE['image_url']['url'] in json.dumps(sent['messages'])
    assert (sent['reasoning_effort'],sent['thinking_budget'],sent['max_tokens'])==('max',8192,40960)
    assert (sent['temperature'],sent['top_p'],sent['seed'])==(.4,.9,123)
    saved=json.loads(app.state.vision_jobs.ledger.read(result.headers['x-dgx-vision-job-id'])['body'])
    assert saved[API_FIELD]=='responses' and saved['input']==request['input']


@pytest.mark.parametrize('endpoint',['chat','responses'])
def test_agent_loop_recovers_with_native_termination_receipt(recovery,monkeypatch,endpoint):
    client,fake,_,native=recovery
    fake.frames=[event(reasoning='Let me check this again. '*100)]
    fake.block_after_frames=True
    success_on_second(monkeypatch,fake)
    result=client.post('/v1/chat/completions' if endpoint=='chat' else '/v1/responses',
                       json=agent() if endpoint=='chat' else responses_body())
    assert result.status_code==200,result.text
    assert len(native.posts)==2 and native.cancels
    assert native.cancels[0]['request_id']==native.posts[0]['request_id']


def test_responses_failure_has_terminal_protocol_event_without_partial_answer(recovery):
    client,fake,_,native=recovery
    fake.finish='length'
    result=client.post('/v1/responses',json=responses_body(stream=True))
    failed=next(frame for frame in frames(result) if frame['type']=='response.failed')
    assert failed['response']['error']['attempts']==2
    assert failed['response']['output']==[] and 'Professor' not in result.text
    assert len(native.posts)==2


@pytest.mark.parametrize('endpoint',['chat','responses'])
def test_missing_native_control_does_not_dispatch_image_history(recovery,endpoint):
    client,fake,_,native=recovery
    native.available=False
    result=client.post('/v1/chat/completions' if endpoint=='chat' else '/v1/responses',
                       json=agent() if endpoint=='chat' else responses_body())
    assert result.status_code==503
    assert result.json()['error']['code']=='vision_control_unavailable'
    assert not native.posts and not fake.calls


def test_text_only_work_keeps_requested_budget_and_avoids_vision_jobs(recovery):
    client,fake,app,_=recovery
    request=body(max_tokens=131072,thinking_budget=16384,reasoning_effort='max',tools=TOOLS)
    request['messages']=[{'role':'user','content':'Continue text work'}]
    result=client.post('/v1/chat/completions',json=request)
    assert result.status_code==200 and 'x-dgx-vision-job-id' not in result.headers
    assert fake.calls[0]['max_tokens']==131072 and fake.calls[0]['thinking_budget']==16384
    assert app.state.vision_jobs.ledger.counts()=={}


def test_repair_preserves_responses_tool_pairing_and_sampling():
    request={**responses_body(),API_FIELD:'responses'}
    fixed=json.loads(repair_body(json.dumps(request)))
    assert {k:v for k,v in fixed.items() if k!='instructions'}=={k:v for k,v in request.items() if k!='instructions'}


def test_fragmented_tool_arguments_are_assembled_but_invalid_json_is_rejected():
    collector=Collector(body=agent())
    first=tool_frame(arguments='{"ok":')
    first=first.replace(b'"finish_reason": "tool_calls"',b'"finish_reason": null').replace(b'data: [DONE]\n\n',b'')
    collector.feed(first)
    assert collector.result('id')['status_code']==422
    second={'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'function':{'arguments':'true}'}}]},'finish_reason':'tool_calls'}]}
    collector.feed(('data: '+json.dumps(second)+'\n\ndata: [DONE]\n\n').encode())
    result=collector.result('id')
    assert result['status_code']==200
    assert result['data']['choices'][0]['message']['tool_calls'][0]['function']['arguments']=='{"ok":true}'


def test_n_multiple_images_is_rejected_before_dispatch(recovery):
    client,fake,_,native=recovery
    result=client.post('/v1/chat/completions',json=agent(n=2))
    assert result.status_code==409 and not fake.calls and not native.posts


def test_responses_internal_api_marker_cannot_be_spoofed_by_chat_client(recovery):
    client,_,app,_=recovery
    result=client.post('/v1/chat/completions',json={**agent(),API_FIELD:'responses'})
    assert result.status_code==200,result.text
    saved=json.loads(app.state.vision_jobs.ledger.read(result.headers['x-dgx-vision-job-id'])['body'])
    assert API_FIELD not in saved


@pytest.mark.parametrize('stream',[False,True])
def test_explicit_refusal_is_a_complete_answer_not_a_repair_loop(recovery,stream):
    client,fake,_,native=recovery
    payload={'id':'chat-refusal','model':'deepseek-v4.1-flash','choices':[{
        'index':0,'delta':{'refusal':'I cannot fulfill this request.'},'finish_reason':'stop'}]}
    fake.frames=[('data: '+json.dumps(payload)+'\n\ndata: [DONE]\n\n').encode()]
    result=client.post('/v1/chat/completions',json=agent(stream=stream))
    assert result.status_code==200 and 'I cannot fulfill' in result.text
    assert len(native.posts)==1


def test_namespaced_response_function_call_keeps_namespace_and_identity(recovery):
    from laguna_gateway.custom_tools import CustomTools
    client,fake,_,_=recovery
    request=responses_body()
    request['tools']=[{'type':'namespace','name':'viewer','tools':[
        {'type':'function','name':'inspect','parameters':{'type':'object'}}]}]
    payload=json.loads(tool_frame().split(b'\n\n')[0][6:])
    payload['choices'][0]['delta']['tool_calls'][0]['function']['name']=CustomTools.namespace_alias('viewer','inspect')
    fake.frames=[('data: '+json.dumps(payload)+'\n\ndata: [DONE]\n\n').encode()]
    result=client.post('/v1/responses',json=request)
    assert result.status_code==200,result.text
    call=next(item for item in result.json()['output'] if item['type']=='function_call')
    assert call['namespace']=='viewer' and call['name']=='inspect'


def test_responses_custom_tool_input_is_not_mistaken_for_function_json(recovery):
    client,fake,_,_=recovery
    request=responses_body()
    request['tools']=[{'type':'custom','name':'inspect','format':{'type':'text'}}]
    fake.frames=[tool_frame(arguments=json.dumps({'input':'display image.png'}))]
    result=client.post('/v1/responses',json=request)
    assert result.status_code==200,result.text
    call=next(item for item in result.json()['output'] if item['type']=='custom_tool_call')
    assert call['name']=='inspect' and call['input']=='display image.png'


def test_unconfirmed_responses_attempt_holds_one_slot_and_chat_image_still_works(recovery):
    client,fake,app,native=recovery
    native.confirm=False
    fake.frames=[event(reasoning='Let me check this again. '*100)]
    fake.block_after_frames=True
    app.state.vision_jobs.ledger.job_timeout=.12
    result=client.post('/v1/responses',json=responses_body())
    assert result.status_code==409 and len(native.posts)==1
    assert app.state.vision_jobs.engine._media_request_gate._value==1
    fake.frames=None; fake.block_after_frames=False
    app.state.vision_jobs.ledger.job_timeout=3590
    next_result=client.post('/v1/chat/completions',json=agent(seed=456))
    assert next_result.status_code==200,next_result.text
    assert len(native.posts)==2
    assert app.state.vision_jobs.engine._media_request_gate._value==1


@pytest.mark.parametrize('endpoint',['chat','responses'])
def test_image_inside_tool_result_is_not_missed(recovery,endpoint):
    client,_,_,native=recovery
    if endpoint=='chat':
        request=agent()
        request['messages'][0]['content']='Inspect a screenshot via a tool'
        request['messages'][-1]['content']=[deepcopy(IMAGE)]
    else:
        request=responses_body()
        request['input'][0]['content']='Inspect a screenshot via a tool'
        request['input'][-1]['output']=[{'type':'input_image','image_url':IMAGE['image_url']['url']}]
    result=client.post('/v1/chat/completions' if endpoint=='chat' else '/v1/responses',json=request)
    assert result.status_code==200,result.text
    assert 'x-dgx-vision-job-id' in result.headers
    sent=native.posts[0]
    assert sent['max_tokens']==40960 and sent['thinking_budget']==8192
    assert IMAGE['image_url']['url'] in json.dumps(sent['messages'])


@pytest.mark.parametrize('selectors',[{'reasoning_effort':'none'},
    {'reasoning_effort':'none','chat_template_kwargs':{'enable_thinking':False}},
    {'reasoning_effort':'none','thinking_token_budget':0}])
def test_explicit_vision_profile_consistently_overrides_agent_thinking_off(recovery,selectors):
    client,_,_,native=recovery
    result=client.post('/v1/chat/completions',json=agent(**selectors))
    assert result.status_code==200,result.text
    sent=native.posts[0]
    assert sent['reasoning_effort']=='max' and sent['thinking_budget']==8192
    assert sent['chat_template_kwargs']=={'thinking':True,'reasoning_effort':'max'}

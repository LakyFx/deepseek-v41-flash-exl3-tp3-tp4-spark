import asyncio
import json
import time
import pytest
from test_vision_jobs import api, Fake, body, event, IMAGE
from test_vision_control import controlled, Native, eventually
from laguna_gateway.vision_recovery import RecoveryLedger, failure, repair_body
from laguna_gateway.vision_control import SCHEMA

@pytest.fixture
def recovery(monkeypatch,request):
    monkeypatch.setenv('DGX_VISION_RECOVERY_ATTEMPTS','2')
    monkeypatch.setenv('DGX_VISION_REASONING_EFFORT','max')
    monkeypatch.setenv('DGX_VISION_THINKING_BUDGET','8192')
    monkeypatch.setenv('DGX_VISION_FINAL_ANSWER_TOKENS','32768')
    return request.getfixturevalue('controlled')

def success_on_second(monkeypatch,fake):
    original=Fake.__call__
    async def serve(self,req):
        if req.url.path=='/v1/chat/completions' and fake.calls:
            fake.frames=None; fake.block_after_frames=False
        return await original(self,req)
    monkeypatch.setattr(Fake,'__call__',serve)

def test_loop_cancel_receipt_then_one_success_quality_and_dedup_preserved(recovery,monkeypatch):
    client,fake,app,native=recovery
    fake.frames=[event(reasoning='Let me check this again. '*100)]
    fake.block_after_frames=True
    success_on_second(monkeypatch,fake)
    result=client.post('/v1/chat/completions',json=body())
    assert result.status_code==200,result.text
    identity=result.headers['x-dgx-vision-job-id']
    attempts=app.state.vision_jobs.ledger.attempts(identity)
    assert [a['state'] for a in attempts]==['failed','succeeded']
    assert len(native.posts)==2 and native.cancels
    assert native.posts[0]['request_id']!=native.posts[1]['request_id']
    assert native.cancels[0]['request_id']==native.posts[0]['request_id']
    for sent in native.posts:
        assert sent['max_tokens']==40960 and sent['thinking_budget']==8192 and sent['reasoning_effort']=='max'
        assert IMAGE in sent['messages'][-1]['content']
    assert attempts[0]['deadline']==attempts[1]['deadline']
    assert 'previous generation' in attempts[1]['body'] and 'Let me check' not in attempts[1]['body']
    assert client.post('/v1/chat/completions',json=body()).status_code==200
    assert len(native.posts)==2

def test_two_bad_final_answers_exhaust_budget_without_automatic_third(recovery):
    client,fake,app,native=recovery
    fake.finish='length'
    for _ in range(3):
        result=client.post('/v1/chat/completions',json=body())
        assert result.status_code==422
        err=result.json()['error']
        assert err['code']=='vision_output_truncated' and err['attempts']==2 and not err['retryable']
    assert len(native.posts)==2
    assert not native.cancels # Complete terminal streams already proved both ended.
    assert app.state.vision_jobs.ledger.counts()=={'failed':1}

def test_unknown_retains_one_slot_joins_duplicates_and_second_lane_works(recovery):
    client,fake,app,native=recovery
    native.confirm=False
    fake.frames=[event(reasoning='Let me check this again. '*100)]; fake.block_after_frames=True
    first=client.post('/v1/vision/jobs',json=body()).json()['job_id']
    ledger=app.state.vision_jobs.ledger
    eventually(lambda:ledger.read(first)['state']=='unknown')
    assert client.post('/v1/vision/jobs',json=body()).json()['job_id']==first
    assert len(native.posts)==1 and not ledger.blocked()
    assert app.state.vision_jobs.engine._media_request_gate._value==1
    fake.frames=None; fake.block_after_frames=False
    assert client.post('/v1/chat/completions',json=body(seed=42)).status_code==200
    assert len(native.posts)==2
    assert len(ledger.attempts(first))==1
    assert app.state.vision_jobs.engine._media_request_gate._value==1

def dispatch(ledger,row):
    d={'instance':'test','request_id':f"request-{row['attempt']}",'expires':row['deadline']}
    ledger.dispatch(row['id'],d,attempt=row['attempt']); return d

def receipt(d):
    return {'schema':SCHEMA,'instance':d['instance'],'requested_instance':d['instance'],
            'request_id':d['request_id'],'termination_confirmed':True,'state':'terminated'}

def test_restart_uses_durable_attempt_and_stale_dispatch_or_receipt_cannot_touch_next(tmp_path):
    l=RecoveryLedger(tmp_path/'db',controlled=True)
    identity,_=l.accept('o',body()); a=l.claim(); d=dispatch(l,a)
    l.unresolved(a,failure(identity,'vision_output_repetitive'))
    l=RecoveryLedger(l.path,controlled=True); l.recover()
    assert l.confirmed(identity,receipt(d))
    b=l.claim(); assert b['attempt']==2 and b['deadline']==a['deadline']
    with pytest.raises(ValueError,match='owned'): l.dispatch(identity,d,attempt=1)
    second=dispatch(l,b)
    l.unresolved(b,failure(identity,'vision_output_incomplete'))
    with pytest.raises(ValueError,match='proof'): l.confirmed(identity,receipt(d))
    l.confirmed(identity,receipt(second))
    assert l.read(identity)['state']=='failed' and l.claim() is None
    assert len(l.attempts(identity))==2

def test_total_deadline_and_pre_dispatch_restart_do_not_reset_budget(tmp_path):
    clock=[1000.0]; l=RecoveryLedger(tmp_path/'db',clock=lambda:clock[0],controlled=True)
    identity,_=l.accept('o',body()); a=l.claim(); l.recover()
    l.not_dispatched(identity,'vision_interrupted',attempt=1)
    b=l.claim(); assert b['deadline']==a['deadline']
    l.unresolved(b,failure(identity,'vision_transport_failed'))
    clock[0]=b['deadline']+1
    l.not_dispatched(identity,'vision_interrupted',attempt=2)
    assert l.read(identity)['state']=='failed' and l.claim() is None

def test_repair_preserves_all_sampling_images_and_question():
    original=body(seed=123,temperature=0,top_p=.9,thinking_budget=8192,max_tokens=40960,reasoning_effort='max')
    repaired=json.loads(repair_body(json.dumps(original)))
    assert {k:v for k,v in original.items() if k!='messages'}=={k:v for k,v in repaired.items() if k!='messages'}
    assert repaired['messages'][0]['content'][:-1]==original['messages'][0]['content']

def test_retained_legacy_failure_is_reconciled_on_client_retry_not_blindly_replayed(tmp_path):
    from laguna_gateway.vision_jobs import Ledger
    old=Ledger(tmp_path/'db',controlled=True)
    identity,_=old.accept('o',body(),'stable'); row=old.claim()
    d={'instance':'test','request_id':'old','expires':row['deadline']}
    old.dispatch(identity,d); old.finish(identity,'failed',failure(identity,'vision_output_repetitive'))
    l=RecoveryLedger(old.path,controlled=True)
    assert l.accept('o',body(),'stable')==(identity,False)
    assert l.read(identity)['state']=='unknown' and l.claim() is None
    l.confirmed(identity,receipt(d))
    second=l.claim(); assert second['attempt']==2
    assert second['deadline']==row['deadline']
    l.resolve(identity,2,failure(identity,'vision_output_truncated'),{'kind':'terminal_stream','attempt':2})
    assert l.accept('o',body(),'stable')==(identity,False)
    assert l.read(identity)['state']=='failed' and l.claim() is None

def test_control_unavailable_explains_failure_and_never_dispatches(recovery):
    client,fake,app,native=recovery; native.available=False
    response=client.post('/v1/chat/completions',json=body())
    assert response.status_code==503
    assert response.json()['error']['code']=='vision_control_unavailable'
    assert response.json()['error']['attempts']==1
    assert not native.posts and not app.state.vision_jobs.ledger.blocked()

def test_ledger_migration_preserves_completed_results_and_stale_pending_marker(tmp_path):
    from laguna_gateway.vision_jobs import Ledger
    old=Ledger(tmp_path/'db',controlled=True)
    identity,_=old.accept('o',body()); old.claim()
    result={'status_code':200,'headers':{},'data':{'answer':'saved'}}
    old.finish(identity,'succeeded',result)
    new=RecoveryLedger(old.path,controlled=True); new.recover()
    assert new.status(identity,'o')['result']==result and new.accept('o',body())==(identity,False)
    second,_=new.accept('o',body(seed=2)); a=new.claim(); new.recover()
    new.not_dispatched(second,'vision_interrupted',attempt=1); b=new.claim()
    new.not_dispatched(second,'vision_interrupted',attempt=1)
    assert b['attempt']==2 and new.read(second)['state']=='running'

def test_provider_error_retains_safe_status_code_but_not_raw_content(recovery,monkeypatch):
    import httpx
    client,fake,app,native=recovery
    original=Fake.__call__
    async def serve(self,req):
        if req.url.path=='/v1/chat/completions':
            return httpx.Response(413,json={'error':{'code':'image_too_large','message':'SECRET data:image/png;base64,ABC'}})
        return await original(self,req)
    monkeypatch.setattr(Fake,'__call__',serve)
    response=client.post('/v1/chat/completions',json=body())
    assert response.status_code==422
    err=response.json()['error']
    assert err['upstream_status']==413 and err['upstream_code']=='image_too_large'
    assert 'SECRET' not in response.text and 'data:image' not in response.text
    assert len(app.state.vision_jobs.ledger.attempts(err['job_id']))==1

def test_rollback_closes_only_confirmed_queued_repair_and_refuses_unknown(tmp_path):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).parents[1]/'rollback_402.py'
    spec=importlib.util.spec_from_file_location('rollback401a',path); rb=importlib.util.module_from_spec(spec);spec.loader.exec_module(rb)
    l=RecoveryLedger(tmp_path/'db',controlled=True)
    identity,_=l.accept('o',body()); a=l.claim(); d=dispatch(l,a)
    l.unresolved(a,failure(identity,'vision_output_repetitive'))
    with pytest.raises(ValueError,match='Reconcile'): rb.prepare(l.path,True)
    l.confirmed(identity,receipt(d))
    assert rb.prepare(l.path)['pending_repairs_closed']==1
    assert l.read(identity)['state']=='queued'
    rb.prepare(l.path,True)
    assert l.read(identity)['state']=='failed'

def test_restart_reserves_physical_media_slot_before_accepting_any_new_image(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from laguna_gateway.vision_jobs import Jobs
    monkeypatch.setenv('DGX_VISION_RECOVERY_ATTEMPTS','2')
    monkeypatch.setenv('DGX_MEDIA_SERIAL_REQUESTS','1')
    monkeypatch.setenv('DGX_MEDIA_MAX_CONCURRENCY','2')
    l=RecoveryLedger(tmp_path/'db',controlled=True)
    identity,_=l.accept('o',body()); a=l.claim(); d=dispatch(l,a)
    async def run():
        class Control:
            async def cancel(self,dispatch): return None
        jobs=Jobs(l.path,SimpleNamespace(),control=Control())
        await jobs.start()
        try:
            assert jobs.engine._media_request_gate._value==1
            assert jobs.ledger.read(identity)['state']=='unknown'
            jobs.ledger.confirmed(identity,receipt(d))
            jobs.release_confirmed_media()
            assert jobs.engine._media_request_gate._value==2
        finally: await jobs.close()
    asyncio.run(run())

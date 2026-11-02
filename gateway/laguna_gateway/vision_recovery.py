"""Opt-in bounded recovery of logical image jobs, with durable physical attempts.

No retry until a complete terminal stream or a native cancellation receipt.
The original request, sampling and output budgets remain immutable. Recovery adds
one short system instruction, never the failed reasoning or a synthetic tool result.
"""
import asyncio
import copy
import json
import logging

from .vision_jobs import Ledger, AdmissionError, error, API_FIELD, standalone
from .vision_control import confirms_termination

log=logging.getLogger(__name__)
REPAIRABLE={'vision_output_repetitive','vision_output_truncated','vision_output_incomplete',
            'vision_transport_failed','vision_interrupted'}
REPAIR_NOTE=('Please complete the same image-analysis request. A previous generation did not '
             'produce a usable final answer. Inspect the image afresh, avoid repeating planning '
             'phrases, and give a clear final answer addressing the original question. '
             'State any visual uncertainty explicitly.')
AGENT_REPAIR_NOTE=(REPAIR_NOTE+' Preserve the original agent instructions and tool-result history. '
                  'Return a complete declared tool call if the next step requires a tool; never invent its result.')

class RecoveryRequired(Exception):
    def __init__(self,code): self.code=code; super().__init__(code)

def repair_body(raw):
    body=json.loads(raw)
    body=copy.deepcopy(body)
    if body.get(API_FIELD) == 'responses':
        # Keep input/function_call_output pairing and the entire user prefix
        # byte-for-byte. A repair must not append a user turn after a tool result.
        body['instructions'] = (body.get('instructions') or '') + '\n\n' + AGENT_REPAIR_NOTE
        return json.dumps(body,ensure_ascii=False,separators=(',',':'))
    if not standalone(body):
        body['messages'] = [*body['messages'], {'role':'system','content':AGENT_REPAIR_NOTE}]
        return json.dumps(body,ensure_ascii=False,separators=(',',':'))
    for message in reversed(body['messages']):
        if message['role']=='user':
            if isinstance(message['content'],list):
                message['content'].append({'type':'text','text':REPAIR_NOTE})
            else: message['content']+='\n\n'+REPAIR_NOTE
            break
    return json.dumps(body,ensure_ascii=False,separators=(',',':'))

def failure(identity,code,status=422):
    messages={
      'vision_output_repetitive':'Image generation entered a repetition loop; no complete analysis was produced.',
      'vision_output_truncated':'Image generation exhausted its output budget before completing the analysis.',
      'vision_output_incomplete':'Image generation ended without a complete final analysis.',
      'vision_transport_failed':'The image stream was interrupted; its generation must be reconciled before another attempt.',
      'vision_interrupted':'Gateway interrupted this image attempt; native termination must be confirmed.',
      'vision_deadline':'The total image-analysis deadline expired.',
    }
    return {'status_code':status,'headers':{},'data':error(code,messages.get(code,'Image analysis failed.'),identity)}

def code_of(result): return (result.get('data',{}).get('error') or {}).get('code')

class RecoveryLedger(Ledger):
    """Additive schema. Legacy dispatch remains the current-attempt compatibility view.

    attempt generation is checked on dispatch and result commits to reject late
    callbacks. Original jobs.body is never rewritten. Legacy active/unknown work
    is reconciled by the existing contract; it is not silently replayed.
    """
    attempt_limit = 2
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not self.controlled: raise ValueError('Vision recovery requires native cancellation control')
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS vision_attempts (
                job_id TEXT NOT NULL, number INTEGER NOT NULL, state TEXT NOT NULL,
                body TEXT NOT NULL, deadline REAL NOT NULL, dispatch TEXT, receipt TEXT,
                result TEXT, created REAL NOT NULL, updated REAL NOT NULL,
                PRIMARY KEY(job_id,number))''')
    def attempts(self,identity, *, include_body=True):
        with self.connect() as db:
            rows = [dict(r) for r in db.execute('SELECT * FROM vision_attempts WHERE job_id=? ORDER BY number',(identity,))]
            if include_body and any(not r['body'] for r in rows):
                archived = self.storage.hydrate(db, identity)
                for row in rows:
                    if not row['body']:
                        row['body'] = archived['attempts'][str(row['number'])]
            return rows
    def accept(self,owner,body,key=None,*,generation_body=None):
        identity,created=super().accept(owner,body,key,generation_body=generation_body)
        if created: return identity,created
        # A retained failure from the previous gateway may receive one repair,
        # but only after reconciling its original dispatch. Explicit retries do
        # not reset the original deadline and cannot replenish attempt credits.
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM jobs WHERE id=?',(identity,)).fetchone()
            old=db.execute('SELECT 1 FROM vision_attempts WHERE job_id=?',(identity,)).fetchone()
            result=json.loads(row['result']) if row['result'] else {}
            d=db.execute('SELECT body FROM vision_dispatch WHERE job_id=?',(identity,)).fetchone()
            if (row['state']=='failed' and not old and code_of(result) in REPAIRABLE and
                    row['deadline']>self.clock() and d and not json.loads(d['body']).get('pending')):
                retained = self.storage.rows(db)[identity]['used']
                extra = max(0, self.storage.charge(len(row['body'].encode()))-retained)
                if not self.storage.make_room(db, extra):
                    raise AdmissionError('Existing image job retained; storage cannot reserve its recovery. '
                                         'Keep this job ID: '+identity)
                db.execute('INSERT INTO vision_attempts VALUES (?,?,?,?,?,?,NULL,?,?,?)',
                    (identity,1,'unknown',row['body'],row['deadline'],d['body'],json.dumps(result),row['created'],self.clock()))
                db.execute("UPDATE jobs SET state='unknown',updated=? WHERE id=?",(self.clock(),identity))
        return identity,False
    def claim(self):
        now=self.clock()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("UPDATE jobs SET state='failed',updated=?,result=? WHERE state='queued' AND deadline<=?",
                       (now,json.dumps(failure(None,'vision_deadline',504)),now))
            if self.legacy_unknown(db) or db.execute("SELECT count(*) FROM jobs WHERE state IN ('running','unknown')").fetchone()[0]>=self.max_running:
                return None
            row=db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY updated,created,rowid LIMIT 1").fetchone()
            if row is None: return None
            old=db.execute('SELECT * FROM vision_attempts WHERE job_id=? ORDER BY number DESC LIMIT 1',(row['id'],)).fetchone()
            number=old['number']+1 if old else 1
            if number>2: raise ValueError('Recovery attempt cap exceeded')
            deadline=old['deadline'] if old else now+self.job_timeout
            body=repair_body(row['body']) if old else row['body']
            # Admission already reserves the original plus BOTH physical body
            # copies and results. The same accounting applies at dispatch.
            if not self.storage.make_room(db):
                db.execute("UPDATE jobs SET state='failed',updated=?,result=? WHERE id=?",
                    (now,json.dumps(failure(row['id'],'vision_ledger_capacity',503)),row['id']))
                return None
            db.execute('INSERT INTO vision_attempts VALUES (?,?,?,?,?,NULL,NULL,NULL,?,?)',
                       (row['id'],number,'running',body,deadline,now,now))
            db.execute('''INSERT INTO vision_dispatch(job_id,body,receipt) VALUES (?,?,NULL)
                ON CONFLICT(job_id) DO UPDATE SET body=excluded.body,receipt=NULL''',(row['id'],json.dumps({'pending':True})))
            db.execute("UPDATE jobs SET state='running',updated=?,deadline=?,result=NULL,backend_id=NULL WHERE id=?",(now,deadline,row['id']))
            return {**dict(row),'state':'running','deadline':deadline,'body':body,'attempt':number}
    def dispatch(self,identity,dispatch,attempt=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            job=db.execute('SELECT state FROM jobs WHERE id=?',(identity,)).fetchone()
            current=db.execute('SELECT * FROM vision_attempts WHERE job_id=? ORDER BY number DESC LIMIT 1',(identity,)).fetchone()
            if not job or job['state']!='running' or not current or current['number']!=attempt or current['dispatch']:
                raise ValueError('Vision dispatch is no longer owned by this attempt')
            raw=json.dumps(dispatch)
            db.execute('UPDATE vision_attempts SET dispatch=?,updated=? WHERE job_id=? AND number=?',(raw,self.clock(),identity,attempt))
            db.execute('UPDATE vision_dispatch SET body=?,receipt=NULL WHERE job_id=?',(raw,identity))
            db.execute('UPDATE jobs SET backend_id=?,updated=? WHERE id=?',(dispatch['request_id'],self.clock(),identity))
    def backend(self,identity,backend_id,attempt=None):
        with self.connect() as db:
            db.execute('''UPDATE jobs SET backend_id=?,updated=? WHERE id=? AND state='running'
                AND ?=(SELECT max(number) FROM vision_attempts WHERE job_id=?)''',
                       (backend_id,self.clock(),identity,attempt,identity))
    def unresolved(self,row,result):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current=db.execute('SELECT max(number) FROM vision_attempts WHERE job_id=?',(row['id'],)).fetchone()[0]
            if current!=row['attempt']: return
            db.execute("UPDATE vision_attempts SET state='unknown',result=?,updated=? WHERE job_id=? AND number=?",
                       (json.dumps(result),self.clock(),row['id'],current))
            db.execute("UPDATE jobs SET state='unknown',result=?,updated=? WHERE id=? AND state='running'",
                       (json.dumps({'status_code':409,'headers':{},'data':error('termination_unconfirmed',
                        'The image attempt is being reconciled. Keep this job ID; recovery waits for confirmed termination.',row['id'])}),self.clock(),row['id']))
    def resolve(self,identity,number,result,proof):
        """One transaction: persist result/proof, then queue one repair or finish."""
        now=self.clock()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            job=db.execute('SELECT * FROM jobs WHERE id=?',(identity,)).fetchone()
            a=db.execute('SELECT * FROM vision_attempts WHERE job_id=? ORDER BY number DESC LIMIT 1',(identity,)).fetchone()
            if not a or a['number']!=number or a['state'] not in {'running','unknown'}: return False
            if job['state'] not in {'running','unknown'}: return False
            dispatch=json.loads(a['dispatch']) if a['dispatch'] else None
            valid=(proof.get('kind')=='terminal_stream' and proof.get('attempt')==number and a['state']=='running')
            valid=valid or (proof.get('kind')=='not_dispatched' and dispatch is None)
            valid=valid or (dispatch is not None and confirms_termination(proof,dispatch))
            if not valid: raise ValueError('Retry requires termination proof for this attempt')
            result=copy.deepcopy(result); result.pop('_termination',None)
            ok=result['status_code']==200
            repair=not ok and code_of(result) in REPAIRABLE and number<2 and now< a['deadline']
            if not ok:
                detail=result.setdefault('data',{}).setdefault('error',{})
                detail.update(attempts=number,max_attempts=2,job_id=identity,retryable=False)
                if not repair:
                    detail['message']=detail.get('message','Image analysis failed.')+(
                        ' Both permitted attempts were used; revise the image/question or contact the operator with this job ID.' if number==2
                        else ' Automatic recovery is unavailable for this failure or its deadline has expired; keep this job ID for diagnosis.')
            db.execute('UPDATE vision_attempts SET state=?,result=?,receipt=?,updated=? WHERE job_id=? AND number=?',
                       ('succeeded' if ok else 'failed',json.dumps(result),json.dumps(proof),now,identity,number))
            db.execute('UPDATE vision_dispatch SET receipt=? WHERE job_id=?',(json.dumps(proof),identity))
            db.execute('UPDATE jobs SET state=?,result=?,updated=? WHERE id=?',
                       ('succeeded' if ok else 'queued' if repair else 'failed',None if repair else json.dumps(result),now,identity))
            log.info('Vision %s attempt=%s result=%s repair_queued=%s',identity,number,code_of(result) or 'success',repair)
            return repair
    def recover(self):
        super().recover()
        with self.connect() as db:
            db.execute("UPDATE vision_attempts SET state='unknown',updated=? WHERE state='running'",(self.clock(),))
    def unknown(self):
        rows=super().unknown()
        for row in rows:
            attempts=self.attempts(row['id'],include_body=False)
            if attempts: row['attempt']=attempts[-1]['number']
        return rows
    def confirmed(self,identity,receipt,code='vision_interrupted'):
        attempts=self.attempts(identity,include_body=False)
        if not attempts: return super().confirmed(identity,receipt,code)
        a=attempts[-1]
        result=json.loads(a['result']) if a['result'] else failure(identity,code)
        return self.resolve(identity,a['number'],result,receipt)
    def not_dispatched(self,identity,code,attempt=None):
        attempts=self.attempts(identity,include_body=False)
        if not attempts: return super().not_dispatched(identity,code)
        a=attempts[-1]
        if a['number']!=attempt or a['state']!='unknown': return False
        result=json.loads(a['result']) if a['result'] else failure(identity,code)
        return self.resolve(identity,a['number'],result,{'kind':'not_dispatched'})
    def awaiting_repair(self,identity):
        attempts=self.attempts(identity,include_body=False)
        if not attempts: return False
        a=attempts[-1]
        result=json.loads(a['result']) if a['result'] else failure(identity,'vision_interrupted')
        return a['state']=='unknown' and code_of(result) in REPAIRABLE and self.clock()<a['deadline']
    def status(self,identity,owner):
        status=super().status(identity,owner)
        if status is not None:
            attempts=self.attempts(identity,include_body=False)
            status.update(attempts=len(attempts),max_attempts=2,
                recovery_pending=status['status'] in {'queued','unknown'} and bool(attempts))
        return status

async def terminate_safely(jobs,row,code):
    try: await jobs.terminate(row,code)
    except Exception:
        # A delayed receipt may race reconciliation and a newly claimed attempt.
        # Never crash the healthy worker or alter that newer attempt.
        log.warning('Vision termination still unresolved or superseded for %s',row['id'],exc_info=True)

async def run_attempt(jobs,row):
    """Called by existing two workers. UNKNOWN continues to occupy its slot."""
    identity=row['id']
    task=jobs.active[identity]=asyncio.create_task(jobs.sender(row))
    try:
        done,_=await asyncio.wait({task},timeout=max(0,row['deadline']-jobs.ledger.clock()))
        if not done: raise TimeoutError()
        result=task.result()
        proof=result.get('_termination')
        if proof is None:
            # Even an HTTP error must be reconciled if its dispatch was persisted.
            jobs.ledger.unresolved(row,result)
            await terminate_safely(jobs,row,code_of(result) or 'vision_transport_failed')
        else: jobs.ledger.resolve(identity,row['attempt'],result,proof)
    except (RecoveryRequired,TimeoutError,asyncio.CancelledError) as exc:
        code=exc.code if isinstance(exc,RecoveryRequired) else 'vision_deadline' if isinstance(exc,TimeoutError) else 'vision_interrupted'
        jobs.ledger.unresolved(row,failure(identity,code,504 if code=='vision_deadline' else 422))
        task.cancel()
        task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        if asyncio.current_task().cancelling(): raise
        await terminate_safely(jobs,row,code)
    except Exception:
        log.exception('Vision attempt failed; native reconciliation required for %s',identity)
        jobs.ledger.unresolved(row,failure(identity,'vision_transport_failed'))
        await terminate_safely(jobs,row,'vision_transport_failed')
    finally:
        jobs.release_confirmed_media()
        jobs.active.pop(identity,None); jobs.wake.set()

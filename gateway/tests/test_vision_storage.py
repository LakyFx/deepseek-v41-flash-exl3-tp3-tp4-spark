"""Capacity regression: terminal history must not permanently stop image dispatch."""
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3

import pytest

from laguna_gateway import vision_storage
from laguna_gateway.vision_jobs import Ledger, AdmissionError
from laguna_gateway.vision_recovery import RecoveryLedger, failure
from test_vision_jobs import body, api
from test_vision_recovery import dispatch


@pytest.fixture
def clock():
    return [100_000.0]


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(vision_storage, 'HOT_LIMIT', 64 * 1024)
    monkeypatch.setattr(vision_storage, 'RESULT_RESERVE', 1024)
    monkeypatch.setattr(vision_storage, 'REPAIR_RESERVE', 512)


def old_job(ledger, payload, *, state='succeeded', key='old'):
    identity, _ = ledger.accept('owner', payload, key)
    with ledger.connect() as db:
        db.execute('UPDATE jobs SET state=?,deadline=?,updated=?,result=? WHERE id=?',
                   (state, ledger.clock()-1800, ledger.clock()-1800,
                    json.dumps({'status_code': 200, 'data': {'answer': 'evidence preserved'}}), identity))
    return identity


def test_full_history_archives_losslessly_then_accepts_and_dispatches(small, tmp_path, clock):
    ledger = RecoveryLedger(tmp_path/'db', clock=lambda: clock[0], controlled=True)
    # Seed the old gateway's state, whose admission ignored physical attempts.
    old = old_job(ledger, body(text='x' * 16000))
    with ledger.connect() as db:
        db.execute('UPDATE jobs SET body=? WHERE id=?', (json.dumps(body(text='x' * 64000)), old))
    before = ledger.read(old)
    new, created = ledger.accept('owner', body(seed=42), 'new')
    assert created and ledger.claim()['id'] == new
    assert ledger.read(old) == before
    assert ledger.status(old, 'owner')['result']['data']['answer'] == 'evidence preserved'
    with ledger.connect() as db:
        assert db.execute('SELECT body FROM jobs WHERE id=?', (old,)).fetchone()[0] == ''
        assert db.execute('SELECT count(*) FROM vision_payload_archive').fetchone()[0] == 1
    assert ledger.capacity()['archived_jobs'] == 1


def test_archived_idempotency_and_owner_boundaries_are_preserved(small, tmp_path):
    ledger = Ledger(tmp_path/'db')
    payload = body(seed=43)
    old = old_job(ledger, payload)
    with ledger.connect() as db:
        assert ledger.storage.archive(db, old)
    assert ledger.accept('owner', payload, 'old') == (old, False)
    assert ledger.read(old, 'someone else') is None
    with pytest.raises(AdmissionError, match='different input'):
        ledger.accept('owner', body(seed=44), 'old')


@pytest.mark.parametrize('state', ['queued', 'running', 'unknown'])
def test_active_and_unconfirmed_work_cannot_be_archived(small, tmp_path, state):
    ledger = RecoveryLedger(tmp_path/'db', controlled=True)
    old = old_job(ledger, body(), state=state)
    with ledger.connect() as db:
        assert old not in {r['id'] for r in ledger.storage.eligible(db)}
        assert db.execute('SELECT body FROM jobs WHERE id=?', (old,)).fetchone()[0]


def test_dedup_window_and_unresolved_physical_attempt_protected(small, tmp_path):
    ledger = RecoveryLedger(tmp_path/'db', controlled=True)
    identity, _ = ledger.accept('owner', body())
    attempt = ledger.claim()
    dispatch(ledger, attempt)
    with ledger.connect() as db:
        db.execute("UPDATE jobs SET state='failed',deadline=0,updated=0 WHERE id=?", (identity,))
        db.execute("UPDATE vision_attempts SET state='unknown' WHERE job_id=?", (identity,))
        assert not ledger.storage.eligible(db)
        db.execute("UPDATE vision_attempts SET state='failed' WHERE job_id=?", (identity,))
        db.execute('UPDATE jobs SET deadline=?,updated=0 WHERE id=?', (ledger.clock(), identity))
        assert not ledger.storage.eligible(db)


def test_both_attempt_bodies_results_and_proofs_survive_archive(small, tmp_path, clock):
    ledger = RecoveryLedger(tmp_path/'db', clock=lambda: clock[0], controlled=True)
    identity, _ = ledger.accept('owner', body())
    a = ledger.claim(); dispatch(ledger, a)
    assert ledger.resolve(identity, 1, failure(identity, 'vision_output_truncated'),
                          {'kind': 'terminal_stream', 'attempt': 1})
    b = ledger.claim(); dispatch(ledger, b)
    assert not ledger.resolve(identity, 2, {'status_code': 200, 'data': {'answer': 'complete'}},
                              {'kind': 'terminal_stream', 'attempt': 2})
    before = ledger.attempts(identity)
    clock[0] = b['deadline'] + 1000
    with ledger.connect() as db:
        assert ledger.storage.archive(db, identity)
    assert ledger.attempts(identity) == before
    assert ledger.accept('owner', body(), 'different-key')[1]
    # Explicit original identity cannot replenish its attempt credits.
    assert ledger.read(identity)['state'] == 'succeeded'
    assert len(ledger.attempts(identity)) == 2


def test_non_ascii_utf8_and_all_copies_share_one_reservation(small, tmp_path):
    ledger = RecoveryLedger(tmp_path/'db', controlled=True)
    identity, _ = ledger.accept('owner', body(text='žluťoučký 🐳' * 70))
    with ledger.connect() as db:
        raw = db.execute('SELECT body FROM jobs WHERE id=?', (identity,)).fetchone()[0]
        assert ledger.storage.usage(db)['charged_bytes'] == ledger.storage.charge(len(raw.encode()))
    first = ledger.claim(); dispatch(ledger, first)
    assert ledger.resolve(identity, 1, failure(identity, 'vision_output_incomplete'),
                          {'kind': 'terminal_stream', 'attempt': 1})
    with ledger.connect() as db:
        assert ledger.storage.usage(db)['charged_bytes'] <= vision_storage.HOT_LIMIT
    assert ledger.claim()['attempt'] == 2


def test_rejects_before_accept_when_copies_would_exceed_budget(small, tmp_path):
    ledger = RecoveryLedger(tmp_path/'db', controlled=True)
    with pytest.raises(AdmissionError, match='no job accepted'):
        ledger.accept('owner', body(text='x'*24000))
    assert ledger.counts() == {}


def test_concurrent_admission_never_oversubscribes_reserved_capacity(small, tmp_path):
    ledger = RecoveryLedger(tmp_path/'db', controlled=True)
    def submit(i):
        try:
            return ledger.accept('owner', body(text='x'*4000, seed=i))[1]
        except AdmissionError:
            return False
    with ThreadPoolExecutor(8) as pool:
        accepted = sum(pool.map(submit, range(12)))
    assert 0 < accepted < 12
    with ledger.connect() as db:
        assert ledger.storage.usage(db)['charged_bytes'] <= vision_storage.HOT_LIMIT


def test_legacy_repair_also_reserves_capacity_before_copying(small, tmp_path):
    ledger = RecoveryLedger(tmp_path/'db', controlled=True)
    payload = body(text='x'*16000)
    identity, _ = ledger.accept('owner', payload, 'legacy')
    with ledger.connect() as db:
        db.execute("UPDATE jobs SET state='failed',result=? WHERE id=?",
                   (json.dumps(failure(identity, 'vision_output_incomplete')), identity))
        db.execute('INSERT INTO vision_dispatch VALUES (?,?,NULL)',
                   (identity, json.dumps({'instance':'legacy', 'request_id':'old'})))
    # Another retained recent result prevents a safe repair reservation. It
    # cannot be archived while its dedup/deadline interval is still open.
    other, _ = ledger.accept('owner', body(seed=99), 'other')
    with ledger.connect() as db:
        db.execute("UPDATE jobs SET state='succeeded',body=? WHERE id=?", ('x'*40000, other))
    with pytest.raises(AdmissionError, match='cannot reserve its recovery'):
        ledger.accept('owner', payload, 'legacy')
    assert ledger.attempts(identity) == []
    assert ledger.read(identity)['state'] == 'failed'


def test_archive_and_hot_replacement_are_one_transaction(small, tmp_path):
    ledger = Ledger(tmp_path/'db')
    old = old_job(ledger, body())
    with ledger.connect() as db:
        db.execute("CREATE TRIGGER refuse_clear BEFORE UPDATE OF body ON jobs BEGIN SELECT RAISE(ABORT,'test'); END")
    with pytest.raises(sqlite3.IntegrityError):
        with ledger.connect() as db:
            ledger.storage.archive(db, old)
    with ledger.connect() as db:
        assert not db.execute('SELECT 1 FROM vision_payload_archive').fetchone()
        assert db.execute('SELECT body FROM jobs WHERE id=?', (old,)).fetchone()[0]


def test_corrupted_archive_is_reported_and_never_replayed(small, tmp_path):
    ledger = Ledger(tmp_path/'db')
    old = old_job(ledger, body())
    with ledger.connect() as db:
        ledger.storage.archive(db, old)
        db.execute("UPDATE vision_payload_archive SET sha256='wrong'")
    with pytest.raises(ValueError, match='integrity'):
        ledger.read(old)
    assert ledger.status(old, 'owner')['status'] == 'succeeded'
    assert ledger.claim() is None


def test_health_exposes_capacity_without_stopping_text_api(api, monkeypatch):
    client, _, app = api
    monkeypatch.setattr(vision_storage, 'HOT_LIMIT', 1)
    app.state.vision_jobs.ledger.storage.cached = None
    result = client.get('/health')
    assert result.status_code == 200
    vision = result.json()['vision_jobs']
    assert not vision['ready'] and vision['blocked'] and vision['degraded']
    assert not vision['storage']['available']


def test_terminal_images_expire_but_result_and_idempotency_survive(tmp_path, clock):
    ledger = RecoveryLedger(tmp_path/'db', clock=lambda: clock[0], controlled=True)
    identity, _ = ledger.accept('owner', body(), 'delayed-client')
    a = ledger.claim(); dispatch(ledger, a)
    ledger.resolve(identity, 1, {'status_code': 200, 'data': {'answer': 'delivered later'}},
                   {'kind': 'terminal_stream', 'attempt': 1})
    clock[0] = a['deadline'] + vision_storage.PAYLOAD_TTL + 1
    ledger.storage.maintenance()
    assert ledger.status(identity, 'owner')['result']['data']['answer'] == 'delivered later'
    assert ledger.accept('owner', body(), 'delayed-client') == (identity, False)
    with pytest.raises(ValueError, match='retention expired'):
        ledger.read(identity)
    with ledger.connect() as db:
        assert db.execute('SELECT length(payload) FROM vision_payload_archive').fetchone()[0] == 0
        assert db.execute('SELECT body FROM vision_attempts').fetchone()[0] == ''


@pytest.mark.parametrize('state', ['queued', 'running', 'unknown'])
def test_retention_does_not_discard_unfinished_or_unconfirmed_work(tmp_path, clock, state):
    ledger = RecoveryLedger(tmp_path/'db', clock=lambda: clock[0], controlled=True)
    identity = old_job(ledger, body(), state=state)
    clock[0] += vision_storage.RESULT_TTL * 2
    ledger.storage.maintenance()
    assert ledger.read(identity)['body']


def test_old_archive_expires_without_rehydration_then_result_expires(tmp_path, clock, monkeypatch):
    ledger = Ledger(tmp_path/'db', clock=lambda: clock[0])
    identity = old_job(ledger, body())
    with ledger.connect() as db:
        ledger.storage.archive(db, identity)
    clock[0] += vision_storage.PAYLOAD_TTL
    monkeypatch.setattr(ledger.storage, 'hydrate', lambda *_: pytest.fail('No need to read old image'))
    ledger.storage.maintenance()
    assert ledger.status(identity, 'owner')['status'] == 'succeeded'
    with ledger.connect() as db:
        assert db.execute('SELECT length(payload) FROM vision_payload_archive').fetchone()[0] == 0
    clock[0] += vision_storage.RESULT_TTL
    ledger.storage.maintenance()
    assert ledger.status(identity, 'owner') is None
    with ledger.connect() as db:
        assert db.execute('SELECT count(*) FROM vision_payload_archive').fetchone()[0] == 0


def test_expiry_rolls_back_payload_and_tombstone_together(tmp_path, clock):
    ledger = Ledger(tmp_path/'db', clock=lambda: clock[0])
    identity = old_job(ledger, body())
    clock[0] += vision_storage.PAYLOAD_TTL
    with ledger.connect() as db:
        db.execute("CREATE TRIGGER fail_expiry BEFORE UPDATE OF body ON jobs BEGIN SELECT RAISE(ABORT,'test'); END")
    with pytest.raises(sqlite3.IntegrityError):
        ledger.storage.maintenance()
    assert ledger.storage.maintained_at is None
    assert ledger.read(identity)['body']
    with ledger.connect() as db:
        assert not db.execute('SELECT 1 FROM vision_payload_archive').fetchone()


def test_image_api_delivers_after_history_capacity_is_reclaimed(api, small):
    client, fake, app = api
    ledger = app.state.vision_jobs.ledger
    old = old_job(ledger, body(seed=71))
    with ledger.connect() as db:
        db.execute('UPDATE jobs SET body=? WHERE id=?', ('x'*64000, old))
    response = client.post('/v1/chat/completions', json=body(seed=72))
    assert response.status_code == 200, response.text
    assert response.json()['choices'][0]['message']['content'] == fake.content
    assert len(fake.calls) == 1
    assert client.post('/v1/chat/completions', json=body(seed=72)).status_code == 200
    assert len(fake.calls) == 1  # reconnect retrieves the same completed answer


def test_failed_job_with_unconfirmed_attempt_never_loses_payload(tmp_path, clock):
    ledger = RecoveryLedger(tmp_path/'db', clock=lambda: clock[0], controlled=True)
    identity, _ = ledger.accept('owner', body())
    a = ledger.claim(); dispatch(ledger, a)
    with ledger.connect() as db:
        db.execute("UPDATE jobs SET state='failed',deadline=0,updated=0 WHERE id=?", (identity,))
        db.execute("UPDATE vision_attempts SET state='unknown' WHERE job_id=?", (identity,))
    clock[0] += vision_storage.RESULT_TTL*2
    ledger.storage.maintenance()
    assert ledger.read(identity)['body']
    assert ledger.attempts(identity)[0]['body']

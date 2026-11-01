"""Prepare additive vision DB for old gateway; no service management.

Default is inspection. --apply is only for a STOPPED gateway after GO. Completed
results remain intact. Refuse running or unknown generations; they need receipts.
"""
import argparse
import json
import sqlite3
from pathlib import Path

def prepare(path,apply=False):
    db=sqlite3.connect('file:'+Path(path).resolve().as_posix()+'?mode='+('rw' if apply else 'ro'),uri=True)
    db.row_factory=sqlite3.Row
    try:
        db.execute('BEGIN IMMEDIATE' if apply else 'BEGIN')
        if db.execute("SELECT count(*) FROM jobs WHERE state IN ('running','unknown')").fetchone()[0]:
            raise ValueError('Reconcile active/unknown attempts before rollback')
        rows=db.execute('''SELECT j.id,a.result FROM jobs j JOIN vision_attempts a ON a.job_id=j.id
            WHERE j.state='queued' AND a.number=(SELECT max(number) FROM vision_attempts WHERE job_id=j.id)
            AND a.state='failed' AND a.receipt IS NOT NULL''').fetchall()
        if apply:
            for row in rows: db.execute("UPDATE jobs SET state='failed',result=? WHERE id=?",(row['result'],row['id']))
            db.commit()
        else: db.rollback()
        return {'pending_repairs_closed':len(rows),'applied':apply,'successful_results_preserved':True}
    finally: db.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('database',type=Path);p.add_argument('--apply',action='store_true')
    a=p.parse_args();print(json.dumps(prepare(a.database,a.apply)))

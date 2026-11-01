"""CPU-only, lossless old TP4 rank cache -> balanced401a-v1.

Default: plan only (footer/index reads). --write requires --verify-writes and an
explicit new output path. Never replaces an existing file, never loads CUDA.
Inputs are own rank and its column partner (rank XOR 2), both original layout.
Run after GO, offline: considerable disk IO (~250 GiB output across four ranks).
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

LAYOUT='balanced401a-v1'
LAYERS={0,1,2,3,4,5,6,7,8,15,30,31,32,33,34,35,36,37,38,39}
MAGIC=b'TFDSRK01'
SIZES={'bfloat16':2,'float16':2,'float32':4,'int16':2,'int32':4,'int64':8,'uint8':1,'float8_e4m3fn':1}
GU=re.compile(r'^layers\.(\d+)\.ffn\.(?:experts\.\d+|shared_experts)\.w[13]$')

class Cache:
    def __init__(self,path,expected_index=None):
        self.path=Path(path); self.file=self.path.open('rb')
        try:
            self.file.seek(-16,2); footer=self.file.read(16)
            if footer[8:]!=MAGIC: raise ValueError('Bad rank cache footer')
            self.end=struct.unpack('<Q',footer[:8])[0]
            length=self.file.tell()-16-self.end
            if not 0<length<64*1024*1024: raise ValueError('Invalid index length')
            self.file.seek(self.end); raw=self.file.read(length)
            self.digest=hashlib.sha256(raw).hexdigest()
            if expected_index and self.digest!=expected_index: raise ValueError('Source index changed')
            self.rows=json.loads(raw); self.index={}; at=0
            for row in self.rows:
                key,dtype,shape,offset,size=row
                if key in self.index or dtype not in SIZES or len(shape)>4 or any(type(d)!=int or d<0 for d in shape):
                    raise ValueError('Invalid or duplicate entry')
                if offset!=at or size!=math.prod(shape)*SIZES[dtype] or offset+size>self.end:
                    raise ValueError('Invalid tensor bounds')
                self.index[key]=row; at+=size
            if at!=self.end: raise ValueError('Unindexed payload')
        except BaseException:
            self.file.close(); raise
    def read(self,row):
        self.file.seek(row[3]); data=self.file.read(row[4])
        if len(data)!=row[4]: raise ValueError('Short tensor read')
        return data
    def close(self): self.file.close()
    def chunks(self,row):
        self.file.seek(row[3]); remaining=row[4]
        while remaining:
            data=self.file.read(min(remaining,8*1024*1024))
            if not data: raise ValueError('Short tensor read')
            remaining-=len(data)
            yield data

def ranges(rank,balanced):
    lo=(rank%2)*1152; mid=lo+(512 if balanced else 640)
    return (lo,mid) if rank//2==0 else (mid,lo+1152)

def changed(key,rank):
    fields=key.split('|')
    if len(fields)!=4: return None
    match=GU.fullmatch(fields[0])
    if not match: return None
    layer=int(match[1])
    if not 0<=layer<40 or fields[2]!='None' or fields[3] not in {'tr','suh','svh'}:
        raise ValueError('Unexpected expert key')
    old=ast.literal_eval(fields[1])
    if old!=ranges(rank,False): raise ValueError('Input must be original TP4 layout')
    if layer not in LAYERS: return None
    new=ranges(rank,True)
    return fields,old,new

def plan(own,partner,rank):
    jobs=[]; at=0; changes=0
    for row in own.rows:
        key,dtype,shape,_,size=row
        spec=changed(key,rank)
        peer=None
        if spec:
            fields,old,new=spec
            peerkey='|'.join([fields[0],str(ranges(rank^2,False)),fields[2],fields[3]])
            peer=partner.index[peerkey]
            if peer[1]!=dtype: raise ValueError('Partner dtype differs')
            shape=list(shape)
            if fields[3]=='tr':
                if dtype!='int16' or len(shape)!=3 or shape[1]*16!=old[1]-old[0] or peer[2][0]!=shape[0] or peer[2][2]!=shape[2]:
                    raise ValueError('Bad trellis layout')
                shape[1]=(new[1]-new[0])//16
            elif fields[3]=='svh':
                if dtype!='float16' or shape!=[old[1]-old[0]]: raise ValueError('Bad output signs')
                shape=[new[1]-new[0]]
            elif shape!=peer[2] or dtype!='float16': raise ValueError('Bad input signs')
            size=math.prod(shape)*SIZES[dtype]
            key='|'.join([fields[0],str(new),fields[2],fields[3]])
            changes+=1
        target=[key,dtype,shape,at,size]; at+=size
        jobs.append((row,peer,spec,target))
    return jobs,at,changes

def reshape(own,partner,row,peer,spec,rank):
    data=own.read(row)
    if not spec: return data
    fields,old,new=spec
    other=partner.read(peer)
    if fields[3]=='suh':
        if data!=other: raise ValueError('Partner input rotation signs differ')
        return data
    # Rank caches store raw K-tile-major trellises, NOT GPU strip format.
    # Concatenate within each K row before slicing whole 128-column blocks.
    rows=row[2][0] if fields[3]=='tr' else 1
    unit=row[2][2]*2 if fields[3]=='tr' else 2
    divisor=16 if fields[3]=='tr' else 1
    a,b=(data,other) if rank//2==0 else (other,data)
    aw=len(a)//rows; bw=len(b)//rows
    base=(rank%2)*1152
    start=(new[0]-base)//divisor*unit; width=(new[1]-new[0])//divisor*unit
    return b''.join((a[i*aw:(i+1)*aw]+b[i*bw:(i+1)*bw])[start:start+width] for i in range(rows))

def write(own,partner,rank,output,expected_keys=None):
    output=Path(output)
    if output.resolve() in {own.path.resolve(),partner.path.resolve()} or output.exists():
        raise ValueError('Output must be a new file; rollback inputs are immutable')
    jobs,total,changes=plan(own,partner,rank)
    if expected_keys is not None and [j[3][0] for j in jobs]!=expected_keys:
        raise ValueError('Native loader plan differs from export')
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix(output.suffix+'.partial')
    hashes=[]; digest=hashlib.sha256()
    with partial.open('xb') as f:
        for row,peer,spec,target in jobs:
            tensor_hash=hashlib.sha256(); written=0
            chunks=(reshape(own,partner,row,peer,spec,rank),) if spec else own.chunks(row)
            for data in chunks:
                f.write(data); tensor_hash.update(data); digest.update(data); written+=len(data)
            if written!=target[4]: raise ValueError('Wrong output tensor length')
            hashes.append(tensor_hash.hexdigest())
        raw=json.dumps([j[3] for j in jobs],separators=(',',':')).encode()
        footer=struct.pack('<Q',total)+MAGIC
        f.write(raw); f.write(footer); f.flush(); os.fsync(f.fileno())
        digest.update(raw); digest.update(footer)
    check=Cache(partial)
    try:
        if check.rows!=[j[3] for j in jobs]: raise ValueError('Written index differs')
        for row,h in zip(check.rows,hashes):
            tensor_hash=hashlib.sha256()
            for data in check.chunks(row): tensor_hash.update(data)
            if tensor_hash.hexdigest()!=h: raise ValueError('Written bytes differ')
    finally: check.close()
    # Hard-link publication refuses concurrent overwrite on Windows and POSIX.
    os.link(partial,output)
    receipt={'layout':LAYOUT,'rank':rank,'world':4,'output':str(output),'payload_bytes':total,
             'sha256':digest.hexdigest(),'entries':len(jobs),'changed_entries':changes,
             'index_sha256':hashlib.sha256(raw).hexdigest(),'file_bytes':output.stat().st_size,
             'input_index_sha256':own.digest,'partner_index_sha256':partner.digest,
             'all_written_tensors_readback_verified':True,'native_key_plan_verified':expected_keys is not None,
             'preserved_partial_hardlink':str(partial),'gpu_qualified':False}
    with output.with_suffix('.receipt.json').open('x',encoding='utf-8') as f: json.dump(receipt,f,indent=2)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True); p.add_argument('--partner',required=True)
    p.add_argument('--rank',required=True,type=int,choices=range(4))
    p.add_argument('--source-index-sha256',required=True); p.add_argument('--partner-index-sha256',required=True)
    p.add_argument('--output'); p.add_argument('--write',action='store_true'); p.add_argument('--verify-writes',action='store_true')
    p.add_argument('--native-keys',help='Output of candidate tf-dsv41-plan with balanced layout enabled')
    a=p.parse_args()
    own=Cache(a.source,a.source_index_sha256)
    try:
        peer=Cache(a.partner,a.partner_index_sha256)
        try:
            jobs,total,count=plan(own,peer,a.rank)
            if count!=20*385*2*3: raise ValueError('Incomplete 1017 expert set')
            if not a.write:
                print(json.dumps(dict(layout=LAYOUT,rank=a.rank,payload_bytes=total,changed_entries=count,write=False))); return
            if not a.output or not a.verify_writes or not a.native_keys:
                p.error('--write requires new --output, --verify-writes and --native-keys')
            if not Path(a.output).name.startswith(f'rank{a.rank}of4-{LAYOUT}-'):
                p.error('Output filename must include rank and versioned layout')
            keys=Path(a.native_keys).read_text().splitlines()
            print(json.dumps(write(own,peer,a.rank,a.output,keys)))
        finally: peer.close()
    finally: own.close()

if __name__=='__main__': main()

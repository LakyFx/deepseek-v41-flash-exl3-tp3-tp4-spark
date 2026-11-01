"""Independent full-byte check against the actual production TP4 reference shards.

Production model directories contain metadata, not the original converted
safetensors. Compare all output bytes to overlapping reference column segments,
without calling the exporter's plan/reshape algorithm or generating new weights.
"""
import argparse,ast,hashlib,json,re,time
from pathlib import Path
from .repack_weights import Cache

def compare(out,own,peer):
    # Match by semantic tensor identity and native-plan output ranges.
    grouped={}
    for source in (own,peer):
        for row in source.rows:
            fields=row[0].split('|')
            if len(fields)==4:
                grouped.setdefault((fields[0],fields[2],fields[3]),[]).append((source,row,ast.literal_eval(fields[1])))
    total=changed=0
    for target in out.rows:
        direct=own.index.get(target[0])
        if direct is not None:
            if target[1:3]!=direct[1:3]:raise ValueError('Unchanged shape differs')
            for a,b in zip(out.chunks(target),own.chunks(direct),strict=True):
                if a!=b:raise ValueError('Unchanged bytes differ: '+target[0])
            total+=target[4];continue
        fields=target[0].split('|')
        if len(fields)!=4 or fields[3] not in ('tr','suh','svh'):raise ValueError('Unexpected changed tensor')
        sources=grouped[(fields[0],fields[2],fields[3])]
        if len(sources)!=2:raise ValueError('Require exactly two column references')
        sources=sorted(sources,key=lambda x:x[2][0]);(lo,hi)=ast.literal_eval(fields[1])
        if sources[0][2][1]!=sources[1][2][0] or not sources[0][2][0]<=lo<hi<=sources[1][2][1]:raise ValueError('Invalid coverage')
        raw=out.read(target)
        if fields[3]=='suh':
            if not all(target[1:3]==row[1:3] and raw==source.read(row) for source,row,_ in sources):raise ValueError('Rotation signs differ')
        else:
            rows=target[2][0] if fields[3]=='tr' else 1
            divisor=16 if fields[3]=='tr' else 1
            unit=target[2][2]*2 if fields[3]=='tr' else 2
            if fields[3]=='tr' and (len(target[2])!=3 or target[2][1]*16!=hi-lo):raise ValueError('Output column shape differs')
            if fields[3]=='svh' and target[2]!=[hi-lo]:raise ValueError('Output sign shape differs')
            stride=(hi-lo)//divisor*unit
            if len(raw)!=rows*stride:raise ValueError('Output layout differs')
            covered=0
            for source,row,(a,b) in sources:
                left=max(a,lo);right=min(b,hi)
                if right<=left:continue
                if target[1]!=row[1]:raise ValueError('Dtype differs')
                if fields[3]=='tr' and (target[2][0]!=row[2][0] or target[2][2]!=row[2][2]):raise ValueError('K/trellis layout differs')
                original=source.read(row);source_stride=(b-a)//divisor*unit
                count=(right-left)//divisor*unit;src=(left-a)//divisor*unit;dst=(left-lo)//divisor*unit
                for k in range(rows):
                    if raw[k*stride+dst:k*stride+dst+count]!=original[k*source_stride+src:k*source_stride+src+count]:raise ValueError('Column bytes differ: '+target[0])
                covered+=right-left
            if covered!=hi-lo:raise ValueError('Incomplete column comparison')
        total+=len(raw);changed+=1
    if total!=out.end:raise ValueError('Uncompared output bytes')
    return dict(entries=len(out.rows),changed_entries=changed,compared_bytes=total)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--partner',type=Path,required=True)
    a=p.parse_args(); out=Cache(a.output);own=Cache(a.source);peer=Cache(a.partner)
    try:print(json.dumps(dict(passed=True,**compare(out,own,peer))))
    finally:out.close();own.close();peer.close()

if __name__=='__main__':main()

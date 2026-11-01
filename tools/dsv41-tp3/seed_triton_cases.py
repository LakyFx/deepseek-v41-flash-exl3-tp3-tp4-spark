"""Seed an offline TP3 kernel set from upstream recorded launch cases.

This is NOT a coverage certificate. Additional variants for future request shapes
must be collected from native wrappers and checked before loading the model.
"""
import argparse
import importlib
import json
from pathlib import Path
import struct
from compile_triton import TYPES


MARKOV={'_bias','_score','_finish','_pick'}


def module_for(name):
    if name=='_hc_mix_pf': return 'tp3_triton_kernels'
    if name in MARKOV: return 'tensorfold.families.deepseek_v41.cuda.markov'
    if name=='_gemm': return 'tensorfold.cuda.exl3.prefill'
    return 'tensorfold.families.deepseek_v41.cuda.kernels'


def seed(name,c):
    module=module_for(name);fn=getattr(importlib.import_module(module),name)
    annotated={fn.arg_names[i] for i in fn.constexprs}
    consts={k:v for k,v in c['consts'].items() if k in annotated}
    for k,v in list(consts.items()):
        if isinstance(v,dict): consts[k]=struct.unpack('<f',struct.pack('<I',int(v['fp32_bits'],16)))[0]
    if consts.get('WORLD')==2: consts['WORLD']=3
    # Indexer heads (HD=128) are replicated and remain 32. Main attention HD=512 is sharded.
    if consts.get('H')==32 and consts.get('HD')==512:
        consts['H']=24;consts['HB']=8
    params={k:{'type':TYPES[v[0]],'div16':v[3]} for k,v in c['tensors'].items()}
    params.update({k:{'type':'fp32' if isinstance(v,dict) else 'i32'} for k,v in c['scalars'].items() if k not in annotated})
    for k in c['consts']:
        # Historical fixtures predate removal of this constexpr from the current indexer.
        if name=='_index_score_rows' and k=='TREE' and k not in fn.arg_names: continue
        if k not in annotated: params[k]={'type':'i32'}
    if set(fn.arg_names)!=set(params)|set(consts): raise ValueError('Fixture/source mismatch: '+name)
    return dict(module=module,function=name,consts=consts,params=params,
        options={'num_warps':c['num_warps'],'num_stages':1 if name=='_sparse_attn_part' else 4 if name=='_gemm' else 3,
                 'launch_pdl':bool(consts.get('PDL',False))})


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--out',required=True)
    p.add_argument('--functions',nargs='+');a=p.parse_args()
    fixtures=json.loads(Path(a.fixtures).read_bytes());cases={}
    names=a.functions or list(fixtures)
    for name in names:
        for c in fixtures[name]:
            case=seed(name,c);cases[json.dumps(case,sort_keys=True)]=case
    with Path(a.out).open('x') as f: json.dump(list(cases.values()),f,indent=2)
    print('SEED_CASES_ONLY',len(cases),'functions',len(names))


if __name__=='__main__':main()

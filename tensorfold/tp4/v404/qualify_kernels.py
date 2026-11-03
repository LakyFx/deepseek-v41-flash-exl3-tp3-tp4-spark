"""Offline ELF/ABI checks by default. GPU matrix/graph checks REQUIRE --run-gpu.

Run after future GO, on a quiescent test GPU before loading a candidate model.
This never controls services/API, loads checkpoint weights or changes selection.
"""
import argparse,ctypes as C,hashlib,json,re,struct,time
from pathlib import Path

class GLayer(C.Structure):
    _fields_=[(x,C.c_uint64) for x in ['xh','T','svh','bias','y','Z','counters']]+[(x,C.c_int64) for x in ['ldx','ldy','stride_k','stride_nb']]+[(x,C.c_int32) for x in ['y_dtype','K','N','SK','first']]+[(x,C.c_uint64) for x in ['rsuh','rxh']]+[('ldr',C.c_int64),('roff',C.c_int32)]+[(x,C.c_uint64) for x in ['rcos','rsin','rpos']]+[(x,C.c_int32) for x in ['rhd','rrd']]
class GArgs(C.Structure):
    _fields_=[('layers',GLayer*8),('n',C.c_int32),('m',C.c_int32),('discard',C.c_int32)]
class LinearArgs(C.Structure):
    _fields_=[(x,C.c_uint64) for x in ['xh','t']]+[(x,C.c_int64) for x in ['stride_k','stride_nb']]+[(x,C.c_uint64) for x in ['svh','bias','y','z','counters']]+[(x,C.c_int32) for x in ['dtype','n','sk','m','k','discard']]
class WcArgs(C.Structure):
    _fields_=LinearArgs._fields_[:-1]+[('warps',C.c_int32),('discard',C.c_int32)]

def symbols(blob):
    assert blob[:6]==b'\x7fELF\x02\x01','expected little-endian ELF64 CUDA cubin'
    off=struct.unpack_from('<Q',blob,40)[0];size,count=struct.unpack_from('<HH',blob,58)
    sections=[struct.unpack_from('<IIQQQQIIQQ',blob,off+i*size) for i in range(count)]
    result=[]
    for s in sections:
        if s[1]!=2:continue
        st=sections[s[6]];names=blob[st[4]:st[4]+st[5]]
        for pos in range(s[4],s[4]+s[5],s[9]):
            i=struct.unpack_from('<I',blob,pos)[0]
            if i:result.append(names[i:names.find(b'\0',i)].decode())
    return result

def check(root):
    assert C.sizeof(GLayer)==176 and C.sizeof(GArgs)==1424
    assert C.sizeof(LinearArgs)==96 and LinearArgs.discard.offset==92
    assert C.sizeof(WcArgs)==104 and WcArgs.warps.offset==92 and WcArgs.discard.offset==96
    spec=json.loads((root/'BUILD.json').read_bytes());result={}
    for item in spec['artifacts']:
        p=root/item['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
        names=set(symbols(p.read_bytes()));result[p.name]=len(names)
        if p.name.startswith('midm'):
            for h in [6,8,10,12]:
                for w in [4,8]:
                    for q in [1,2]:assert f'tf_exl3_mpg_{h}_{w}_{q}' in names
            for h in [8,12]:
                for mt in [2,4]:
                    for ks in [2,4]:
                        for fold in [0,1]:assert f'tf_exl3_wc_{h}_{mt}_{ks}_{fold}' in names
        else:
            prefix='tf404_group' if p.name.startswith('grouped') else 'tf404_unpack'
            for h in [2,4,6,8,10,12,14,16,3,5,7]:
                for cb in ([0,1,2] if h%2==0 else [2]):
                    if prefix.endswith('group'):
                        for w in [4,8]:assert f'{prefix}_{h}_{cb}_{w}' in names
                    else:assert f'{prefix}_{h}_{cb}' in names
    return result

def plan(k,n):
    sk=1;wk=8 if n//128>=64 else 4
    if n//128<64:
        while n//128*sk<192 and sk<64:sk*=2
    while (wk>4 or sk>1) and ((k//16)%(sk*wk) or (k//16)//(sk*wk)<8):
        if wk>4:wk=4
        elif sk>1:sk//=2
        else:break
    return sk,wk

def choice(h,k,n,m,sk,wk,shared_limit):
    if m<17 or m>128:return None
    per=k//16//sk//wk;nb=n//128;fold=sk>1 and nb>=32
    if h in [8,12] and per%2==0 and (fold or h!=8 or m<=32 or m>40):
        ks=4 if per%4==0 and not(h==12 and m>96) else 2
        mt=2 if m<=32 or 64<m<=96 or fold and m>96 and nb>=64 else 4
        shared=max(4*(ks*8*4*h*4+ks*16*mt*32),16*mt*128*4)
        if shared<=shared_limit:return f'tf_exl3_wc_{h}_{mt}_{ks}_{int(fold)}',16*mt,1 if fold else sk,256,shared
    p=2 if m>48 else 1
    return f'tf_exl3_mpg_{h}_{wk}_{p}',16*p,sk,wk*32,wk*min(m,8)*128*4

def gpu(args):
    import torch
    assert args.baseline_grouped and args.baseline_prefill,'both baseline cubins are required for comparison to actual 4.0.3'
    torch.cuda.set_device(args.device);torch.manual_seed(404)
    driver=C.CDLL('libcuda.so.1')
    def call(name,*a):
        rc=getattr(driver,name)(*a)
        if rc:raise RuntimeError(f'{name} CUDA error {rc}; stop qualification, do not publish this candidate')
    driver.cuModuleLoadData.argtypes=[C.POINTER(C.c_void_p),C.c_void_p]
    driver.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p]
    driver.cuFuncSetAttribute.argtypes=[C.c_void_p,C.c_int,C.c_int]
    driver.cuLaunchKernel.argtypes=[C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());modules={};functions={}
    shared_limit=C.c_int();call('cuDeviceGetAttribute',C.byref(shared_limit),97,args.device)
    shared_limit=max(0,shared_limit.value-1024)
    def load(path):
        b=C.create_string_buffer(path.read_bytes());m=C.c_void_p();call('cuModuleLoadData',C.byref(m),b);modules[str(path)]=m;return m
    blobs={p.stem:load(p) for p in args.kernels.glob('*.cubin')}
    old=load(args.baseline_grouped);old_names=symbols(args.baseline_grouped.read_bytes())
    old_pf=load(args.baseline_prefill);old_pf_names=symbols(args.baseline_prefill.read_bytes())
    def fn(module,name,shared=0):
        key=(module.value,name)
        if key not in functions:
            f=C.c_void_p();call('cuModuleGetFunction',C.byref(f),module,name.encode());functions[key]=f
            if shared>49152:call('cuFuncSetAttribute',f,8,shared)
        return functions[key]
    def launch(f,arg,gx,gy,bx,shared):
        values=arg if isinstance(arg,list) else [arg]
        packed=(C.c_void_p*len(values))(*(C.cast(C.byref(v),C.c_void_p) for v in values))
        call('cuLaunchKernel',f,gx,gy,1,bx,1,1,shared,C.c_void_p(stream.cuda_stream),packed,None)
    rows=[]
    shapes=[(5120,1024),(5120,4096),(1280,16384)]
    counts=[17,49,97] if args.quick else [1,16,17,32,33,40,48,49,64,96,97,128]
    dtypes=[torch.bfloat16] if args.quick else [torch.float16,torch.bfloat16,torch.float32]
    for k,n in shapes:
      for h in [6,8,10,12]:
       for m in counts:
        for dtype in dtypes:
         sk,wk=plan(k,n);dtype_id={torch.float16:0,torch.bfloat16:1,torch.float32:2}[dtype]
         with torch.cuda.stream(stream):
            xh=torch.randn((m,k),device='cuda',dtype=torch.float16)*.01
            trellis=torch.randint(-(1<<31),(1<<31)-1,(n//128*(k//16)*8*4*h,),device='cuda',dtype=torch.int32)
            svh=torch.ones(n,device='cuda',dtype=torch.float16)
            out=torch.empty((m,n),device='cuda',dtype=dtype)
            z=torch.empty((sk,m,n),device='cuda',dtype=torch.float32)
            counters=torch.zeros(8*n//128,device='cuda',dtype=torch.int32)
         stride_k=8*4*h;stride_nb=k//16*stride_k
         a=LinearArgs(xh.data_ptr(),trellis.data_ptr(),stride_k,stride_nb,svh.data_ptr(),0,out.data_ptr(),z.data_ptr(),counters.data_ptr(),dtype_id,n,sk,m,k,1)
         g=GArgs();g.n=1;g.m=m;g.discard=1;l=g.layers[0]
         for name,value in dict(xh=a.xh,T=a.t,svh=a.svh,y=a.y,Z=a.z,counters=a.counters,ldx=k,ldy=n,stride_k=stride_k,stride_nb=stride_nb,y_dtype=dtype_id,K=k,N=n,SK=sk).items():setattr(l,name,value)
         old_name=next(s for s in old_names if f'glinear_kernelILi{h}ELi2ELi{wk}E' in s)
         launch(fn(old,old_name),g,n//128*sk,1,wk*32,wk*min(m,8)*128*4)
         stream.synchronize();reference=out.clone();reference_ready=torch.cuda.current_stream();stream.wait_stream(reference_ready)
         selected=choice(h,k,n,m,sk,wk,shared_limit)
         for variant in ['grouped-l10','grouped-l11','midm-l10','midm-l11']:
            if variant.startswith('midm'):
                if selected is None:continue
                name,rg,gy,bx,shared=selected;gx=n//128*((m+rg-1)//rg)
                arg=a
                if '_wc_' in name:
                    arg=WcArgs()
                    for key,_ in LinearArgs._fields_:setattr(arg,key,getattr(a,key))
                    arg.warps=wk
            else:name=f'tf404_group_{h}_2_{wk}';arg=g;gx=n//128*sk;gy=1;bx=wk*32;shared=wk*min(m,8)*128*4
            kernel=fn(blobs[variant],name,shared)
            launch(kernel,arg,gx,gy,bx,shared);stream.synchronize()
            assert torch.equal(out,reference),(variant,k,n,h,m,str(dtype),'different output bits')
            assert not counters.count_nonzero().item(),'split-K counter not reset'
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):launch(kernel,arg,gx,gy,bx,shared)
            with torch.cuda.stream(stream):
                for _ in range(3):graph.replay()
            stream.synchronize();assert torch.equal(out,reference),(variant,'graph replay differs')
            rows.append(dict(variant=variant,k=k,n=n,halves=h,rows=m,dtype=str(dtype),same_bits=True,graph_replays=3,counters_zero=True))
         if m==counts[0] and dtype==dtypes[0]:
            with torch.cuda.stream(stream):unpacked=torch.empty((k,n),device='cuda',dtype=torch.float16)
            unpack_args=[C.c_uint64(trellis.data_ptr()),C.c_uint64(unpacked.data_ptr()),C.c_int(n),C.c_int64(stride_k),C.c_int64(stride_nb)]
            original=next(s for s in old_pf_names if f'unpack_kernelILi{h}ELi2E' in s)
            launch(fn(old_pf,original),unpack_args,n//16,k//16,32,0);stream.synchronize()
            unpack_ref=unpacked.clone();stream.wait_stream(torch.cuda.current_stream())
            for variant in ['unpack-l10','unpack-l11']:
                launch(fn(blobs[variant],f'tf404_unpack_{h}_2'),unpack_args,n//16,k//16,32,0);stream.synchronize()
                assert torch.equal(unpacked,unpack_ref),(variant,k,n,h,'unpacked weight bits differ')
                rows.append(dict(variant=variant,k=k,n=n,halves=h,same_bits=True))
    result={'gpu':torch.cuda.get_device_name(args.device),'seed':404,'records':rows,'passed':True,
            'baseline_sha256':hashlib.sha256(args.baseline_grouped.read_bytes()).hexdigest(),
            'note':'isolated linear correctness/graph test, not whole-model or TP4 performance qualification'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'records':len(rows),'output':str(args.output)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernels',type=Path,required=True);parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--baseline-grouped',type=Path);parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--baseline-prefill',type=Path)
    parser.add_argument('--quick',action='store_true');parser.add_argument('--output',type=Path,default=Path('GPU-KERNEL-QUALIFICATION.json'))
    args=parser.parse_args();print(json.dumps({'offline_elf_abi':check(args.kernels)}))
    if args.run_gpu:gpu(args)

"""CPU-only SM121 build; GPU oracles run separately with model stopped."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    a.output.mkdir(exist_ok=False)
    nvcc=Path('/usr/local/cuda/bin/nvcc');src=Path(__file__).with_name('greedy_head.cu')
    flags=['-arch=sm_121','-O3','-std=c++17'];commands=[]
    for name,extra in [('greedy_head.cubin',['--cubin']),('greedy_head_test',['-DTF_GREEDY_TEST'])]:
        cmd=[str(nvcc),*flags,*extra,str(src),'-o',str(a.output/name)]
        r=subprocess.run(cmd,capture_output=True,text=True);commands.append(dict(argv=cmd,returncode=r.returncode,stderr=r.stderr))
        assert r.returncode==0,r.stderr
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (a.output/'COMPILED.json').write_text(json.dumps(dict(status='PASS',gpu_access=False,source_sha256=sha(src),nvcc_sha256=sha(nvcc),commands=commands,files={n:sha(a.output/n) for n in ['greedy_head.cubin','greedy_head_test']}),indent=2))
if __name__=='__main__':main()

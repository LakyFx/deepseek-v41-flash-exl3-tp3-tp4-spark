// Exact greedy reduction over vocabulary shards. Full logits remain the path
// for sampled waves; no rounding, weight or target-verification changes.
#include <cuda_runtime.h>
#include <stdint.h>
struct Best { float value; unsigned id; };
__device__ __forceinline__ Best better(Best a, Best b) {
    if (b.value == b.value && (b.value > a.value ||
        (b.value == a.value && b.id < a.id))) return b;
    return a;
}
extern "C" __global__ void tf_greedy_local(const float* logits,
    unsigned long long stride, unsigned width, unsigned offset,
    unsigned valid, Best* out) {
    const float* row = logits + (unsigned long long)blockIdx.x * stride;
    Best a = {-__int_as_float(0x7f800000), 0xffffffffu};
    for (unsigned i=threadIdx.x; i<width && offset+i<valid; i+=blockDim.x)
        a=better(a, Best{row[i],offset+i});
    for(int d=16;d;d>>=1) {
        Best b={__shfl_down_sync(0xffffffffu,a.value,d),__shfl_down_sync(0xffffffffu,a.id,d)};
        a=better(a,b);
    }
    __shared__ Best shared[8];
    const unsigned lane=threadIdx.x%32, warp=threadIdx.x/32;
    if(lane==0) shared[warp]=a;
    __syncthreads();
    if(warp) return;
    a=lane<8?shared[lane]:Best{-__int_as_float(0x7f800000),0xffffffffu};
    for(int d=16;d;d>>=1) {
        Best b={__shfl_down_sync(0xffffffffu,a.value,d),__shfl_down_sync(0xffffffffu,a.id,d)};
        a=better(a,b);
    }
    if(lane==0) {
        // sampling.argmax starts with row[0], so a global leading NaN wins.
        if(offset==0 && row[0]!=row[0]) a=Best{row[0],0};
        out[blockIdx.x]=a;
    }
}
extern "C" __global__ void tf_greedy_merge(const Best* gathered,
    unsigned rows, unsigned* out) {
    if(threadIdx.x) return;
    const unsigned row=blockIdx.x;
    Best a={-__int_as_float(0x7f800000),0xffffffffu};
    for(unsigned rank=0;rank<4;++rank) {
        Best b=gathered[rank*rows+row];
        if(b.id==0 && b.value!=b.value) {out[row]=0;return;}
        a=better(a,b);
    }
    out[row]=a.id==0xffffffffu?0:a.id;
}
#ifdef TF_GREEDY_TEST
#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <limits>
#define CK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
int main() {
    unsigned cases=0;
    std::mt19937 rng(42);
    for(unsigned rows: {1u,3u,16u,48u}) for(unsigned vocab: {17u,1024u,129280u})
    for(unsigned pattern=0;pattern<10;++pattern) {
        const unsigned quarter=(vocab+3)/4;
        unsigned widths[4]={quarter,quarter,quarter,vocab-3*quarter};
        unsigned offsets[4]={0,quarter,2*quarter,3*quarter};
        if(vocab==129280) {widths[0]=32384;widths[1]=32256;widths[2]=32384;widths[3]=32256;offsets[0]=0;offsets[1]=32384;offsets[2]=64640;offsets[3]=97024;}
        const unsigned order[4]={0,2,1,3}; // TP4 g=2*p+r
        std::vector<float> host(rows*vocab);
        for(auto& v:host) v=float(int(rng()%20001)-10000)*0.125f;
        std::vector<unsigned> reference(rows),got(rows);
        for(unsigned row=0;row<rows;++row) {
            float* h=host.data()+row*vocab;
            if(pattern==1) for(unsigned i=0;i<vocab;++i) h[i]=-std::numeric_limits<float>::infinity();
            if(pattern==2) for(unsigned i=0;i<vocab;++i) h[i]=std::numeric_limits<float>::quiet_NaN();
            if(pattern==3) h[0]=std::numeric_limits<float>::quiet_NaN();
            if(pattern==4) h[vocab/2]=std::numeric_limits<float>::quiet_NaN();
            if(pattern==5) {h[1]=INFINITY;h[vocab-1]=INFINITY;}
            if(pattern==6) {h[vocab/4]=8192.f;h[vocab/2]=8192.f;}
            if(pattern==7) for(unsigned i=0;i<vocab;++i) h[i]=(i%2)?-0.f:0.f;
            if(pattern==8) {for(unsigned i=0;i<vocab;++i) h[i]=NAN;h[0]=-INFINITY;h[vocab-1]=7.f;}
            if(pattern==9) {for(unsigned i=0;i<vocab;++i) h[i]=-INFINITY;h[0]=0.f;h[vocab/2]=NAN;}
            unsigned best=0;for(unsigned i=1;i<vocab;++i)if(h[i]>h[best])best=i;reference[row]=best;
        }
        Best* devPairs;unsigned* devOut;CK(cudaMalloc(&devPairs,4*rows*sizeof(Best)));CK(cudaMalloc(&devOut,rows*4));
        for(unsigned rank=0;rank<4;++rank) {
            unsigned q=order[rank], width=widths[q], off=offsets[q],stride=width+7;
            std::vector<float> local(rows*stride,1234567.f);
            for(unsigned row=0;row<rows;++row) for(unsigned i=0;i<width;++i)local[row*stride+i]=host[row*vocab+off+i];
            float* dev;CK(cudaMalloc(&dev,local.size()*4));CK(cudaMemcpy(dev,local.data(),local.size()*4,cudaMemcpyHostToDevice));
            tf_greedy_local<<<rows,256>>>(dev,stride,width,off,vocab,devPairs+rank*rows);
            CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaFree(dev));
        }
        tf_greedy_merge<<<rows,32>>>(devPairs,rows,devOut);CK(cudaGetLastError());
        CK(cudaMemcpy(got.data(),devOut,rows*4,cudaMemcpyDeviceToHost));
        if(got!=reference){fprintf(stderr,"mismatch rows=%u vocab=%u pattern=%u\n",rows,vocab,pattern);return 3;}
        CK(cudaFree(devOut));CK(cudaFree(devPairs));++cases;
    }
    printf("{\"status\":\"PASS\",\"cases\":%u,\"nan_inf_ties_padding_rank_permutation\":true}\n",cases);
}
#endif

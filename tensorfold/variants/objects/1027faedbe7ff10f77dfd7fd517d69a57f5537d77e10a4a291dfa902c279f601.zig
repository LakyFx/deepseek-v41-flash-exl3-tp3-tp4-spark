//! Prompt MoE pipeline: computation on the original stream, ordered column
//! exchanges on one side stream. Two disjoint views of existing scratch.
const std=@import("std");
const cuda=@import("cuda");
const prompt=@import("prompt.zig");
const Two=@import("prompt2d.zig").Two;
const weights=@import("weights.zig");
const X=@import("exl3_experts.zig");
const X2=@import("exl3_experts2d.zig");
const plan=@import("pipe_plan.zig");
const checks=@import("candidate_checks.zig");

pub const State=struct {
    exchange:cuda.Stream,
    ready:[2]cuda.Event,
    landed:[2]cuda.Event,
    banks:[2]X.Scratch,
    metadata:plan.Metadata,
    original_xg:u64,
    full:[2]u64,
    calls:u64=0,
    failed:bool=false,
    checked:[96]struct {key:u64,tail:bool}=undefined,
    nchecked:usize=0,

    pub fn init(e:*const prompt.Engine,ch:*const prompt.Chunk) !State {
        const t=e.two orelse return error.PipeNeedsTP4;
        if(e.comm.world!=4) return error.PipeNeedsTP4;
        const ex=weights.scratchExperts(e.w.layers);
        const sc=ch.xs;
        const meta=try plan.Metadata.of(sc.rows,sc.slots,ex.dims,ex.count);
        const size=X.Scratch.sizes(plan.tile_rows,sc.slots,ex.dims,ex.width,ex.count);
        const capacity=X.Scratch.sizes(sc.rows,sc.slots,ex.dims,ex.width,ex.count);
        var banks:[2]X.Scratch=undefined;
        for(&banks,0..) |*bank,i| {
            bank.*=sc;bank.rows=plan.tile_rows;
            inline for(.{"xg","xu","xd","z"},0..) |field,j| {
                if(2*size[j]>capacity[j]) return error.PipeScratchCapacity;
                @field(bank,field)=@field(sc,field)+i*size[j];
            }
            // no_y stays the original immutable zero scalar.
            inline for(.{"ids","count","counts","members","work_gu","work_d"},0..) |field,j|
                @field(bank,field)=sc.xg+meta.offsets[i][j];
        }
        var stream=try cuda.Stream.init(e.d,true);
        errdefer stream.deinit();
        var ready:[2]cuda.Event=undefined;
        var landed:[2]cuda.Event=undefined;
        var made_ready:usize=0;var made_landed:usize=0;
        errdefer for(ready[0..made_ready]) |*event| event.deinit();
        errdefer for(landed[0..made_landed]) |*event| event.deinit();
        for(&ready) |*event| {event.*=try cuda.Event.init(e.d,false);made_ready+=1;}
        for(&landed) |*event| {event.*=try cuda.Event.init(e.d,false);made_landed+=1;}
        const bytes=plan.tile_rows*sc.slots*(t.gu_capacity[0]+t.gu_capacity[1])*2;
        var state:State=.{.exchange=stream,.ready=ready,.landed=landed,.banks=banks,
            .metadata=meta,.original_xg=sc.xg,.full=.{t.xd_full,t.xd_full+bytes}};
        try state.resetMetadata(e);
        std.debug.print("prefill_pipe: rank={d} tile_rows={d} minimum_rows={d} gpu_scratch_added=0 decode=unchanged\n",.{e.comm.rank,plan.tile_rows,plan.threshold});
        return state;
    }

    fn resetMetadata(st:*State,e:*const prompt.Engine) !void {
        try prompt.fill(e,st.original_xg+st.metadata.first,0,st.metadata.end-st.metadata.first);
        for(st.banks) |b| try prompt.fill(e,b.members,255,st.metadata.sizes[3]);
    }

    pub fn close(st:*State) void {
        st.exchange.synchronize() catch {};
        for(&st.ready) |*event| event.deinit();
        for(&st.landed) |*event| event.deinit();
        st.exchange.deinit();
    }

    fn baseline(t:*const Two,e:*const prompt.Engine,ch:*prompt.Chunk,ex:weights.Experts) !void {
        try X2.gateUp(e.ex,e.s,ex,ch.xs,ch.x,e.c.hidden,ch.pick,ch.n,e.c.swiglu_limit);
        try t.catRank(e,ch.xs.xd,t.xd_full,ch.n*ch.xs.slots,t.expertWidths(ex),2);
        try X2.down(e.ex,e.s,ex,ch.xs,t.xd_full,ch.pick,ch.wts,ch.pm,ch.n);
    }

    fn finish(st:*State,e:*const prompt.Engine,ch:*prompt.Chunk,ex:weights.Experts,tile:plan.Tile,reference:?[]const u8) !void {
        const sc=st.banks[tile.bank];
        try e.s.wait(st.landed[tile.bank]);
        try X2.down(e.ex,e.s,ex,sc,st.full[tile.bank],ch.pick+tile.first*sc.slots*4,
            ch.wts+tile.first*sc.slots*4,ch.pm+tile.first*ex.down_n*4,tile.rows);
        if(checks.enabled) if(reference) |expected| {
            // Separate qualification only. Validate every slot before combine
            // can hide errors; the normal binary contains no download/sync.
            const bytes=tile.rows*sc.slots*ex.down_n*4;
            const actual=try std.heap.page_allocator.alloc(u8,bytes);
            defer std.heap.page_allocator.free(actual);
            try e.s.synchronize();
            try e.d.check(e.d.api.cuMemcpyDtoH_v2(actual.ptr,sc.z,bytes),"pipe slot check");
            const start=tile.first*sc.slots*ex.down_n*4;
            if(!std.mem.eql(u8,actual,expected[start..][0..bytes])) return error.PipeSlotMismatch;
        };
    }

    pub fn run(st:*State,t:*const Two,e:*const prompt.Engine,ch:*prompt.Chunk,ex:weights.Experts) !void {
        if(st.failed) return error.PipeFailedClosed;
        if(ch.n<plan.threshold or ch.n>ch.xs.rows) return error.PipeShape;
        errdefer {st.failed=true;e.s.synchronize() catch {};st.exchange.synchronize() catch {};}
        const allocator=std.heap.page_allocator;
        var reference_slots:?[]u8=null;var reference_out:?[]u8=null;
        defer if(reference_slots) |b| allocator.free(b);
        defer if(reference_out) |b| allocator.free(b);
        const tail=ch.n%plan.tile_rows!=0;
        var qualify=checks.enabled;
        if(checks.enabled) for(st.checked[0..st.nchecked]) |item| {
            if(item.key==ex.gate_ptr and item.tail==tail) qualify=false;
        };
        if(checks.enabled and qualify) {
            // Same input, same weights, original grouping and execution.
            reference_slots=try allocator.alloc(u8,ch.n*ch.xs.slots*ex.down_n*4);
            reference_out=try allocator.alloc(u8,ch.n*ex.down_n*4);
            try baseline(t,e,ch,ex);
            try e.s.synchronize();
            try e.d.check(e.d.api.cuMemcpyDtoH_v2(reference_slots.?.ptr,ch.xs.z,reference_slots.?.len),"pipe reference slots");
            try e.d.check(e.d.api.cuMemcpyDtoH_v2(reference_out.?.ptr,ch.pm,reference_out.?.len),"pipe reference out");
            // Original full-chunk xg overlaps the reserved metadata region.
            try st.resetMetadata(e);
        }
        var transfer=e.*;
        transfer.s=st.exchange;transfer.t.stream=st.exchange;
        // Keep every NCCL operation in host order. No decode ring state is
        // touched on this prompt-only path; both streams join before return.
        var transport=t.*;transport.rings=null;transport.neighbor_gather=null;
        var it:plan.Iter=.{.n=ch.n};var previous:?plan.Tile=null;
        while(it.next()) |tile| {
            const sc=st.banks[tile.bank];
            try X2.gateUp(e.ex,e.s,ex,sc,ch.x+tile.first*e.c.hidden*2,e.c.hidden,
                ch.pick+tile.first*sc.slots*4,tile.rows,e.c.swiglu_limit);
            try st.ready[tile.bank].record(e.s);
            try st.exchange.wait(st.ready[tile.bank]);
            try transport.catRank(&transfer,sc.xd,st.full[tile.bank],tile.rows*sc.slots,t.expertWidths(ex),2);
            try st.landed[tile.bank].record(st.exchange);
            // G_i is independent of C_(i-1); D_(i-1) can overlap C_i.
            if(previous) |prev| try st.finish(e,ch,ex,prev,reference_slots);
            previous=tile;
        }
        try st.finish(e,ch,ex,previous orelse return error.PipeShape,reference_slots);
        if(checks.enabled and qualify) {
            const actual=try allocator.alloc(u8,reference_out.?.len);defer allocator.free(actual);
            try e.s.synchronize();
            try e.d.check(e.d.api.cuMemcpyDtoH_v2(actual.ptr,ch.pm,actual.len),"pipe combined check");
            if(!std.mem.eql(u8,actual,reference_out.?)) return error.PipeCombineMismatch;
            if(st.nchecked>=st.checked.len) return error.PipeCheckCapacity;
            st.checked[st.nchecked]=.{.key=ex.gate_ptr,.tail=tail};st.nchecked+=1;
            std.debug.print("prefill_pipe_check: rank={d} rows={d} slots=PASS combine=PASS tail={}\n",.{e.comm.rank,ch.n,tail});
        }
        st.calls+=1;
        if(st.calls==1 or st.calls%1024==0)
            std.debug.print("prefill_pipe: rank={d} calls={d} rows={d} tiles={d} side_stream=active\n",.{e.comm.rank,st.calls,ch.n,it.index});
    }
};

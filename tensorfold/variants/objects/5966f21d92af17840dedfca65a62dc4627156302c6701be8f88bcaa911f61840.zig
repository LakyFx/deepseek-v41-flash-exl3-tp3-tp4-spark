//! TP3-only serving envelope. CPU-only; run before model allocations/collectives.
const std=@import("std");
pub const Input=struct {
    world:u32, streams:usize, round_rows:usize, pool:usize, prompt_context:usize,
    direct_rdma:bool, neighbor:bool, balanced:bool, midm_b:bool, policy_control:bool,
};
pub fn check(o:Input) !void {
    if(o.world!=3) return error.TP129RequiresThreeRanks;
    if(o.streams==0 or o.streams>12 or o.round_rows==0 or o.round_rows>48)
        return error.TP129ServingCapacity;
    if(o.pool==0 or o.pool>7_000_000 or o.prompt_context==0 or o.prompt_context>401408)
        return error.TP129MemoryEnvelope;
    if(!o.direct_rdma or o.neighbor or o.balanced) return error.TP129RequiresOriginalTransportAndLayout;
    if(!o.midm_b or o.policy_control) return error.TP129RequiresQualifiedBKernelPolicy;
}
test "TP129 refuses inherited TP4 capacity, workspace, transport and kernel modes" {
    const good:Input=.{.world=3,.streams=12,.round_rows=48,.pool=7_000_000,
        .prompt_context=401408,.direct_rdma=true,.neighbor=false,.balanced=false,.midm_b=true,.policy_control=false};
    try check(good);
    var x=good;x.world=4;try std.testing.expectError(error.TP129RequiresThreeRanks,check(x));
    x=good;x.streams=16;try std.testing.expectError(error.TP129ServingCapacity,check(x));
    x=good;x.pool=9_000_000;try std.testing.expectError(error.TP129MemoryEnvelope,check(x));
    x=good;x.prompt_context=0;try std.testing.expectError(error.TP129MemoryEnvelope,check(x));
    x=good;x.neighbor=true;try std.testing.expectError(error.TP129RequiresOriginalTransportAndLayout,check(x));
    x=good;x.balanced=true;try std.testing.expectError(error.TP129RequiresOriginalTransportAndLayout,check(x));
    x=good;x.direct_rdma=false;try std.testing.expectError(error.TP129RequiresOriginalTransportAndLayout,check(x));
    x=good;x.midm_b=false;try std.testing.expectError(error.TP129RequiresQualifiedBKernelPolicy,check(x));
    x=good;x.policy_control=true;try std.testing.expectError(error.TP129RequiresQualifiedBKernelPolicy,check(x));
    for(1..13) |c| {x=good;x.streams=c;try check(x);}
}

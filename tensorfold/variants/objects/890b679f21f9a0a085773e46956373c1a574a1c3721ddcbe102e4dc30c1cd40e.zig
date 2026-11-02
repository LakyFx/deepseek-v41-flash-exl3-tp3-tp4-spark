//! Actual CUDA wrapper and native graph owners over fake driver calls. No GPU.
const std = @import("std");
const cuda = @import("cuda");
const round = @import("round.zig");
const draft = @import("draft.zig");
const control = @import("smart_graph_controller.zig");
const p = @import("smart_graph_policy.zig");
const fake = @import("smart_graph_fake_link.zig");

threadlocal var destroy_result: cuda.abi.Result = 0;
threadlocal var destroys: usize = 0;
threadlocal var error_lookups: usize = 0;
fn destroy(_: cuda.abi.GraphExec) callconv(.c) cuda.abi.Result {
    destroys += 1;
    return destroy_result;
}
fn errorText(_: cuda.abi.Result, out: *?[*:0]const u8) callconv(.c) cuda.abi.Result {
    error_lookups += 1;
    out.* = "injected CUDA destroy failure";
    return 0;
}
fn driver() cuda.Driver {
    var d: cuda.Driver = undefined;
    d.api.cuGraphExecDestroy = destroy;
    d.api.cuGetErrorName = errorText;
    d.api.cuGetErrorString = errorText;
    return d;
}
fn executable(d: *const cuda.Driver) cuda.graph.Exec {
    return .{ .d = d, .handle = @ptrFromInt(0x1000) };
}

test "SCGC regression: checked CUDA destroy preserves ownership on failure and succeeds once" {
    const d = driver();
    var exec = executable(&d);
    const handle = exec.handle;
    destroys = 0; error_lookups = 0;
    for ([_]cuda.abi.Result{ 1, 2, 999 }) |code| {
        destroy_result = code;
        if (code == 2) try std.testing.expectError(error.OutOfDeviceMemory, exec.deinitChecked())
        else try std.testing.expectError(error.CudaFailed, exec.deinitChecked());
        try std.testing.expectEqual(handle, exec.handle);
    }
    try std.testing.expectEqual(@as(usize, 6), error_lookups);
    destroy_result = 0;
    try exec.deinitChecked();
    try std.testing.expectEqual(@as(usize, 4), destroys);
}

test "SCGC regression: round and draft owners keep the failed executable and its storage until checked success" {
    const gpa = std.testing.allocator;
    const d = driver();
    var rg: round.Graphs = .{ .gpa = gpa };
    defer rg.deinit();
    var dg: draft.PassGraphs = .{ .gpa = gpa };
    defer dg.deinit();
    const rk: round.Graphs.Key = .{ .rows = 6, .bucket = 4096, .stretch = 0 };
    const dk: draft.PassGraphs.Key = .{ .streams = 2, .steps = 2, .split = true };
    try rg.map.put(gpa, rk, .{ .exec = executable(&d), .start = undefined, .end = undefined, .storage = try gpa.dupe(u64, &.{11}) });
    try dg.map.put(gpa, dk, .{ .exec = executable(&d), .storage = try gpa.dupe(u64, &.{22}) });
    destroy_result = 1;
    try std.testing.expectError(error.CudaFailed, rg.retire(.{ .rows = 6, .bucket = 4096 }));
    try std.testing.expectError(error.CudaFailed, dg.retire(dk));
    try std.testing.expectEqualSlices(u64, &.{11}, rg.map.get(rk).?.storage.?);
    try std.testing.expectEqualSlices(u64, &.{22}, dg.map.get(dk).?.storage.?);
    destroy_result = 0;
    try rg.retire(.{ .rows = 6, .bucket = 4096 });
    try dg.retire(dk);
    try std.testing.expectEqual(@as(u32, 0), rg.map.count());
    try std.testing.expectEqual(@as(u32, 0), dg.map.count());
}

const Runner = struct {
    rank: u32,
    peers: [3]?fake.Link,
    fault_kind: p.Kind,
    rg: round.Graphs = .{ .gpa = std.heap.page_allocator },
    dg: draft.PassGraphs = .{ .gpa = std.heap.page_allocator },
    err: ?anyerror = null,
    failed: bool = false,
    generation: u64 = 0,
    rejected: usize = 0,
    retained_failed_owner: bool = false,
    fn mask(ptr: *anyopaque, key: p.Group) u64 {
        const s: *Runner = @ptrCast(@alignCast(ptr));
        return if (key.kind == .round) s.rg.groupMask(.{ .rows = key.rows, .bucket = key.bucket })
            else @intFromBool(s.dg.map.contains(.{ .streams = key.rows, .steps = key.steps, .split = key.split }));
    }
    fn retire(ptr: *anyopaque, key: p.Group) !void {
        const s: *Runner = @ptrCast(@alignCast(ptr));
        destroy_result = if (s.rank == 2 and key.kind == s.fault_kind) 1 else 0;
        errdefer { s.rejected += 1; s.retained_failed_owner = mask(ptr, key) != 0; }
        if (key.kind == .round) try s.rg.retire(.{ .rows = key.rows, .bucket = key.bucket })
        else try s.dg.retire(.{ .streams = key.rows, .steps = key.steps, .split = key.split });
        fake.available += 16 * p.mib;
    }
    fn run(s: *Runner) void { s.work() catch |err| { s.err = err; }; }
    fn work(s: *Runner) !void {
        const d = driver();
        destroy_result = 0;
        defer { destroy_result = 0; s.rg.deinit(); s.dg.deinit(); }
        fake.available = 8 * 1024 * p.mib; fake.rss = 1024 * p.mib;
        var rt = try control.Runtime.open(std.heap.page_allocator, .{ .enabled = true, .poll_primitives = 16 },
            0x403, s.rank, 4, s.peers[0..if (s.rank == 0) 3 else 1], s, retire, mask);
        defer rt.close();
        defer { s.failed = rt.failed; s.generation = rt.ledger.?.generation; }
        const seeds = [_]p.Group{
            .{ .kind = .round, .rows = 6, .bucket = 1 }, .{ .kind = .round, .rows = 6, .bucket = 2 },
            .{ .kind = .draft, .rows = 1, .steps = 1 }, .{ .kind = .draft, .rows = 2, .steps = 1 },
        };
        for (seeds) |key| {
            const leaf: p.Leaf = .{ .group = key };
            try rt.boundary(true, key);
            if (!try rt.prepare(leaf, true)) return error.UnexpectedRefusal;
            fake.available -= 16 * p.mib; fake.rss += 16 * p.mib;
            try rt.captured(leaf, true);
            if (key.kind == .round) try s.rg.map.put(s.rg.gpa, .{ .rows = key.rows, .bucket = key.bucket, .stretch = 0 },
                .{ .exec = executable(&d), .start = undefined, .end = undefined })
            else try s.dg.map.put(s.dg.gpa, .{ .streams = key.rows, .steps = key.steps, .split = key.split }, .{ .exec = executable(&d) });
        }
        rt.ledger.?.clock = 127;
        fake.available = 1200 * p.mib;
        try rt.boundary(true, .{});
    }
};
test "SCGC regression: actual native retirement propagates a CUDA error and aborts all four ranks" {
    for ([_]p.Kind{ .round, .draft }) |kind| {
        var channels: [3]fake.Channel = @splat(.{});
        var ranks: [4]Runner = undefined;
        ranks[0] = .{ .rank = 0, .fault_kind = kind, .peers = .{
            .{ .channel = &channels[0], .head = true }, .{ .channel = &channels[1], .head = true }, .{ .channel = &channels[2], .head = true } } };
        for (1..4) |i| ranks[i] = .{ .rank = @intCast(i), .fault_kind = kind, .peers = .{ .{ .channel = &channels[i-1], .head = false }, null, null } };
        var threads: [4]std.Thread = undefined;
        for (&threads, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, Runner.run, .{&ranks[i]});
        for (threads) |t| t.join();
        for (ranks) |s| try std.testing.expect(s.err != null and s.failed);
        try std.testing.expectEqual(error.CudaFailed, ranks[2].err.?);
        try std.testing.expectEqual(@as(u64, 0), ranks[2].generation);
        try std.testing.expectEqual(@as(usize, 1), ranks[2].rejected);
        try std.testing.expect(ranks[2].retained_failed_owner);
    }
}

test "SCGC regression: native qualification dirties split and head samples but clean replay resumes" {
    var owner: u8 = 0;
    const Stub = struct {
        fn mask(_: *anyopaque, _: p.Group) u64 { return 0; }
        fn retire(_: *anyopaque, _: p.Group) !void { return error.UnexpectedRetirement; }
    };
    fake.available = 8 * 1024 * p.mib; fake.rss = 0;
    var rt = try control.Runtime.open(std.testing.allocator, .{ .enabled = true }, 0x403, 0, 1, &.{}, &owner, Stub.retire, Stub.mask);
    defer rt.close();
    var graphs: draft.PassGraphs = .{ .gpa = std.testing.allocator, .control = &rt };
    var dr: draft.Drafter = undefined;
    dr.graphs = &graphs; dr.n = 4;
    dr.split_checked = @splat(@splat(false)); dr.head_checked = @splat(@splat(false));
    for ([_]bool{ true, false }) |split| {
        try rt.boundary(true, .{}); // the preceding warm verify is clean
        try std.testing.expect(rt.timing_clean);
        try std.testing.expectEqual(if (split) draft.Drafter.Qualification.split else .head, dr.qualification(2, 2, split));
        try std.testing.expect(!rt.timing_clean);
    }
    dr.split_checked[2][2] = true; dr.head_checked[2][2] = true;
    try rt.boundary(true, .{});
    try std.testing.expectEqual(draft.Drafter.Qualification.none, dr.qualification(2, 2, true));
    try std.testing.expect(rt.timing_clean);
    // A normal draft may not clear a dirty verify/capture sample.
    rt.timing_clean = false;
    _ = dr.qualification(2, 2, true);
    try std.testing.expect(!rt.timing_clean);
    // Full-width split qualification still dirties timing without a head oracle.
    rt.timing_clean = true;
    try std.testing.expectEqual(draft.Drafter.Qualification.split, dr.qualification(1, 4, true));
    try std.testing.expect(!rt.timing_clean);
    // Disabled/no-graph execution selects the identical reference branch safely.
    dr.graphs = null;
    try std.testing.expectEqual(draft.Drafter.Qualification.head, dr.qualification(1, 2, false));
}

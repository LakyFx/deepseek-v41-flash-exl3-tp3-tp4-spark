//! Host half of exact per-rank top-k compaction. No sampling rule changes.
const std = @import("std");
const sampling = @import("sampling.zig");
pub const candidates = 64;
pub const packet = candidates + 1;
pub const Pair = extern struct { value: f32, id: u32 };
pub const Snapshot = struct { data: []const Pair, rows: usize, world: usize, vocab: usize };

pub fn supported(s: ?sampling.Sampling) bool {
    const sm = s orelse return true;
    if (!std.math.isFinite(sm.temperature)) return false;
    if (sm.temperature <= 0) return true;
    return sm.top_k > 0 and sm.top_k <= candidates - sampling.margin and
        std.math.isFinite(sm.top_p) and std.math.isFinite(sm.min_p) and
        sm.top_p >= 0 and sm.top_p <= 1 and sm.min_p >= 0 and sm.min_p <= 1;
}

pub fn valid(s: Snapshot) bool {
    return s.world > 0 and s.world <= 4 and s.rows > 0 and s.rows <= 48 and
        s.vocab > 0 and s.data.len == s.world * s.rows * packet;
}

/// Invalid source values and all-masked rows use the original full-logit path,
/// preserving its error/edge-case behavior. Padding alone never causes fallback.
pub fn needsFallback(s: Snapshot) bool {
    if (!valid(s)) return true;
    for (0..s.rows) |row| {
        var finite = false;
        for (0..s.world) |rank| {
            const base = (rank * s.rows + row) * packet;
            if (s.data[base + candidates].id != 0) return true;
            const top = s.data[base];
            finite = finite or (top.id < s.vocab and std.math.isFinite(top.value));
        }
        if (!finite) return true;
    }
    return false;
}

fn before(a: Pair, b: Pair) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

/// Merge four sorted lists without rescanning the vocabulary. Each list
/// contains its rank's exact first 64 candidates, including global tie order.
pub fn merge(s: Snapshot, row: usize, values: *[candidates]f32, ids: *[candidates]i64) !usize {
    if (!valid(s) or row >= s.rows) return error.BadCompactShape;
    var at: [4]usize = @splat(0);
    const count: usize = @min(candidates, s.vocab);
    for (0..count) |i| {
        var winner: ?usize = null;
        var best: Pair = .{ .value = -std.math.inf(f32), .id = std.math.maxInt(u32) };
        for (0..s.world) |rank| {
            if (at[rank] >= candidates) continue;
            const p = s.data[(rank * s.rows + row) * packet + at[rank]];
            if (p.id >= s.vocab) continue;
            if (winner == null or before(p, best)) { winner = rank; best = p; }
        }
        const rank = winner orelse return error.BadCompactCandidates;
        at[rank] += 1;
        values[i] = best.value;
        ids[i] = best.id;
    }
    return count;
}

pub fn draw(s: Snapshot, row: usize, position: u64, sm: ?sampling.Sampling) !u32 {
    if (!supported(sm)) return error.UnsupportedCompactSampling;
    var values: [candidates]f32 = undefined;
    var ids: [candidates]i64 = undefined;
    const count = try merge(s, row, &values, &ids);
    if (sm) |params| {
        if (params.temperature > 0)
            return @intCast(try sampling.chooseCandidates(values[0..count], ids[0..count], position, params));
    }
    return @intCast(ids[0]);
}

pub fn window(s: Snapshot, first: usize, positions: []const u64, drafts: []const u32,
    sm: ?sampling.Sampling, prefix_only: bool, out: []u32) !usize {
    if (out.len == 0 or positions.len != out.len or drafts.len + 1 != out.len or first + out.len > s.rows)
        return error.BadSamplingWindow;
    var reached: usize = 0;
    var reachable = true;
    for (out, positions, 0..) |*pick, position, row| {
        if (!reachable) { pick.* = std.math.maxInt(u32); continue; }
        pick.* = try draw(s, first + row, position, sm);
        reached += 1;
        if (prefix_only and row < drafts.len and pick.* != drafts[row]) reachable = false;
    }
    return reached;
}

/// Reserve the upper 16 existing graph-piece slots for the compact head.
/// Body pieces remain shared; greedy/full/compact terminal graphs cannot alias.
/// The graph controller's wire format and 64-bit ownership mask stay intact.
pub fn graphPiece(stretch: u32, compact: bool) !u32 {
    if (stretch >= 16) return error.TooManyCompactGraphPieces;
    return stretch + (if (compact) @as(u32, 16) else 0);
}

/// Qualification only: compare the whole candidate set and original float
/// bits, so a lucky sampled winner cannot hide a kernel/transport mistake.
pub fn checkCandidates(s: Snapshot, full: []const f32, stride: usize) !void {
    if(!valid(s) or stride < s.vocab or full.len != s.rows*stride) return error.BadCompactShape;
    for(0..s.rows) |row| {
        var want_values: [candidates]f32=undefined;var want_ids: [candidates]i64=undefined;
        var got_values: [candidates]f32=undefined;var got_ids: [candidates]i64=undefined;
        const n=sampling.topCandidates(full[row*stride..][0..s.vocab],candidates,&want_values,&want_ids);
        if(n!=try merge(s,row,&got_values,&got_ids)) return error.CompactCandidateMismatch;
        if(!std.mem.eql(i64,want_ids[0..n],got_ids[0..n]) or
            !std.mem.eql(u8,std.mem.sliceAsBytes(want_values[0..n]),std.mem.sliceAsBytes(got_values[0..n])))
            return error.CompactCandidateMismatch;
    }
}

fn fixture(gpa: std.mem.Allocator, logits: []const f32, rows: usize, vocab: usize, widths: [4]usize) ![]Pair {
    const data = try gpa.alloc(Pair, rows * 4 * packet);
    @memset(data, .{ .value = -std.math.inf(f32), .id = std.math.maxInt(u32) });
    var offset: usize = 0;
    for (widths, 0..) |width, rank| {
        for (0..rows) |row| {
            var values: [candidates]f32 = undefined;
            var ids: [candidates]i64 = undefined;
            const n = sampling.topCandidates(logits[row * vocab + offset ..][0..width], candidates, &values, &ids);
            const start = (rank * rows + row) * packet;
            for (0..n) |i| data[start + i] = .{ .value = values[i], .id = @intCast(ids[i] + @as(i64, @intCast(offset))) };
            data[start + candidates] = .{ .value = 0, .id = 0 };
        }
        offset += width;
    }
    return data;
}

test "SAMP exact compact draw matches old sampler across seeds ties modes and uneven shards" {
    const gpa = std.testing.allocator;
    const vocab = 129;
    var row: [vocab]f32 = undefined;
    for (&row, 0..) |*v, i| v.* = if (i % 19 == 0) -std.math.inf(f32) else @as(f32, @floatFromInt(i % 17)) / 8 - 2;
    row[33] = -0.0;
    row[34] = 0.0;
    const data = try fixture(gpa, &row, 1, vocab, .{ 33, 32, 33, 31 });
    defer gpa.free(data);
    const snap: Snapshot = .{ .data = data, .rows = 1, .world = 4, .vocab = vocab };
    try std.testing.expect(!needsFallback(snap));
    try checkCandidates(snap,&row,vocab);
    for ([_]usize{ 1, 20, 56 }) |k| for ([_]f64{ 0, 0.6, 1.0 }) |temperature| {
        for ([_]f64{ 0.1, 0.95, 1 }) |top_p| for (0..16) |seed| {
            const sm: sampling.Sampling = .{ .seed = seed, .temperature = temperature,
                .top_k = k, .top_p = top_p, .min_p = 0.02 };
            const expected = try sampling.sampleRow(&row, 401000 + seed, sm);
            try std.testing.expectEqual(@as(u32, @intCast(expected)), try draw(snap, 0, 401000 + seed, sm));
        };
    };
}

test "SAMP preserves unreachable sentinels and whole-window fallback contract" {
    const gpa = std.testing.allocator;
    var logits: [3 * 129]f32 = undefined;
    for (&logits, 0..) |*v, i| v.* = @floatFromInt(i % 31);
    const data = try fixture(gpa, &logits, 3, 129, .{ 33, 32, 33, 31 });
    defer gpa.free(data);
    const snap: Snapshot = .{ .data = data, .rows = 3, .world = 4, .vocab = 129 };
    const sm: sampling.Sampling = .{ .seed = 42, .temperature = 0.6 };
    const positions = [_]u64{ 19, 20, 21 };
    const drafts = [_]u32{ 999, 999 };
    var got: [3]u32 = undefined;
    var expected: [3]u32 = undefined;
    var workspace: sampling.Workspace = .{};
    defer workspace.deinit(gpa);
    const reached = try sampling.sampleWindow(&workspace, gpa, &logits, 129, &positions, &drafts, sm, true, &expected);
    try std.testing.expectEqual(reached, try window(snap, 0, &positions, &drafts, sm, true, &got));
    try std.testing.expectEqualSlices(u32, &expected, &got);
    try checkCandidates(snap,&logits,129);
    data[0].value+=1;
    try std.testing.expectError(error.CompactCandidateMismatch,checkCandidates(snap,&logits,129));
    data[candidates].id = 1;
    try std.testing.expect(needsFallback(snap));
    try std.testing.expect(!supported(.{ .seed = 42, .temperature = 0.6, .top_k = 0 }));
    try std.testing.expect(!supported(.{ .seed = 42, .temperature = 0.6, .top_k = 57 }));
}

test "SAMP compact graph ownership cannot collide with full or greedy head" {
    for (0..16) |n| {
        const normal = try graphPiece(@intCast(n), false);
        const compact = try graphPiece(@intCast(n), true);
        try std.testing.expect(normal < 16 and compact >= 16 and compact < 32);
        try std.testing.expect(2 * normal != 2 * compact and 2 * normal + 1 != 2 * compact);
    }
    try std.testing.expectError(error.TooManyCompactGraphPieces, graphPiece(16, true));
}

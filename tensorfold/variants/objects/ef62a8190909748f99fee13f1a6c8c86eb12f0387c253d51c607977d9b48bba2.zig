//! Bounded host policy for the existing Markov bias cache. It sees the real
//! predecessor IDs already downloaded by the drafter; no extra GPU readback.
//! All ranks see the same gathered IDs and use the same deterministic policy.
const std = @import("std");

pub const capacity = 256;
pub const max_vocab = 131072;
pub const max_changes = 16;
pub const Change = struct { slot: u16, old: u32, new: u32 };
pub const Plan = struct {
    changes: [max_changes]Change = undefined,
    len: usize = 0,
};

pub const State = struct {
    counts: [max_vocab]u32 = @splat(0),
    slots: [max_vocab]i32 = @splat(-1),
    ids: [capacity]i64 = undefined,
    vocab: usize,
    interval: u32 = 2048,
    observed: u64 = 0,
    hits: u64 = 0,
    since_plan: u32 = 0,
    since_decay: u32 = 0,
    refreshes: u64 = 0,
    replacements: u64 = 0,

    pub fn init(vocab: usize, initial_ids: []const i64, interval: u32) !State {
        if (vocab > max_vocab or vocab < capacity or initial_ids.len != capacity or interval < 16) return error.BadMarkovPolicy;
        var s: State = .{ .vocab = vocab, .interval = interval };
        for (initial_ids, 0..) |id, slot| {
            if (id < 0 or id >= vocab or s.slots[@intCast(id)] != -1) return error.BadMarkovInitialTokens;
            s.ids[slot] = id;
            s.slots[@intCast(id)] = @intCast(slot);
        }
        return s;
    }

    pub fn observe(s: *State, token: i64) void {
        if (token < 0 or token >= s.vocab) return;
        const id: usize = @intCast(token);
        s.counts[id] +|= 1;
        s.observed +|= 1;
        s.hits +|= @intFromBool(s.slots[id] >= 0);
        s.since_plan +|= 1;
        s.since_decay +|= 1;
    }

    fn better(s: *const State, a: u32, b: u32) bool {
        return s.counts[a] > s.counts[b] or (s.counts[a] == s.counts[b] and a < b);
    }

    /// At most sixteen rows change, with a 25%+4-count admission margin.
    /// Planning does not modify the installed map. Commit only after the GPU
    /// has finished computing and installing the new rows and slot table.
    pub fn plan(s: *State) Plan {
        var p: Plan = .{};
        const interval = if (s.refreshes < 8) s.interval else @max(s.interval, 8192);
        if (s.since_plan < interval) return p;
        s.since_plan = 0;
        var best: [max_changes]u32 = undefined;
        var n: usize = 0;
        for (s.counts[0..s.vocab], 0..) |count, id| {
            if (s.slots[id] >= 0 or count < 8) continue;
            const token: u32 = @intCast(id);
            var at: usize = 0;
            while (at < n and !s.better(token, best[at])) : (at += 1) {}
            if (at == max_changes) continue;
            var j = @min(n, max_changes - 1);
            while (j > at) : (j -= 1) best[j] = best[j - 1];
            best[at] = token;
            n = @min(n + 1, max_changes);
        }
        var selected: [capacity]bool = @splat(false);
        for (best[0..n]) |token| {
            var victim: ?usize = null;
            for (s.ids, 0..) |old, slot| {
                if (selected[slot]) continue;
                if (victim == null or s.counts[@intCast(old)] < s.counts[@intCast(s.ids[victim.?])]) victim = slot;
            }
            const slot = victim orelse break;
            const old: u32 = @intCast(s.ids[slot]);
            const resident = s.counts[old];
            if (@as(u64, s.counts[token]) * 4 <= @as(u64, resident) * 5 + 16) break;
            selected[slot] = true;
            p.changes[p.len] = .{ .slot = @intCast(slot), .old = old, .new = token };
            p.len += 1;
        }
        if (s.since_decay >= 32768) {
            for (s.counts[0..s.vocab]) |*count| count.* >>= 1;
            s.since_decay = 0;
        }
        return p;
    }

    pub fn commit(s: *State, p: Plan) !void {
        if (p.len == 0 or p.len > max_changes) return error.BadMarkovUpdate;
        // Validate the complete transaction before changing any map entry.
        for (p.changes[0..p.len], 0..) |c, i| {
            if (c.slot >= capacity or c.old >= s.vocab or c.new >= s.vocab or
                s.ids[c.slot] != c.old or s.slots[c.old] != c.slot or s.slots[c.new] != -1) return error.BadMarkovUpdate;
            for (p.changes[0..i]) |prior| if (prior.slot == c.slot or prior.new == c.new) return error.BadMarkovUpdate;
        }
        for (p.changes[0..p.len]) |c| {
            s.slots[c.old] = -1;
            s.slots[c.new] = c.slot;
            s.ids[c.slot] = c.new;
        }
        s.refreshes += 1;
        s.replacements += p.len;
    }

    pub fn digest(s: *const State) u64 {
        var h: u64 = 14695981039346656037;
        for (s.ids) |id| {
            const value: u64 = @intCast(id);
            inline for (0..8) |i| h = (h ^ ((value >> (8 * i)) & 255)) *% 1099511628211;
        }
        return h;
    }
};

fn initial() [capacity]i64 {
    var ids: [capacity]i64 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i);
    return ids;
}

test "MK real predecessor hits and bounded deterministic admission" {
    var s = try State.init(1024, &initial(), 64);
    for (0..32) |_| {
        s.observe(400);
        s.observe(401);
        s.observe(12);
    }
    s.observe(-1);
    s.observe(1024);
    try std.testing.expectEqual(@as(u64, 96), s.observed);
    try std.testing.expectEqual(@as(u64, 32), s.hits);
    const before = s.digest();
    const p = s.plan();
    try std.testing.expectEqual(@as(usize, 2), p.len);
    try std.testing.expectEqual(@as(u32, 400), p.changes[0].new);
    try std.testing.expectEqual(before, s.digest());
    try s.commit(p);
    try std.testing.expect(s.digest() != before);
    try std.testing.expect(s.slots[400] >= 0 and s.slots[401] >= 0 and s.slots[12] == 12);
    s.observe(400);
    try std.testing.expectEqual(@as(u64, 33), s.hits);
}

test "MK limits replacements, preserves bijection and rejects malformed transaction atomically" {
    var s = try State.init(1024, &initial(), 16);
    for (256..356) |id| for (0..8) |_| s.observe(@intCast(id));
    const p = s.plan();
    try std.testing.expectEqual(max_changes, p.len);
    var bad = p;
    bad.changes[1].slot = bad.changes[0].slot;
    const before = s.digest();
    try std.testing.expectError(error.BadMarkovUpdate, s.commit(bad));
    try std.testing.expectEqual(before, s.digest());
    try s.commit(p);
    var cached: usize = 0;
    for (s.slots[0..s.vocab], 0..) |slot, token| if (slot >= 0) {
        cached += 1;
        try std.testing.expectEqual(@as(i64, @intCast(token)), s.ids[@intCast(slot)]);
    };
    try std.testing.expectEqual(capacity, cached);
}

test "MK decay adapts to new workload while hysteresis prevents near-tie churn" {
    var s = try State.init(1024, &initial(), 16);
    for (0..256) |id| s.counts[id] = 100;
    for (0..120) |_| s.observe(500);
    try std.testing.expectEqual(@as(usize, 0), s.plan().len);
    for (0..32768) |_| s.observe(600);
    const p = s.plan();
    try std.testing.expect(p.len > 0 and p.changes[0].new == 600);
    try std.testing.expectEqual(@as(u32, 50), s.counts[1]);
    try s.commit(p);
    try std.testing.expectEqual(@as(usize, 0), s.plan().len);
}

test "MK ranks replay identical history into identical cache generations" {
    var a = try State.init(1024, &initial(), 64);
    var b = try State.init(1024, &initial(), 64);
    for (0..16000) |i| {
        const token: i64 = @intCast(300 + i % 90);
        a.observe(token);
        b.observe(token);
        if (i % 5 == 0) {
            const pa = a.plan();
            const pb = b.plan();
            try std.testing.expectEqual(pa.len, pb.len);
            if (pa.len > 0) {
                try a.commit(pa);
                try b.commit(pb);
            }
        }
    }
    try std.testing.expect(a.refreshes >= 5);
    try std.testing.expectEqual(a.digest(), b.digest());
    try std.testing.expectEqual(a.hits, b.hits);
}

//! Host-only scheduling state shared between the drafter and the round that consumes its output.
const std = @import("std");

pub const Receipt = struct { sequence: u64, members: usize, ms: f64, shape:u4=0 };
pub const Work = struct {
    requested: u32 = 0,
    body: u32 = 0,
    scheduled_body: u32 = 0,
    projected: u32 = 0,
    exposed: u32 = 0,
};
pub const Totals = struct {
    requested: u64 = 0,
    body: u64 = 0,
    scheduled_body: u64 = 0,
    projected: u64 = 0,
    exposed: u64 = 0,
    dense_equivalent: u64 = 0,
    compact_batches: u64 = 0,
    head_available: u64 = 0, local_available: u64 = 0, shared_available: u64 = 0,
    head_selected: u64 = 0, local_selected: u64 = 0, shared_selected: u64 = 0,
    head_granted: u64 = 0, local_granted: u64 = 0, shared_granted: u64 = 0,
    head_delivered: u64 = 0, local_delivered: u64 = 0, shared_delivered: u64 = 0,
    head_verified: u64 = 0,
    head_kept: u64 = 0,
    local_verified: u64 = 0,
    local_kept: u64 = 0,
    shared_verified: u64 = 0,
    shared_kept: u64 = 0,
    copy_emitted: u64 = 0,
    copy_us: u64 = 0,
    head_emitted: u64 = 0,
    head_us: u64 = 0,
    mixed_emitted: u64 = 0,
    mixed_us: u64 = 0,
    bonus_verified: u64 = 0,
    bonus_kept: u64 = 0,
    origin_history_changes: u64 = 0,
    origin_contests: u64 = 0,
    origin_winner_changes: u64 = 0,
    origin_copy_plans: u64 = 0,
    head_wins: u64 = 0,
    copy_wins: u64 = 0,
    draft_us: u64 = 0,
    paired_rounds: u64 = 0,
    unmatched_rounds: u64 = 0,
    due: u64 = 0,
    opportunity: u64 = 0,
    funded: u64 = 0,
};
pub const ByLoad = [16]Totals;

pub fn visible(requested: usize, projected: usize, room: i64, limit: usize) usize {
    if (requested == 0) return 0;
    const available: usize = @intCast(@max(0, room -| 1));
    return @min(projected, available, limit);
}

// A shared draft pass is charged exactly once and only if the same cohort consumes it.
pub fn incoming(items: anytype) ?f64 {
    if (items.len == 0) return null;
    const first = items[0].receipt orelse return null;
    if (first.members != items.len) return null;
    for (items[1..]) |item| {
        const receipt = item.receipt orelse return null;
        if (receipt.sequence != first.sequence or receipt.members != first.members) return null;
    }
    return first.ms;
}
pub fn paired(wall: f64, outgoing: f64, previous: ?f64, verify: f64) ?f64 {
    const prior = previous orelse return null;
    if (!std.math.isFinite(wall) or !std.math.isFinite(outgoing) or !std.math.isFinite(prior) or
        !std.math.isFinite(verify) or verify < 0 or wall <= 0 or outgoing > wall or prior < 0 or outgoing < 0 or wall - outgoing + 0.000001 < verify) return null;
    return wall - outgoing + prior;
}

pub const Recent = struct {
    entries: [8]struct { tokens: usize, ms: f64, round: u64 } = undefined,
    count: usize = 0,
    cursor: usize = 0,
    pub fn add(s: *Recent, tokens: usize, ms: f64, round: u64) void {
        s.entries[s.cursor] = .{ .tokens = tokens, .ms = ms, .round = round };
        s.cursor = (s.cursor + 1) % s.entries.len;
        s.count = @min(s.count + 1, s.entries.len);
    }
    pub fn price(s: *const Recent, round: u64) ?f64 {
        var tokens: usize = 0;
        var ms: f64 = 0;
        var n: usize = 0;
        for (s.entries[0..s.count]) |x| {
            if (round -| x.round > 64) continue;
            tokens += x.tokens;
            ms += x.ms;
            n += 1;
        }
        return if (n >= 3 and ms > 0 and tokens > 0) @as(f64, @floatFromInt(tokens)) / ms else null;
    }
};

// Conditional continuation outcomes after the first mismatch are unobservable.
pub const Quality = struct {
    hits: u16 = 0,
    seen: u16 = 0,
    pub fn record(q: *Quality, offered: usize, kept: usize) void {
        if (offered == 0) return;
        if (q.seen >= 48) {
            q.hits /= 2;
            q.seen /= 2;
        }
        const hit = @min(kept, offered);
        q.hits += @intCast(hit);
        q.seen += @intCast(hit + @intFromBool(hit < offered));
    }
    pub fn rate(q: Quality, matched: i64) f64 {
        const prior: f64 = if (matched >= 24) 0.95 else if (matched >= 12) 0.90 else 0.84;
        return (8 * prior + @as(f64, @floatFromInt(q.hits))) / (8 + @as(f64, @floatFromInt(q.seen)));
    }
};

pub fn copyValue(rate: f64, n: usize, price: f64) f64 {
    var survival: f64 = 1;
    var reward: f64 = 0;
    for (0..n) |_| {
        survival *= rate;
        if (survival <= price) break;
        reward += survival - price;
    }
    return reward;
}
pub fn headValue(survivals: []const f64, price: f64) f64 {
    var reward: f64 = 0;
    for (survivals) |p| {
        if (!std.math.isFinite(p) or p <= price) break;
        reward += p - price;
    }
    return reward;
}

// Two depth groups give the CUDA graph one additional stable shape for a mixed batch.
// The first (longer) group keeps full width. Every request retains its requested prefix.
pub fn supportsCompact(model: anytype) bool {
    // On TP4 the engine executes each model shard within a two-rank pair.
    return model.comm.world == 4 and model.eng.two != null;
}

pub const Layout = struct {
    compact: bool = false,
    order: [16]usize = undefined, // logical index -> physical row
    pub fn plan(depths: []const usize, steps: usize, enabled: bool) Layout {
        var result: Layout = .{};
        for (depths, 0..) |_, i| result.order[i] = i;
        if (!enabled or depths.len < 4 or steps < 3 or steps > 5) return result;
        var longer: usize = 0;
        for (depths) |d| longer += @intFromBool(d > steps - 2);
        if (longer == 0 or longer > (depths.len + 1) / 2) return result;
        var hi: usize = 0;
        var lo = longer;
        for (depths, 0..) |d, i| {
            if (d > steps - 2) {
                result.order[i] = hi;
                hi += 1;
            } else {
                result.order[i] = lo;
                lo += 1;
            }
        }
        result.compact = true;
        return result;
    }
    pub fn width(compact: bool, streams: usize, steps: usize, row: usize) usize {
        return if (compact and row >= (streams + 1) / 2) steps - 2 else steps;
    }
    pub fn active(streams: usize, steps: usize, step: usize) usize {
        return if (step < steps - 2) streams else (streams + 1) / 2;
    }
    pub fn projected(compact: bool, streams: usize, steps: usize) usize {
        return streams * steps - if (compact) (streams / 2) * 2 else 0;
    }
};

//! 4.0.6f: compare a neighbouring whole-batch action against an actual 6d control.
//! All times are host-observed milliseconds. No CUDA operations or allocations.
const std = @import("std");
const work = @import("spec_work.zig");

pub const max_lanes = 16;
pub const max_depth = 5;
pub const Depths = [max_lanes]u8;
pub const Stage = enum(u1) { prepare, verify };
pub const Action = enum(u2) { add, drop, swap, copy_select };
pub const Arm = enum(u2) { control, trial, learned };
pub const Source = enum(u2) { head, local_copy, shared_copy };
pub fn number(n: anytype) f64 {
    return @floatFromInt(n);
}
pub fn gain(reward: f64, rate: f64, extra_ms: f64) f64 {
    return reward - rate * extra_ms;
}

/// Exact matched features; a table-index collision can discard history but cannot mix arms.
pub const Key = packed struct(u64) {
    stage: Stage,
    action: Action,
    lanes: u5,
    context: u4,
    sampled: bool,
    sources: u3,
    source: Source,
    donor_source: Source = .head,
    before: u4,
    after: u4,
    rows: u6,
    depth: u3,
    donor_depth: u3 = 0,
    band: u2, // estimated marginal gain, in output tokens
    changed: u5 = 1,
    context_after: u4 = 0,
    match_band: u2 = 0,
    loss_sources: u3 = 0,
    loss_positions: u5 = 0,
    spare: u3 = 0,

    fn pooled(k: Key) Key {
        var p = k;
        // Retain the shape transition, action, source, context regime and probability band.
        p.context = k.context / 2;
        p.context_after = k.context_after / 2;
        p.rows = k.rows / 4;
        p.lanes = if (k.lanes <= 2) k.lanes else (k.lanes - 1) / 4 + 3;
        return p;
    }
};
pub const Move = struct {
    key: Key,
    depths: Depths,
    predicted_gain: f64,
    measured_cost: bool = false,
    affected: u16 = 0,
    copy: ?CopyTerms = null,
    recipient: u8,
    donor: ?u8 = null,
};
pub const CopyTerms = struct { price: f64, limit: u8 };
pub fn bit(i: usize) u16 {
    return @as(u16, 1) << @as(u4, @intCast(i));
}
pub const Decision = struct {
    reference: f64 = 0,
    affected: u16 = 0,
    copy: ?CopyTerms = null,
    key: Key,
    arm: Arm,
    recipient: u8,
    donor: ?u8,
    sequence: u64 = 0,
    recipient_key: usize = 0,
    donor_key: usize = 0,
    donor_prior_ns: u64 = 0,
    round_started: u64 = 0,
    reserved_ms: f64 = 0,
    pub fn applied(d: Decision) bool {
        return d.arm != .control;
    }
};
pub const Observation = struct {
    emitted: usize,
    elapsed: f64,
    draft: f64,
    verify: f64,
    gap_ms: f64 = 0,
    offered: usize = 0,
    kept: usize = 0,
    donor_offered: usize = 0,
    donor_kept: usize = 0,
};
pub const Counts = struct {
    selected: u64 = 0,
    measured: u64 = 0,
    incomplete: u64 = 0,
    tokens: u64 = 0,
    elapsed_us: u64 = 0,
    draft_us: u64 = 0,
    verify_us: u64 = 0,
    gap_us: u64 = 0,
    offered: u64 = 0,
    kept: u64 = 0,
    donor_offered: u64 = 0,
    donor_kept: u64 = 0,
    budget_us: u64 = 0,
    delivery_samples: u64 = 0,
    delivery_us: u64 = 0,
    delivery_max_us: u64 = 0,
};
pub const Summary = [4][2][4][3]Counts;
pub const Diagnostics = struct {
    prepare_due: u64 = 0, verify_due: u64 = 0, prepare_idle: u64 = 0, verify_idle: u64 = 0,
    estimate_only: u64 = 0, expired_reservations: u64 = 0, demoted: u64 = 0, evicted: u64 = 0,
    incomplete_donor_groups: u64 = 0, baseline_depth_probes: u64 = 0,
};
pub fn loadGroup(n: usize) usize {
    return (n - 1) / 4;
}
fn micros(ms: f64) u64 {
    return @intFromFloat(@round(ms * 1000));
}
pub const Donor = struct {
    until: u64 = 0,
    last_delivery_ns: u64 = 0,
    marked: ?struct { decision: Decision, since: u64, prior_gap: u64 = 0, measured: bool = false } = null,
    pub fn available(s: Donor, round: u64) bool {
        return s.marked == null and round >= s.until;
    }
    pub fn mark(s: *Donor, d: Decision, now: u64, round: u64) void {
        const start = if (s.last_delivery_ns != 0) s.last_delivery_ns else now;
        s.marked = .{ .decision = d, .since = start, .prior_gap = if (d.donor_prior_ns > 0) start -| d.donor_prior_ns else 0 };
        s.until = round +| 16;
    }
};

const Amount = struct {
    n: u32 = 0,
    tokens: f64 = 0,
    ms: f64 = 0,
    rate_sum: f64 = 0,
    rate_sq: f64 = 0,
    fn add(a: *Amount, o: Observation) void {
        if (o.elapsed <= 0) return;
        if (a.n >= 32) {
            a.n /= 2;
            a.tokens *= 0.5;
            a.ms *= 0.5;
            a.rate_sum *= 0.5;
            a.rate_sq *= 0.5;
        }
        a.n += 1;
        a.tokens += number(o.emitted);
        a.ms += o.elapsed;
        const rate = number(o.emitted) / o.elapsed;
        a.rate_sum += rate;
        a.rate_sq += rate * rate;
    }
    fn meanVariance(a: Amount) f64 {
        if (a.n < 2) return 0;
        const n = number(a.n);
        return @max(0, a.rate_sq - a.rate_sum * a.rate_sum / n) / (n * (n - 1));
    }
    fn price(a: Amount) f64 {
        return if (a.ms > 0) a.tokens / a.ms else 0;
    }
};
const Evidence = struct {
    key: ?Key = null,
    arms: [2]Amount = @splat(.{}),
    last: u64 = 0,
    last_arm: [2]u64 = .{0, 0},
    isolated: bool = false,
    watch_n: u8 = 0,
    watch_tokens: f64 = 0,
    watch_expected: f64 = 0,
    missed_live: u8 = 0,
    late_live: u8 = 0,
    gap_started: [2]u16 = .{0,0},
    gap_returned: [2]u16 = .{0,0},
    missing: [2]u16 = .{ 0, 0 },
    treatment_gap: [2]Amount = @splat(.{}),
    fn matches(e: *const Evidence, k: Key, round: u64) bool {
        return e.key != null and @as(u64, @bitCast(e.key.?)) == @as(u64, @bitCast(k)) and round -| e.last <= 262144;
    }
    fn learn(e: *Evidence, k: Key, d: Decision, o: Observation, round: u64) void {
        if (!e.matches(k, round)) e.* = .{ .key = k };
        if (e.arms[@backingInt(d.arm)].n >= 32) e.missing[@backingInt(d.arm)] /= 2;
        e.arms[@backingInt(d.arm)].add(o);
        e.last_arm[@backingInt(d.arm)] = round;
        if (k.loss_sources != 0 or k.action == .drop or k.action == .swap) {
            const arm=@backingInt(d.arm);
            if (e.gap_started[arm]>=64) { e.gap_started[arm]/=2; e.gap_returned[arm]/=2; }
            e.gap_started[arm]+=1;
        }
        e.last = round;
    }
    fn verdict(e: *const Evidence, k: Key, round: u64, need: usize) ?bool {
        if (!e.matches(k, round) or @min(e.arms[0].n, e.arms[1].n) < need) return null;
        if (round -| @min(e.last_arm[0], e.last_arm[1]) > 32768) return null;
        const coverage0 = number(e.arms[0].n) / (number(e.arms[0].n) + number(e.missing[0]));
        const coverage1 = number(e.arms[1].n) / (number(e.arms[1].n) + number(e.missing[1]));
        if (@min(coverage0, coverage1) < 0.75 or @abs(coverage0 - coverage1) > 0.10) return false;
        if (k.loss_sources != 0 or k.action == .swap or k.action == .drop) {
            var coverage:[2]f64=undefined;
            for (0..2) |i| coverage[i]=number(e.gap_returned[i])/number(@max(1,e.gap_started[i]));
            if (@min(coverage[0],coverage[1])<0.75 or @abs(coverage[0]-coverage[1])>0.10) return false;
            const g = e.treatment_gap;
            if (@min(g[0].n, g[1].n) < need) return null;
            if (g[1].ms / number(g[1].n) > 1.02 * g[0].ms / number(g[0].n)) return false;
        }
        const uncertainty = @sqrt(e.arms[0].meanVariance() + e.arms[1].meanVariance());
        return e.arms[1].price() > 1.02 * e.arms[0].price() + uncertainty;
    }
};
fn index(k: Key, comptime size: usize) usize {
    var x: u64 = @bitCast(k);
    x ^= x >> 30;
    x *%= 0xbf58476d1ce4e5b9;
    x ^= x >> 27;
    x *%= 0x94d049bb133111eb;
    return @intCast((x ^ (x >> 31)) % size);
}
pub const Shape = struct {
    n: u8,
    steps: u8,
    compact: bool,
    pub fn id(s: Shape) u4 {
        return @intCast(s.steps * 2 + @intFromBool(s.compact));
    }
    pub fn rows(s: Shape) usize {
        return work.Layout.projected(s.compact, s.n, s.steps);
    }
};
pub const Physical = struct {
    shape: Shape,
    projected: Depths,
    exposed: Depths,
    pub fn plan(depths: Depths, room: Depths, n: usize) Physical {
        var ahead: [max_lanes]usize = undefined;
        var steps: u8 = 0;
        for (depths[0..n], 0..) |d, i| {
            ahead[i] = d;
            steps = @max(steps, d);
        }
        const layout = work.Layout.plan(ahead[0..n], steps, true);
        var result: Physical = .{ .shape = .{ .n = @intCast(n), .steps = steps, .compact = layout.compact }, .projected = @splat(0), .exposed = @splat(0) };
        for (0..n) |i| {
            result.projected[i] = @intCast(work.Layout.width(layout.compact, n, steps, layout.order[i]));
            result.exposed[i] = @min(room[i], result.projected[i]);
        }
        return result;
    }
};
const Cost = struct {
    n: u8 = 0,
    value: f64 = 0,
    last: u64 = 0,
    fn add(s: *Cost, ms: f64, round: u64) void {
        if (round -| s.last > 4096) s.* = .{};
        s.n = @min(16, s.n + 1);
        s.value += (ms - s.value) / number(s.n);
        s.last = round;
    }
    fn known(s: Cost, round: u64) ?f64 {
        return if (s.n >= 3 and round -| s.last <= 4096) s.value else null;
    }
};
const Position = struct {
    seen: u16 = 0,
    used: u16 = 0,
    kept: u16 = 0,
    last: u64 = 0,
    fn add(p: *Position, used: bool, kept: bool, round: u64) void {
        if (round -| p.last > 4096) p.* = .{};
        if (p.seen >= 128) {
            p.seen /= 2;
            p.used /= 2;
            p.kept /= 2;
        }
        p.seen += 1;
        p.used += @intFromBool(used);
        p.kept += @intFromBool(used and kept);
        p.last = round;
    }
};
pub const Feature = struct { source: Source, grant: u8, max: u8, room: u8, mutable: bool = true };
pub const Origin = struct { sequence: u64, feature: Feature, exposed: u8 };
pub const Context = struct { band: u4, sampled: bool, sources: u3, rows: u6 };
pub const Slot = struct { use: f64, value: f64 };
pub const Audit = struct { key: u64 = 0, arm: Arm = .control, round: u64 = 0, tokens: u32 = 0, elapsed_us: u32 = 0, draft_us: u32 = 0, verify_us: u32 = 0 };
pub const AuditRing = [16]Audit;
pub const Policy = struct {
    enabled: bool = false,
    counts: Summary = @splat(@splat(@splat(@splat(.{})))),
    audit: AuditRing = @splat(.{}),
    cursor: usize = 0,
    exact: [512]Evidence = @splat(.{}),
    pool: [256]Evidence = @splat(.{}),
    costs: [16][12][10][2]Cost = @splat(@splat(@splat(@splat(.{})))),
    // Depth and previous grant discriminate easily consumed capacity from speculative slack.
    use: [4][3][6][5]Position = @splat(@splat(@splat(@splat(.{})))),
    rng: u64 = 0xd1b54a32d192ed03,
    pending: ?Decision = null,
    current: ?Decision = null,
    outgoing: ?Decision = null,
    block: u8 = 0,
    evidence_until: [2][4]u64 = @splat(@splat(0)),
    copy_until: [4]u64 = @splat(0),
    next_stage: Stage = .verify,
    waiting: ?u64 = null,
    legacy_due: bool = true,
    legacy_since: ?u64 = null,
    copy_applied_seq: ?u64 = null,
    funded: [2]u64 = .{0, 0},
    diagnostics: Diagnostics = .{},
    pending_donors: [16]struct { key: usize = 0, prior_ns: u64 = 0 } = @splat(.{}),
    deliveries: [32]struct { key: u64 = 0, round: u64 = 0, n: u5 = 0, worst_us: u64 = 0 } = @splat(.{}),

    selections: [2]u64 = .{ 0, 0 },
    last_end_ns: u64 = 0,
    interrupted: bool = false,
    continuous: bool = true,
    gap_ms: f64 = 0,

    pub fn opportunity(p: *Policy, stage: Stage, round: u64, credit: bool) bool {
        if (!credit) {
            p.waiting = null;
            return false;
        }
        // Reserve one full 6d depth-probe period. If no eligible legacy request
        // arrives the same credit can be used by either marginal phase.
        if (p.legacy_due) {
            if (p.legacy_since == null) p.legacy_since = round;
            if (round -| p.legacy_since.? < 16) return false;
            p.legacyClaimed();
        }
        if (p.waiting == null) p.waiting = round;
        if (stage != p.next_stage and round -| p.waiting.? < 2) return false;
        if (stage == .prepare) p.diagnostics.prepare_due += 1 else p.diagnostics.verify_due += 1;
        return true;
    }
    pub fn legacyClaimed(p: *Policy) void {
        p.legacy_due = false;
        p.legacy_since = null;
    }
    pub fn canSearch(p: *const Policy, stage: Stage, n: usize, round: u64) bool {
        const until = p.evidence_until[@backingInt(stage)][loadGroup(n)];
        return until > 0 and round <= until;
    }
    pub fn startClock(p: *Policy, now: u64) void {
        p.gap_ms = if (now != 0 and p.last_end_ns != 0) number(now -| p.last_end_ns) / 1e6 else 0;
        p.continuous = !p.interrupted and p.gap_ms <= 2000;
        p.interrupted = false;
    }
    pub fn endClock(p: *Policy, now: u64, ms: f64) void {
        p.last_end_ns = if (now != 0 and std.math.isFinite(ms) and ms > 0 and ms <= 60000) now +| @as(u64, @intFromFloat(ms * 1e6)) else 0;
    }

    fn counter(p: *Policy, d: Decision) *Counts {
        return &p.counts[loadGroup(d.key.lanes)][@backingInt(d.key.stage)][@backingInt(d.key.action)][@backingInt(d.arm)];
    }
    fn coin(p: *Policy) bool {
        var x = p.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        p.rng = x;
        return (x *% 0x2545f4914f6cdd1d) >> 63 != 0;
    }
    pub fn supported(p: *const Policy, m: Move, round: u64) bool {
        if (!m.measured_cost or m.predicted_gain <= 0) return false;
        const k = m.key;
        const local = &p.exact[index(k, p.exact.len)];
        const pk = k.pooled();
        const pooled = &p.pool[index(pk, p.pool.len)];
        if (local.verdict(k, round, 6)) |verdict| return verdict;
        if (local.matches(k, round) and local.isolated) return false;
        return pooled.verdict(pk, round, 12) orelse false;
    }
    pub fn decide(p: *Policy, m: Move, rate: f64, can_probe: bool, round: u64) ?Decision {
        if (!std.math.isFinite(m.predicted_gain) or !std.math.isFinite(rate) or rate <= 0) return null;
        const k = m.key;
        // Occasional fresh controls remain possible after adoption; negative forecasts never auto-promote.
        const arm: Arm = if (can_probe)
            (if (p.coin()) .trial else .control)
        else if (p.supported(m, round)) .learned else return null;
        if (can_probe) {
            if (k.stage != p.next_stage) p.diagnostics.expired_reservations += 1;
            p.next_stage = if (k.stage == .verify) .prepare else .verify;
            p.waiting = null;
            p.funded[@backingInt(k.stage)] += 1;
            if ((p.funded[0]+p.funded[1]) % 3 == 0) { p.legacy_due = true; p.legacy_since = null; }
            if (!m.measured_cost) p.diagnostics.estimate_only += 1;
        }
        const d: Decision = .{ .reference = rate, .copy = m.copy, .affected = m.affected, .key = k, .arm = arm, .recipient = m.recipient, .donor = m.donor, .round_started = round };
        p.counter(d).selected += 1;
        p.selections[@backingInt(k.stage)] += 1;
        return d;
    }
    pub fn debit(p: *Policy, d: *Decision, target: f64, budget: anytype) void {
        if (d.arm == .learned or !std.math.isFinite(target)) return;
        const extra = target - d.reserved_ms;
        if (extra <= 0) return;
        budget.observe(extra, true);
        p.counter(d.*).budget_us += micros(extra);
        d.reserved_ms = target;
    }
    fn entry(p: *Policy, k: Key, pooled: bool, round: u64) *Evidence {
        const e = if (pooled) &p.pool[index(k, p.pool.len)] else &p.exact[index(k, p.exact.len)];
        if (!e.matches(k, round)) {
            if (e.key != null) p.diagnostics.evicted += 1;
            e.* = .{ .key = k, .last = round };
        }
        return e;
    }
    fn watch(p: *Policy, d: Decision, o: ?Observation, round: u64) void {
        const e = p.entry(d.key, false, round);
        if (o) |v| {
            e.missed_live = 0;
            e.watch_n += 1;
            e.watch_tokens += number(v.emitted);
            e.watch_expected += d.reference * v.elapsed;
            if (e.watch_n < 4) return;
            const losing = e.watch_tokens < 0.98 * e.watch_expected;
            e.watch_n = 0;
            e.watch_tokens = 0;
            e.watch_expected = 0;
            if (!losing) return;
        } else {
            e.missed_live +|= 1;
            if (e.missed_live < 4) return;
        }
        p.quarantine(d.key, e, round);
    }
    fn quarantine(p: *Policy, k: Key, e: *Evidence, round: u64) void {
        e.* = .{ .key = k, .last = round, .isolated = true };
        p.diagnostics.demoted += 1;
    }
    fn noteEvidence(p: *Policy, k: Key, e: *Evidence, round: u64, need: usize) void {
        if (e.verdict(e.key.?, round, need) orelse false) {
            e.isolated = false;
            if (k.action == .copy_select) p.copy_until[loadGroup(k.lanes)] = round +| 4096;
            p.evidence_until[@backingInt(k.stage)][loadGroup(k.lanes)] = round +| 4096;
        }
    }
    pub fn lost(p: *Policy, d: Decision) void {
        p.counter(d).incomplete += 1;
        if (d.arm == .learned) { p.watch(d, null, d.round_started); return; }
        for ([_]bool{ false, true }) |pool| {
            const k = if (pool) d.key.pooled() else d.key;
            const e = if (pool) &p.pool[index(k, p.pool.len)] else &p.exact[index(k, p.exact.len)];
            if (!e.matches(k, d.round_started)) e.* = .{ .key = k };
            e.last = @max(e.last, d.round_started);
            e.missing[@backingInt(d.arm)] +|= 1;
        }
    }
    pub fn begin(p: *Policy, sequence: ?u64, coherent: bool) void {
        if (p.current) |stale| p.lost(stale);
        p.current = null;
        if (p.pending) |d| {
            if (coherent and sequence != null and d.sequence == sequence.?) p.current = d else p.lost(d);
        }
        p.pending = null;
        if (p.outgoing) |d| p.lost(d);
        p.outgoing = null;
    }
    pub fn end(p: *Policy, o: ?Observation, round: u64) void {
        const d = p.current orelse return;
        p.current = null;
        const v = o orelse {
            p.lost(d);
            return;
        };
        if (!std.math.isFinite(v.elapsed) or !std.math.isFinite(v.draft) or !std.math.isFinite(v.verify) or
            v.elapsed <= 0 or v.draft < 0 or v.verify < 0 or v.elapsed + 0.001 < v.draft + v.verify or v.elapsed > 60000)
        {
            p.lost(d);
            return;
        }
        const c = p.counter(d);
        c.measured += 1;
        c.tokens += v.emitted;
        c.elapsed_us += micros(v.elapsed);
        c.draft_us += micros(v.draft);
        c.verify_us += micros(v.verify);
        c.gap_us += micros(v.gap_ms);
        c.offered += v.offered;
        c.kept += v.kept;
        c.donor_offered += v.donor_offered;
        c.donor_kept += v.donor_kept;
        p.audit[p.cursor] = .{
            .key = @bitCast(d.key),
            .arm = d.arm,
            .round = round,
            .tokens = @intCast(v.emitted),
            .elapsed_us = @intCast(micros(v.elapsed)),
            .draft_us = @intCast(micros(v.draft)),
            .verify_us = @intCast(micros(v.verify)),
        };
        p.cursor = (p.cursor + 1) % p.audit.len;
        if (d.arm == .learned) { p.watch(d, v, round); return; }
        for ([_]bool{false,true}) |pooled| {
            const k = if (pooled) d.key.pooled() else d.key;
            const e = p.entry(k, pooled, round);
            e.learn(k, d, v, round);
            p.noteEvidence(d.key, e, round, if (pooled) 12 else 6);
        }
    }
    pub fn delivered(p: *Policy, s: *Donor, now: u64, round: u64) void {
        s.last_delivery_ns = now;
        const m = s.marked orelse return;
        s.marked = null;
        if (now <= m.since or !m.measured) return;
        const us = @max(m.prior_gap, now - m.since) / 1000;
        const c = p.counter(m.decision);
        c.delivery_samples += 1;
        c.delivery_us += us;
        c.delivery_max_us = @max(c.delivery_max_us, us);
        const d = m.decision;
        var worst = us;
        const count = @popCount(d.affected);
        if (count > 1) {
            const group = &p.deliveries[d.round_started % p.deliveries.len];
            const key: u64 = @bitCast(d.key);
            if (group.round != d.round_started or group.key != key) {
                if (group.round>d.round_started) { p.diagnostics.incomplete_donor_groups+=1; return; }
                if (group.n > 0) p.diagnostics.incomplete_donor_groups += 1;
                group.* = .{ .key = key, .round = d.round_started };
            }
            group.n += 1; group.worst_us = @max(group.worst_us, us);
            if (group.n != count) return;
            worst = group.worst_us; group.n = 0;
        }
        if (d.arm == .learned) {
            const local = p.entry(d.key, false, round);
            const pk=d.key.pooled();
            const pool=&p.pool[index(pk,p.pool.len)];
            const ctl=if (local.treatment_gap[0].n>=6) local.treatment_gap[0]
                else if(pool.matches(pk,round)) pool.treatment_gap[0] else Amount{};
            if (ctl.n>=6) {
                const late=number(worst)/1000 > 1.10*ctl.ms/number(ctl.n);
                local.late_live=if(late) local.late_live+|1 else 0;
                if (local.late_live>=3) p.quarantine(d.key,local,round);
            }
            return;
        }
        for ([_]bool{ false, true }) |pool| {
            const k = if (pool) m.decision.key.pooled() else m.decision.key;
            const e = if (pool) &p.pool[index(k, p.pool.len)] else &p.exact[index(k, p.exact.len)];
            if (e.matches(k, round)) { e.gap_returned[@backingInt(m.decision.arm)] +|= 1; e.treatment_gap[@backingInt(m.decision.arm)].add(.{
                .emitted = 1,
                .elapsed = number(worst) / 1000,
                .draft = 0,
                .verify = 0,
            }); p.noteEvidence(d.key, e, round, if (pool) 12 else 6); }
        }
    }
    pub fn priceShape(p: *const Policy, s: Shape, ctx: Context, round: u64) ?f64 {
        return p.costs[s.n - 1][s.id()][ctx.band][@intFromBool(ctx.sampled)].known(round);
    }
    pub fn timeShape(p: *Policy, s: Shape, ctx: Context, ms: f64, round: u64) void {
        if (s.n == 0 or s.n > 16 or s.steps == 0 or s.steps > 5 or !std.math.isFinite(ms) or ms <= 0 or ms > 60000) return;
        p.costs[s.n - 1][s.id()][ctx.band][@intFromBool(ctx.sampled)].add(ms, round);
    }
    pub fn observeUse(p: *Policy, n: usize, f: Feature, exposed: usize, granted: usize, kept: usize, round: u64) void {
        const row = &p.use[loadGroup(n)][@backingInt(f.source)][@min(5, f.grant)];
        for (0..@min(exposed, 5)) |i| row[i].add(i < granted, i < kept, round);
    }
    pub fn position(p: *const Policy, n: usize, f: Feature, i: usize, round: u64) Slot {
        const row = p.use[loadGroup(n)][@backingInt(f.source)][@min(5, f.grant)][i];
        const use_prior: f64 = if (i < f.grant) 0.85 else 0.35;
        const hit_prior: f64 = 0.86 - 0.04 * number(i);
        if (round -| row.last > 4096) return .{ .use = use_prior, .value = use_prior * hit_prior };
        const use = (8 * use_prior + number(row.used)) / (8 + number(row.seen));
        // Unconditional delivered prefix among offered positions. A mismatch ends the prefix;
        // it supplies no label about the correctness of later individual guesses.
        const hit = (8 * hit_prior + number(row.kept)) / (8 + number(row.used));
        return .{ .use = use, .value = use * hit };
    }
};

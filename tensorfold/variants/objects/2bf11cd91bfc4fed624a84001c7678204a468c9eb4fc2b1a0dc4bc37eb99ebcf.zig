//! Lifetime of a decision: prepare (optional) -> one receipt -> consuming verify -> delivery.
const std = @import("std");
const m = @import("marginal.zig");
const plan = @import("marginal_plan.zig");
const work = @import("spec_work.zig");

pub fn begin(c: anytype, inputs: anytype, warm: bool) void {
    const seq = if (inputs[0].receipt) |r| r.sequence else null;
    c.policy.startClock(c.now_ns);
    for (inputs) |input| if (!input.observable) {
        c.policy.continuous = false;
        break;
    };
    c.policy.begin(seq, warm and c.policy.continuous and c.frame.previous_ms != null);
    if (c.policy.current) |*d| {
        var found = false;
        var donor_found = d.donor == null;
        for (inputs, 0..) |f, i| {
            if (f.key == d.recipient_key) {
                d.recipient = @intCast(i);
                found = true;
            }
            if (d.donor != null and f.key == d.donor_key) {
                d.donor = @intCast(i);
                donor_found = true;
            }
        }
        if (!found or !donor_found or inputs.len != d.key.lanes or
            (d.copy != null and d.applied() and c.policy.copy_applied_seq != d.sequence)) {
            c.policy.lost(d.*);
            c.policy.current = null;
            return;
        }
        var mask: u16 = 0;
        for (c.policy.pending_donors, 0..) |entry,j| {
            if ((d.affected & m.bit(j)) == 0) continue;
            for (inputs,0..) |input,i| if (input.key == entry.key) {
                mask |= m.bit(i);
                break;
            };
        }
        if (@popCount(mask) != @popCount(d.affected)) {
            c.policy.lost(d.*); c.policy.current = null; return;
        }
        const original = d.affected;
        d.affected = mask;
        for (c.policy.pending_donors,0..) |entry,j| {
            if ((original & m.bit(j)) == 0) continue;
            for (inputs) |input| if (input.key == entry.key) {
                if (input.donor) |s| {
                    var mark = d.*; mark.donor_prior_ns = entry.prior_ns;
                    s.mark(mark,c.now_ns,c.rounds);
                }
                break;
            };
        }
        c.exploring = d.arm != .learned;
    }
}

fn probe(c: anytype, stage: m.Stage) bool {
    return c.policy.opportunity(stage, c.rounds, c.refresh.ready());
}
fn wanted(c: anytype, stage: m.Stage, funded: bool) ?m.Action {
    if (!funded) return null;
    return @fromBackingInt(@intCast(c.policy.funded[@backingInt(stage)] % @as(u64,if (stage == .prepare) 4 else 3)));
}
fn canPlan(c: anytype, funded: bool, stage: m.Stage) bool {
    return funded or c.policy.canSearch(stage, c.frame.count, c.rounds);
}
pub fn verify(c: anytype, inputs: anytype, prob: anytype, table: anytype, counts: anytype, rows: *u32, ctx: *u64, sampled: bool, cap: u32, ref: ?f64, over: f64, comptime context: anytype, comptime band: anytype) void {
    const rate = ref orelse return;
    if (!c.policy.continuous or c.policy.current != null or c.exploring or c.frame.previous_ms == null or inputs[0].receipt.?.shape == 0) return;
    const funded = probe(c, .verify);
    if (!canPlan(c, funded, .verify)) return;
    var base: m.Depths = @splat(0);
    for (inputs, 0..) |_, i| base[i] = @intCast(counts[i]);
    const alternative = plan.verify(c, inputs, prob, table, base, rows.*, cap, rate, wanted(c, .verify, funded), context, band) orelse {
        if (funded) c.policy.diagnostics.verify_idle += 1;
        return;
    };
    var d = c.policy.decide(alternative, rate, funded, c.rounds) orelse return;
    d.recipient_key = inputs[d.recipient].key;
    if (d.donor) |i| {
        d.donor_key = inputs[i].key;
    }
    for (inputs,0..) |input,i| if ((d.affected & m.bit(i)) != 0) {
        input.donor.?.mark(d,c.now_ns,c.rounds);
    };
    if (d.applied()) {
        if (d.key.action == .drop) inputs[d.recipient].donor.?.until = c.rounds +| 16;
        for (inputs, 0..) |_, i| {
            rows.* = rows.* - counts[i] + alternative.depths[i];
            counts[i] = alternative.depths[i];
        }
        ctx.* = context(inputs, counts[0..inputs.len]);
    }
    if (d.arm != .learned) {
        c.exploring = true;
        c.probes += 1;
        c.policy.debit(&d, @max(5, c.predictedCost(rows.*, inputs.len, ctx.*, sampled, table) + over), &c.refresh);
    }
    c.policy.current = d;
}

pub const Preparation = struct {
    n: usize,
    base: m.Depths,
    actual: m.Depths,
    room: m.Depths,
    features: [16]m.Feature,
    decision: ?m.Decision = null,
    context: m.Context = .{.band=0,.rows=0,.sampled=false,.sources=0},
};
pub fn preparation(e: anytype, requests: anytype, modified: anytype) ?Preparation {
    const c = e.learner orelse return null;
    if (!c.policy.enabled or requests.len == 0 or requests.len != c.frame.count or !c.frame.warm or
        c.frame.verify_ms == null or !e.backend.timingClean() or requests.len > 16) return null;
    var p = Preparation{ .n = requests.len, .base = @splat(0), .actual = @splat(0), .room = @splat(0), .features = undefined };
    var sources: u3 = 0;
    for (requests, 0..) |r, i| {
        const s = r.stream;
        const f = c.frame.streams[i];
        if (f.key != @intFromPtr(s) or r.early or r.ranks or r.lanes != null or s.finished or s.force.items.len != 0 or
            r.depth == 0 or r.depth > c.policy.block) return null;
        p.room[i] = @intCast(@min(@as(i64, c.policy.block), @max(0, s.draftRoom() -| 1)));
        if (r.depth > p.room[i]) return null;
        const src = plan.sourceOf(f);
        sources |= @as(u3, 1) << @backingInt(src);
        p.base[i] = @intCast(r.depth);
        p.features[i] = .{
            .source = src,
            .grant = @intCast(@min(5, f.granted)),
            .max = if (r.depth == 1) @min(c.policy.block, @max(1,s.copy_ceiling)) else c.policy.block,
            .room = p.room[i],
            .mutable = s.marginal_donor.available(c.rounds) and f.adjustable,
        };
    }
    p.actual = p.base;
    p.context = .{ .band=@intCast(c.contextIndex()), .rows=@intCast(c.frame.rows), .sampled=c.frame.sampled, .sources=sources };
    if (!c.policy.continuous or c.exploring or c.policy.current != null or c.policy.pending != null) return p;
    const funded = probe(c, .prepare);
    if (!canPlan(c, funded, .prepare)) return p;
    const rate = c.matchedRate(c.frame.count, c.frame.context, c.frame.sampled) orelse return p;
    const target = wanted(c, .prepare, funded);
    const curve = plan.Curve.build(c, &e.cfg.shared_costs, e.cfg.batch_rows);
    const copy = if (target == .copy_select or (!funded and c.policy.copy_until[m.loadGroup(p.n)] >= c.rounds and c.rounds > 0))
        @import("windows.zig").copyMove(e, requests, &p, &curve, rate) else null;
    const alternative = if (copy != null and (funded or c.policy.supported(copy.?,c.rounds))) copy.? else
        plan.prepare(&c.policy, p.base, p.features[0..p.n], p.context, rate, &curve, c.rounds, target) orelse {
            if (funded) c.policy.diagnostics.prepare_idle += 1;
            return p;
        };
    var d = c.policy.decide(alternative, rate, funded, c.rounds) orelse return p;
    d.recipient_key = @intFromPtr(requests[d.recipient].stream);
    if (d.donor) |i| {
        d.donor_key = @intFromPtr(requests[i].stream);
        d.donor_prior_ns = requests[i].stream.marginal_donor.last_delivery_ns;
    }
    for (requests,0..) |r,i| if ((d.affected & m.bit(i)) != 0) {
        c.policy.pending_donors[i] = .{ .key=@intFromPtr(r.stream), .prior_ns=r.stream.marginal_donor.last_delivery_ns };
    };
    // Charge even if the GPU call fails, a job ends or the cohort changes before consumption.
    c.policy.debit(&d, @max(5, c.frame.predicted_ms + (c.frame.previous_ms orelse 0)), &c.refresh);
    if (d.arm != .learned) c.probes += 1;
    c.policy.outgoing = d;
    p.decision = d;
    if (d.applied()) {
        p.actual = alternative.depths;
        for (requests, 0..) |r, i| {
            modified[i] = r;
            modified[i].depth = p.actual[i];
            if ((d.affected & m.bit(i)) != 0) r.stream.marginal_donor.until = c.rounds +| 16;
        }
    }
    return p;
}

/// Match the one actual pass, including its physical ordering, against the proposed shape.
pub fn prepared(c: anytype, p: Preparation, requests: anytype, receipt: *?work.Receipt, physical_ms: f64) void {
    const phys = m.Physical.plan(p.actual, p.room, p.n);
    var valid = receipt.* != null and c.frame.warm;
    for (requests, 0..) |r, i| {
        const w = r.stream.prepared_work orelse {
            valid = false;
            continue;
        };
        if (w.requested != p.actual[i] or w.exposed != phys.exposed[i] or w.projected != phys.projected[i]) valid = false;
    }
    if (valid) {
        receipt.*.?.shape = phys.shape.id();
        c.policy.timeShape(phys.shape, p.context, physical_ms, c.rounds);
        for (requests, 0..) |r, i| r.stream.marginal_feature = .{ .sequence = receipt.*.?.sequence, .feature = p.features[i], .exposed = phys.exposed[i] };
    }
    if (p.decision) |old| {
        var d = old;
        c.policy.outgoing = null;
        c.policy.debit(&d, physical_ms, &c.refresh);
        if (!valid) {
            c.policy.lost(d);
            return;
        }
        d.sequence = receipt.*.?.sequence;
        c.policy.pending = d;
    }
}

pub fn finish(c: anytype, emitted: usize, matched: f64, verify_ms: f64) void {
    if (!c.policy.continuous) return;
    const full = matched + c.policy.gap_ms;
    if (!c.exploring and c.policy.current == null) c.rememberMatched(emitted, full);
    for (c.frame.streams[0..c.frame.count]) |f| {
        const origin = f.origin orelse continue;
        if (!f.observable) continue;
        const used = if (f.source == .head and f.adjustable) f.granted else 0;
        c.policy.observeUse(c.frame.count, origin.feature, origin.exposed, used, @min(f.delivered_kept, used), c.rounds);
    }
    if (c.policy.current) |*d| {
        c.policy.debit(d, full, &c.refresh);
        const rec = c.frame.streams[d.recipient];
        const donor = if (d.donor) |i| c.frame.streams[i] else rec;
        for (c.frame.streams[0..c.frame.count],0..) |f,i| {
            if ((d.affected & m.bit(i)) != 0) if (f.donor) |ptr| if (ptr.marked) |*mark| { mark.measured = true; };
        }
        c.policy.end(.{
            .emitted = emitted,
            .elapsed = full,
            .gap_ms = c.policy.gap_ms,
            .draft = c.frame.previous_ms.?,
            .verify = verify_ms,
            .offered = rec.granted,
            .kept = rec.delivered_kept,
            .donor_offered = if (d.donor != null) donor.granted else 0,
            .donor_kept = if (d.donor != null) donor.delivered_kept else 0,
        }, c.rounds);
    }
}

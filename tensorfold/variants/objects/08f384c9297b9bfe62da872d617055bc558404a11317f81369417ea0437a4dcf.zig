//! Bounded neighbouring plans. The prepare planner prices the GPU layout of the whole vector.
const std = @import("std");
const m = @import("marginal.zig");
const Depths = m.Depths;
const N = m.max_lanes;
fn bucket(score: f64) u2 {
    return if (score < -0.25) 0 else if (score < 0) 1 else if (score < 0.25) 2 else 3;
}
pub fn sourceBit(s: m.Source) u3 {
    return @as(u3, 1) << @as(u2, @intCast(@backingInt(s)));
}
pub fn sourceOf(f: anytype) m.Source {
    return if (f.source != .copy) .head else if (f.shared_copy) .shared_copy else .local_copy;
}

// The next verification may use a fractional expected number of rows. Interpolate real
// neighbouring row costs and charge the displaced prefix when the whole batch is full.
pub const Curve = struct {
    ms: [49]f64 = @splat(0),
    seen: [49]bool = @splat(false),
    rows: u8, minimum: u8, cap: u8, displacement: f64,
    pub fn build(c: anytype, table: anytype, cap: u32) Curve {
        const f = &c.frame;
        var v: Curve = .{ .rows = @intCast(f.rows), .minimum = @intCast(f.count), .cap = @intCast(cap), .displacement = f.last_value };
        for (f.count..cap + 1) |i| {
            v.ms[i] = c.predictedCost(@intCast(i), f.count, f.context, f.sampled, table);
            v.seen[i] = c.observedCost(@intCast(i), f.count, f.context, f.sampled);
        }
        return v;
    }
    pub fn charge(v: *const Curve, shift: f64, rate: f64) struct { value: f64, measured: bool } {
        if (@abs(shift) < 0.000001) return .{ .value = 0, .measured = true };
        const raw = m.number(v.rows) + shift;
        const bounded = @max(m.number(v.minimum), @min(m.number(v.cap), raw));
        const lo: usize = @intFromFloat(@floor(bounded));
        const hi: usize = @intFromFloat(@ceil(bounded));
        const frac = bounded - m.number(lo);
        const ms = v.ms[lo] + frac * (v.ms[hi] - v.ms[lo]) - v.ms[v.rows];
        const cost = if (v.rows == v.cap and shift < 0) @max(0, ms) else ms;
        return .{ .value = rate * cost + @max(0,raw-m.number(v.cap)) * v.displacement,
            .measured = v.seen[v.rows] and v.seen[lo] and v.seen[hi] };
    }
};

const Planner = struct {
    policy: *const m.Policy,
    rate: f64,
    round: u64,
    n: usize,
    ctx: m.Context,
    base: Depths,
    room: Depths,
    features: []const m.Feature,
    physical: m.Physical,
    base_ms: f64,
    slot: [N][5]f64,
    use: [N][5]f64,
    curve: *const Curve,
    best: ?m.Move = null,
    preferred: ?m.Action = null,

    fn consider(p: *Planner, depths: Depths) void {
        var recipient: usize = 0;
        var increased: usize = 0;
        var decreased: usize = 0;
        var donor: ?u8 = null;
        var removed: u16 = 0;
        for (0..p.n) |i| {
            if (depths[i] > p.base[i]) {
                recipient = i;
                increased += 1;
            }
            if (depths[i] < p.base[i]) {
                donor = @intCast(i);
                removed |= m.bit(i);
                decreased += 1;
            }
        }
        if (increased + decreased == 0) return;
        const plan = m.Physical.plan(depths, p.room, p.n);
        const b = p.physical.shape;
        const a = plan.shape;
        var reward: f64 = 0;
        var different = false;
        var extra_rows: f64 = 0;
        var loss_sources:u3=0;
        var loss_positions:u5=0;
        for (0..p.n) |i| {
            const old = p.physical.exposed[i];
            const new = plan.exposed[i];
            if (new < old) {
                if (!p.features[i].mutable) return;
                removed |= m.bit(i);
            }
            if ((removed & m.bit(i)) != 0) {
                loss_sources |= sourceBit(p.features[i].source);
                for (@min(new,old)..old) |j| loss_positions |= @as(u5,1)<<@as(u3,@intCast(j));
            }
            if (old == new) continue;
            different = true;
            const sign: f64 = if (new > old) 1 else -1;
            for (@min(old, new)..@max(old, new)) |d| {
                reward += sign * p.slot[i][d];
                extra_rows += sign * p.use[i][d];
            }
        }
        if (!different and a.id() == b.id()) return;
        // A speculative estimate only ranks unexplored shapes for bounded measurement.
        const price = p.policy.priceShape(a, p.ctx, p.round);
        const forecast = price orelse
            p.base_ms * m.number(a.rows()) / m.number(b.rows());
        const verification = p.curve.charge(extra_rows, p.rate);
        const score = m.gain(reward - verification.value, p.rate, forecast - p.base_ms);
        const action: m.Action = if (increased == 0) .drop else if (decreased == 0) .add else .swap;
        if (p.best) |old| {
            if (!prefer(action, score, old, p.preferred)) return;
        }
        if (action == .drop) recipient = donor.?;
        const d = if (action == .swap) donor else null;
        const candidate: m.Move = .{
            .depths = depths,
            .predicted_gain = score,
            .measured_cost = price != null and verification.measured,
            .affected = removed,
            .recipient = @intCast(recipient),
            .donor = d,
            .key = .{
                .stage = .prepare,
                .action = action,
                .lanes = @intCast(p.n),
                .context = p.ctx.band,
                .sampled = p.ctx.sampled,
                .sources = p.ctx.sources,
                .source = p.features[recipient].source,
                .donor_source = if (d) |v| p.features[v].source else .head,
                .context_after = p.ctx.band,
                .before = b.id(),
                .after = a.id(),
                .rows = p.ctx.rows,
                .depth = @intCast(p.base[recipient]),
                .donor_depth = if (d) |v| @intCast(p.base[v]) else 0,
                .changed = @intCast(increased + decreased),
                .loss_sources=loss_sources, .loss_positions=loss_positions,
                .band = bucket(score),
            },
        };
        if (p.preferred != null or p.policy.supported(candidate, p.round)) p.best = candidate;
    }
};

fn prefer(action: m.Action, score: f64, old: m.Move, wanted: ?m.Action) bool {
    if (wanted) |w| {
        if (action != old.key.action and (action == w or old.key.action == w)) return action == w;
    }
    return score > old.predicted_gain;
}
pub fn prepare(policy: *const m.Policy, base: Depths, features: []const m.Feature, ctx: m.Context, rate: f64, curve: *const Curve, round: u64, preferred: ?m.Action) ?m.Move {
    if (features.len == 0 or features.len > N or rate <= 0 or !std.math.isFinite(rate)) return null;
    var room: Depths = @splat(0);
    for (features, 0..) |f, i| {
        if (base[i] < 1 or base[i] > 5 or f.max < base[i] or f.max > 5 or f.room > 5) return null;
        room[i] = f.room;
    }
    const physical = m.Physical.plan(base, room, features.len);
    const base_ms = policy.priceShape(physical.shape, ctx, round) orelse return null;
    var p: Planner = .{
        .policy = policy,
        .base = base,
        .room = room,
        .n = features.len,
        .ctx = ctx,
        .features = features,
        .rate = rate,
        .round = round,
        .physical = physical,
        .base_ms = base_ms,
        .slot = undefined,
        .use = undefined,
        .curve = curve,
        .preferred = preferred,
    };
    var cheapest: ?usize = null;
    var best_add: ?usize = null;
    for (features, 0..) |f, i| {
        for (0..5) |d| {
            const pos = policy.position(features.len, f, d, round);
            // Conditional use and path survival already folded into expected reward.
            p.slot[i][d] = pos.value;
            p.use[i][d] = pos.use;
        }
        if (!f.mutable) continue;
        if (base[i] > 1 and (cheapest == null or p.slot[i][base[i] - 1] < p.slot[cheapest.?][base[cheapest.?] - 1])) cheapest = i;
        if (base[i] < @min(f.max, f.room) and (best_add == null or p.slot[i][base[i]] > p.slot[best_add.?][base[best_add.?]])) best_add = i;
    }
    var lower_step = base;
    for (features, 0..) |f, i| {
        if (!f.mutable) continue;
        if (base[i] < @min(f.max, f.room)) {
            var d = base;
            d[i] += 1;
            p.consider(d);
        }
        if (base[i] > 1) {
            var d = base;
            d[i] -= 1;
            p.consider(d);
        }
        if (base[i] == physical.shape.steps and base[i] > 1) lower_step[i] -= 1;
    }
    p.consider(lower_step);
    if (cheapest) |first| if (best_add) |best| {
        // Also consider which other donor can actually change the physical layout.
        var second: ?usize = null;
        var boundary: ?usize = null;
        for (features, 0..) |f,i| {
            if (!f.mutable or base[i] <= 1) continue;
            if ((base[i] == physical.shape.steps or base[i] + 1 == physical.shape.steps) and
                (boundary == null or p.slot[i][base[i]-1] < p.slot[boundary.?][base[boundary.?]-1])) boundary=i;
            if (i == first) continue;
            if (second == null or p.slot[i][base[i]-1] < p.slot[second.?][base[second.?]-1]) second = i;
        }
        for ([_]?usize{first,second,boundary},0..) |candidate,j| {
            if (j==2 and (candidate==first or candidate==second)) continue;
            const donor = candidate orelse continue;
            var receiver: ?usize = if (best != donor) best else null;
            if (receiver == null) for (features,0..) |f,i| {
                if (i != donor and f.mutable and base[i] < @min(f.max,f.room) and
                    (receiver == null or p.slot[i][base[i]] > p.slot[receiver.?][base[receiver.?]])) receiver=i;
            };
            if (receiver) |i| {
                var d=base; d[i]+=1; d[donor]-=1; p.consider(d);
            }
        }
    };
    // Cross the compact boundary in one move, even when individual reductions save no work.
    if (features.len >= 4 and physical.shape.steps >= 3 and !physical.shape.compact) {
        var d = base;
        const short = physical.shape.steps - 2;
        var long: usize = 0;
        for (d[0..features.len]) |k| long += @intFromBool(k > short);
        while (long > (features.len + 1) / 2) {
            var best: ?usize = null;
            var loss: f64 = std.math.inf(f64);
            for (features, 0..) |f, i| {
                if (!f.mutable or d[i] <= short) continue;
                var cost: f64 = 0;
                for (short..d[i]) |k| cost += p.slot[i][k];
                if (cost < loss) {
                    loss = cost;
                    best = i;
                }
            }
            const i = best orelse break;
            d[i] = short;
            long -= 1;
        }
        if (long <= (features.len + 1) / 2) p.consider(d);
    }
    return p.best;
}

pub fn verify(c: anytype, inputs: anytype, prob: anytype, baseline: anytype, base: Depths, rows: u32, cap: u32, rate: f64, preferred: ?m.Action, comptime context: anytype, comptime contextBand: anytype) ?m.Move {
    const n = inputs.len;
    var counts: [N]u32 = @splat(0);
    var sampled = false;
    var sources: u3 = 0;
    var donor1: ?usize = null;
    var donor2: ?usize = null;
    var boundary_donor: ?usize = null;
    for (inputs, 0..) |f, i| {
        counts[i] = base[i];
        sampled = sampled or f.sampled;
        sources |= sourceBit(sourceOf(f));
        if (!f.adjustable or base[i] == 0 or base[i] > 5 or f.donor == null or !f.donor.?.available(c.rounds)) continue;
        if (boundary_donor == null or f.context +| f.fixed +| base[i] > inputs[boundary_donor.?].context +| inputs[boundary_donor.?].fixed +| base[boundary_donor.?]) boundary_donor = i;
        if (donor1 == null or prob[i][base[i] - 1] < prob[donor1.?][base[donor1.?] - 1]) {
            donor2 = donor1;
            donor1 = i;
        } else if (donor2 == null or prob[i][base[i] - 1] < prob[donor2.?][base[donor2.?] - 1]) donor2 = i;
    }
    const ctx = context(inputs, counts[0..n]);
    const ms = c.predictedCost(rows, n, ctx, sampled, baseline);
    var best: ?m.Move = null;
    for (inputs, 0..) |f, i| {
        if (!f.adjustable or base[i] > 5) continue;
        // For a swap the cheapest lost token and the request at the context boundary
        // can be different donors. At most two distinct swaps are priced per recipient.
        for ([_]m.Action{ .add, .drop, .swap, .swap }, 0..) |action, index| {
            const cheapest = if (donor1 == i) donor2 else donor1;
            const donor = if (index == 3) boundary_donor else cheapest;
            if (index == 3 and (donor == null or donor == cheapest or donor.? == i)) continue;
            if (action == .drop) {
                if (base[i] == 0 or f.donor == null or !f.donor.?.available(c.rounds)) continue;
            } else if (base[i] >= @min(5, @min(f.limit, f.survivals.len))) continue;
            if (action == .add and rows >= cap or action == .swap and donor == null) continue;
            var next = counts;
            var reward: f64 = 0;
            var nr = rows;
            if (action == .drop) {
                next[i] -= 1;
                nr -= 1;
                reward -= prob[i][next[i]];
            } else {
                reward += prob[i][base[i]];
                next[i] += 1;
                if (action == .swap) {
                    next[donor.?] -= 1;
                    reward -= prob[donor.?][next[donor.?]];
                } else nr += 1;
            }
            const nx = context(inputs, next[0..n]);
            const score = m.gain(reward, rate, c.predictedCost(nr, n, nx, sampled, baseline) - ms);
            if (best) |b| {
                if (!prefer(action, score, b, preferred)) continue;
            }
            var d = base;
            d[i] = @intCast(next[i]);
            const dn: ?u8 = if (action == .swap) @intCast(donor.?) else null;
            if (dn) |di| d[di] = @intCast(next[di]);
            const shape: u4 = if (f.receipt) |r| r.shape else 0;
            const candidate: m.Move = .{
                .key = .{
                    .stage = .verify,
                    .action = action,
                    .lanes = @intCast(n),
                    .context = @intCast(contextBand(ctx)),
                    .context_after = @intCast(contextBand(nx)),
                    .sampled = sampled,
                    .sources = sources,
                    .source = sourceOf(f),
                    .before = shape,
                    .after = shape,
                    .rows = @intCast(rows),
                    .depth = @intCast(@min(7, base[i])),
                    .donor_depth = if (dn) |di| @intCast(@min(7, base[di])) else 0,
                    .donor_source = if (dn) |di| sourceOf(inputs[di]) else .head,
                    .loss_sources=if(dn) |di| sourceBit(sourceOf(inputs[di])) else if(action==.drop) sourceBit(sourceOf(f)) else 0,
                    .loss_positions=if(dn) |di| @as(u5,1)<<@as(u3,@intCast(base[di]-1)) else if(action==.drop) @as(u5,1)<<@as(u3,@intCast(base[i]-1)) else 0,
                    .band = bucket(score),
                },
                .depths = d,
                .recipient = @intCast(i),
                .donor = dn,
                .predicted_gain = score,
                .affected = if (dn) |j| m.bit(j) else if (action == .drop) m.bit(i) else 0,
                .measured_cost = c.observedCost(rows, n, ctx, sampled) and c.observedCost(nr, n, nx, sampled),
            };
            if (preferred != null or c.policy.supported(candidate, c.rounds)) best = candidate;
        }
    }
    return best;
}

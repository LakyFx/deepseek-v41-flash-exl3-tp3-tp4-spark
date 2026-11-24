//! A round's windows (Python `_plan_window`, `_copy_proposal`, `_allocate`): forced, copied or drafted rows.
const std = @import("std");
const accept = @import("accept.zig");
const alloc = @import("allocate.zig");
const be = @import("backend.zig");
const depth = @import("depth.zig");
const fill = @import("fill.zig");
const shape = @import("shape.zig");
const ev = @import("events.zig");
const trail = @import("trail.zig");
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const Value = ev.Value;
const f = ev.f;
const adaptive = @import("adaptive.zig");
const marginal = @import("marginal.zig");
const work = @import("spec_work.zig");
const OriginScorer = @import("proposer.zig").OriginScorer;

pub const Kind = enum { forced, copy, head, fill, none };

/// One stream's window as planned, then as allocated (Python's plan lists).
pub const Plan = struct {
    stream: *Stream,
    position: u64, // the stream's cache length when the round started
    kind: Kind,
    receipt: ?@import("spec_work.zig").Receipt = null,
    requested: u32 = 0,
    origin:?@import("marginal.zig").Origin=null,
    copy_rate: ?f64 = null,
    shared_copy:bool=false,
    copy_band:u2=0, copy_scored:bool=false,
    origin_changed: bool = false,
    origin_contest: bool = false,
    winner_changed: bool = false,
    choice: enum { none, head, copy } = .none,
    held: u32 = 0, // head drafts the backend holds
    tokens: []const u32 = &.{}, // drafts known on the host (forced, copied, tree)
    forced: []const u32 = &.{},
    parents: ?[]const i32 = null, // a tree's drafts' parents
    branch_from: u32 = 0, // the first row of suffix-match branches (0: none; lanes/fill.zig)
    tried: bool = false, // the planner read the head's chain back to graft branches
    lanes: ?*const shape.Shape = null, // the held drafts' tree shape (each lane's depth and rank)

    pub fn count(p: Plan) usize {
        return p.held + p.tokens.len;
    }
};

/// Forced tokens first, then a copied continuation, then the head's drafts (Python `_plan_window`).
pub fn plan(e: *Engine, s: *Stream, copied: ?[]const u32, streams:usize) !Plan {
    const a = e.arena.allocator();
    var p: Plan = .{ .stream = s, .position = s.cache_len, .kind = .none };
    const queued = s.next;
    s.next = null;
    if (queued) |q| {
        if (e.learner) |c| if (streams<=adaptive.max_streams) { c.work[streams-1].head_available+=q.count; };
        p.receipt=q.receipt; p.requested=q.requested orelse q.count;
        if(s.marginal_feature) |o| if(q.receipt) |r| {if(o.sequence==r.sequence)p.origin=o;};
    }
    s.marginal_feature=null;
    defer if (queued) |q| {
        if (q.tokens) |t| e.gpa.free(t);
        if (q.parents) |parents| e.gpa.free(parents);
    };
    if (s.force.items.len > 0) {
        const width: usize = if (s.drafts) @min(e.cfg.base_width, e.cfg.batch_rows) else 1;
        const take = @min(width - 1, s.force.items.len);
        const forced = try a.dupe(u32, s.force.items[0..take]);
        s.force.replaceRangeAssumeCapacity(0, take, &.{});
        p.kind = .forced;
        p.forced = forced;
        p.tokens = forced;
    } else {
        var copy: []const u32 = &.{};
        if (copied) |c| copy = c else if (s.drafts) {
            const terms = if (e.copy_decision) |d| (if (d.recipient_key == @intFromPtr(s)) d.copy else null) else null;
            const limit: usize = if (terms) |t| t.limit else std.math.maxInt(usize);
            const price=if (terms) |t| t.price else if(e.learner!=null and s.proposer!=null and s.proposer.?.vtable.candidate!=null) e.learner.?.rowPrice(streams,s.cache_len,
                if(s.sampling) |sampling| sampling.temperature>0 else false,&e.cfg.shared_costs) else 0.40;
            const sampled = if (s.sampling) |v| v.temperature > 0 else false;
            const scorer: ?OriginScorer = if (e.learner) |c| origin: {
                if (!c.origin_scoring or e.cfg.fill_lanes != 0 or terms != null or streams == 0 or streams > adaptive.max_streams) break :origin null;
                break :origin .{ .ctx = c, .sampled = sampled, .limit = e.cfg.batch_rows -| streams, .value = originValue };
            } else null;
            if (try queryUsing(e,s,price,true,false,terms,scorer)) |candidate| {
                if(terms!=null) e.learner.?.policy.copy_applied_seq=e.copy_decision.?.sequence;
                copy=candidate.tokens;
                if (e.learner) |c| if (streams <= adaptive.max_streams) {
                    const w = &c.work[streams-1]; w.local_available += candidate.local_available; w.shared_available += candidate.shared_available;
                };
                p.copy_rate=candidate.rate; p.shared_copy=candidate.global;
                p.copy_band=if(candidate.matched>=24) 2 else if(candidate.matched>=12) 1 else 0;
                p.origin_changed = candidate.origin_changed;
                const ordinary_fit = scorer != null and e.learner.?.originReady(sampled, candidate.global, p.copy_band);
                p.copy_scored = terms != null or ordinary_fit;
                if(copy.len>0 and queued!=null and queued.?.count>0 and queued.?.parents==null and queued.?.tokens==null) {
                    if (e.backend.vtable.probabilities) |probabilities| {
                        var scores:[adaptive.max_rows]f64=undefined;
                        const count=@min(queued.?.count,scores.len);
                        if (try probabilities(e.backend.ptr,s,scores[0..count])) {
                            const original_head = work.headValue(scores[0..@min(count,limit)],price) >
                                work.copyValue(candidate.rate,@min(copy.len,limit),price);
                            var head = original_head;
                            if (ordinary_fit) {
                                const c = e.learner.?;
                                const cap = scorer.?.limit;
                                const h = c.headValue(sampled, scores[0..@min(count,cap)], price);
                                const v = c.originValue(sampled, candidate.global, p.copy_band,
                                    candidate.rate, @min(copy.len,cap), price).?;
                                if (h != v) head = h > v;
                                p.origin_contest = true;
                                p.winner_changed = head != original_head;
                            }
                            if (head) {
                                copy=&.{}; p.choice=.head;
                            } else p.choice=.copy;
                        }
                    }
                }
                _=try queryUsing(e,s,price,false,copy.len>0,terms,scorer);
            } else copy=try copyProposal(e,s);
        }
        if (copy.len > 0) {
            p.kind = .copy;
            p.tokens = try a.dupe(u32, copy);
        } else if (s.drafts and queued != null) {
            const q = queued.?;
            if (q.tokens) |t| {
                p.tokens = try a.dupe(u32, t);
                if (q.parents) |parents| {
                    if (!accept.isChain(parents)) p.parents = try a.dupe(i32, parents);
                }
            } else {
                p.held = q.count;
                if (s.lanes) |*l| {
                    // a held tree is a lone stream's (shared rounds verify chains); a chain's lanes are observed too
                    if (q.parents) |parents| {
                        if (e.alone) p.parents = try a.dupe(i32, parents) else p.held = 0;
                    }
                    if (p.held > 0) p.lanes = l;
                }
            }
            if (p.count() > 0) p.kind = .head;
        }
    }
    if (e.cfg.fill_lanes > 0 and s.drafts and p.kind != .forced and (p.parents == null or p.lanes != null)) try fillLanes(e, s, &p);
    try trail.event(e, &.{ f("ev", trail.str("plan")), f("stream", trail.str(s.id)), f("kind", trail.str(@tagName(p.kind))), f("n", trail.int(p.count())), f("tree", .{ .bool = p.parents != null }) });
    return p;
}

/// Suffix-match branches grafted onto the planned drafts, up to cfg.fill_lanes rows.
fn fillLanes(e: *Engine, s: *Stream, p: *Plan) !void {
    const a = e.arena.allocator();
    const first = p.count();
    const room = @min(@as(usize, e.cfg.fill_lanes) -| (1 + first), s.graft_room orelse std.math.maxInt(u32));
    if (room == 0) return;
    const found = try fill.suffixCandidates(a, s.context.items, 16, e.cfg.fill_match, room, 8);
    if (found.len == 0 or (s.graft_hit < e.cfg.fill_floor and s.rounds % 8 != 0)) return;
    // the trunk's lanes: the drafts in order for a chain, a tree's rank-0 path from the pending row
    var trunk_lanes: [shape.max_depth * shape.ranks * 4]usize = undefined;
    const levels = if (p.lanes) |l| shape.chainLanes(l.*, &trunk_lanes) else first;
    if (p.lanes == null) {
        for (trunk_lanes[0..first], 0..) |*x, k| x.* = k;
    }
    const trunk = try headTrunk(e, s, p.*, levels) orelse return;
    p.tried = p.held > 0;
    const b = try fill.graft(a, trunk, found, room);
    if (b.tokens.len == 0) return;
    const parents = try a.alloc(i32, first + b.tokens.len);
    if (p.parents) |held| {
        @memcpy(parents[0..first], held);
    } else {
        for (parents[0..first], 0..) |*q, k| q.* = @as(i32, @intCast(k)) - 1;
    }
    // a graft's parent: the pending row, a trunk level (that level's lane), or one of its own rows (after the drafts)
    for (parents[first..], b.parents) |*q, g| q.* = if (g < 0) -1 else if (g < levels) @intCast(trunk_lanes[@intCast(g)]) else @intCast(first + @as(usize, @intCast(g)) - levels);
    p.tokens = if (p.held > 0) b.tokens else try std.mem.concat(a, u32, &.{ p.tokens, b.tokens });
    p.parents = parents;
    p.branch_from = @intCast(1 + first);
    if (p.kind == .none) p.kind = .fill;
}

/// The planned drafts' trunk of `levels` lanes: host drafts as they are, held ones with the head's ranks (null: unread).
fn headTrunk(e: *Engine, s: *Stream, p: Plan, levels: usize) !?fill.Trunk {
    if (p.held == 0) return .{ .tokens = p.tokens };
    const a = e.arena.allocator();
    const read = e.backend.vtable.alternatives orelse return null;
    const alts = try a.alloc(be.Alternative, levels);
    if (try read(e.backend.ptr, s, alts) < levels) return null;
    const chain = try a.alloc(u32, levels);
    const others = try a.alloc([3]u32, levels);
    for (alts, chain, others) |x, *t, *o| {
        t.* = x.tokens[0];
        o.* = x.tokens[1..4].*;
    }
    return .{ .tokens = chain, .others = others };
}

/// Copied spans backed by `enter_match` matching tokens (Python `_copy_proposal`); the proposer owns them.
pub fn copyProposal(e: *Engine, s: *Stream) ![]const u32 {
    const p = s.proposer orelse return &.{};
    if (e.cfg.max_copy == 0 or s.force.items.len > 0) return &.{};
    const width: u32 = if (e.alone) (s.copy_width orelse e.cfg.first_copy) else e.cfg.first_copy;
    const copied = p.propose(s.context.items, @min(@as(i64, width), s.draftRoom() - 1)) catch return &.{};
    if (copied.len < 2 or p.lastMatch() < e.cfg.enter_match) return &.{};
    return copied;
}

fn queryCopy(e:*Engine,s:*Stream,price:f64,preview:bool,offer:bool) !?@import("proposer.zig").Candidate {
    return queryWith(e,s,price,preview,offer,null);
}
fn queryWith(e: *Engine, s: *Stream, price: f64, preview: bool, offer: bool, terms: ?marginal.CopyTerms) !?@import("proposer.zig").Candidate {
    return queryUsing(e,s,price,preview,offer,terms,null);
}

fn originValue(ptr: *const anyopaque, sampled: bool, shared: bool, band: u2, rate: f64, count: usize, price: f64) ?f64 {
    const c: *const adaptive.Controller = @ptrCast(@alignCast(ptr));
    return c.originValue(sampled, shared, band, rate, count, price);
}

fn queryUsing(e: *Engine, s: *Stream, price: f64, preview: bool, offer: bool,
    terms: ?marginal.CopyTerms, scorer: ?OriginScorer) !?@import("proposer.zig").Candidate {
    if (!e.cfg.adaptive_learning) return null;
    const p = s.proposer orelse return null;
    const query = p.vtable.candidate orelse return null;
    if (e.cfg.max_copy == 0 or s.force.items.len > 0) return null;
    const width = if (e.alone) s.copy_width orelse e.cfg.first_copy else e.cfg.first_copy;
    var candidate = try query(p.ptr, s.context.items, .{ .width = @min(@as(i64, width), s.draftRoom() - 1), .round = s.rounds, .minimum = e.cfg.enter_match, .price = price, .preview = preview, .offer = offer, .limit = if (terms) |t| t.limit else std.math.maxInt(usize), .calibrated = terms != null, .scorer = scorer });
    if (candidate.tokens.len < 2 or candidate.matched < e.cfg.enter_match) candidate.tokens = &.{};
    return candidate;
}

// Only a measured prepare choice may change copy arbitration in its consuming round.
pub fn copyDecision(e: *Engine, entries: []const *Stream) ?marginal.Decision {
    const c = e.learner orelse return null;
    const d = c.policy.pending orelse return null;
    if (!d.applied() or d.copy == null or entries.len != d.key.lanes or c.policy.interrupted or
        c.now_ns -| c.policy.last_end_ns > 2 * std.time.ns_per_s or !e.backend.timingClean()) return null;
    var found = false;
    for (entries) |s| {
        const q = s.next orelse return null;
        const rec = q.receipt orelse return null;
        if (rec.sequence != d.sequence or rec.members != entries.len or rec.shape == 0 or s.force.items.len != 0 or
            s.finished or s.rounds <= 1 or q.parents != null or s.lanes != null) return null;
        found = found or @intFromPtr(s) == d.recipient_key;
    }
    return if (found) d else null;
}

pub fn copyMove(e: *Engine, requests: anytype, p: anytype, curve: anytype, rate: f64) ?marginal.Move {
    const c = e.learner.?;
    // This special probe compares the old source policy to an alternative.
    // Its reference ceases to describe the ordinary decision under 6g.
    if (c.origin_scoring) return null;
    const rows = c.frame.rows;
    if (rows == 0 or rate <= 0) return null;
    // Pricing the actual previous allocation bounds host work. Probe at most two streams.
    const next_rows = @min(e.cfg.batch_rows, rows + 1);
    const price = if (rows == e.cfg.batch_rows) c.frame.last_value else @min(1, @max(0, rate * (curve.ms[next_rows] - curve.ms[rows])));
    for (0..@min(2, p.n)) |offset| {
        const i = (c.rounds + offset) % p.n;
        const s = requests[i].stream;
        if (!p.features[i].mutable or s.proposer == null or e.cfg.max_copy == 0) continue;
        const grant = @min(@as(usize, @intCast(e.cfg.batch_rows)) - p.n, p.features[i].grant + @as(usize, 1));
        if (grant < 2) continue;
        const terms = marginal.CopyTerms{ .limit = @intCast(grant), .price = price };
        const old_price = c.rowPrice(p.n, s.cache_len, if (s.sampling) |v| v.temperature > 0 else false, &e.cfg.shared_costs);
        const old = (queryCopy(e, s, old_price, true, false) catch null) orelse continue;
        const alt = (queryWith(e, s, price, true, false, terms) catch null) orelse continue;
        if (alt.tokens.len < 2 or old.tokens.len < 2) continue;
        // Compare both policies under ONE price and grant. Truncating a confidence
        // estimate by itself delivers no tokens. The head term uses observed previous
        // conditional probabilities and does not issue another GPU read.
        const frame_stream = &c.frame.streams[i];
        var head: [5]f64 = undefined;
        const n = if (frame_stream.source == .head) @min(5, frame_stream.offered) else 0;
        var reach: f64 = 1;
        for (head[0..n], 0..) |*h, j| {
            reach *= frame_stream.raw[j];
            h.* = reach;
        }
        const head_value = work.headValue(head[0..@min(n, grant)], price);
        const prior_chose_head = n > 0 and work.headValue(head[0..n], old_price) > work.copyValue(old.rate, old.tokens.len, old_price);
        const old_rate = if (old.global) alt.shared_rate else alt.local_rate;
        const incumbent = if (prior_chose_head) head_value else work.copyValue(old_rate, @min(old.tokens.len, grant), price);
        const alternative = @max(head_value, work.copyValue(alt.rate, @min(alt.tokens.len, grant), price));
        var gain = alternative - incumbent;
        const old_layout = marginal.Physical.plan(p.base, p.room, p.n);
        const base_shape = old_layout.shape;
        var depths = p.base;
        const soft = s.copy_ceiling > 1 and depths[i] == 1;
        const strong = alt.tokens.len >= 3 and grant >= 3 and alt.matched >= 12 and alt.rate >= 0.90;
        if (soft and !strong and p.room[i] >= 2) depths[i] = 2;
        const new_layout = marginal.Physical.plan(depths, p.room, p.n);
        const next_shape = new_layout.shape;
        var shift: f64 = 0;
        var affected: u16 = 0;
        var loss_sources: u3 = 0;
        var loss_positions: u5 = 0;
        var blocked = false;
        for (0..p.n) |j| {
            const old_width = old_layout.exposed[j];
            const new_width = new_layout.exposed[j];
            if (new_width < old_width) {
                if (!p.features[j].mutable) {
                    blocked = true;
                    break;
                }
                affected |= marginal.bit(j);
                loss_sources |= @import("marginal_plan.zig").sourceBit(p.features[j].source);
                for (new_width..old_width) |slot| loss_positions |= @as(u5, 1) << @as(u3, @intCast(slot));
            }
            const sign: f64 = if (new_width > old_width) 1 else -1;
            for (@min(old_width, new_width)..@max(old_width, new_width)) |position| {
                const slot = c.policy.position(p.n, p.features[j], position, c.rounds);
                gain += sign * slot.value;
                shift += sign * slot.use;
            }
        }
        if (blocked) continue;
        const verify = curve.charge(shift, rate);
        gain -= verify.value;
        if (gain <= 0 and depths[i] == p.base[i]) continue;
        const base_ms = c.policy.priceShape(base_shape, p.context, c.rounds);
        const next_ms = c.policy.priceShape(next_shape, p.context, c.rounds);
        const timed = base_ms != null and next_ms != null;
        const draft_delta = if (timed) next_ms.? - base_ms.? else 0;
        const shape_id = base_shape.id();
        return .{ .key = .{ .stage = .prepare, .action = .copy_select, .lanes = @intCast(p.n), .context = p.context.band, .context_after = p.context.band, .sampled = p.context.sampled, .sources = p.context.sources, .source = if (alt.global) .shared_copy else .local_copy, .donor_source = if (old.global) .shared_copy else .local_copy, .before = shape_id, .after = next_shape.id(), .rows = @intCast(rows), .depth = @intCast(p.base[i]), .loss_sources = loss_sources, .loss_positions = loss_positions, .match_band = if (alt.matched >= 24) 2 else if (alt.matched >= 12) 1 else 0, .band = if (gain >= 0.25) 3 else 2 }, .depths = depths, .affected = affected, .predicted_gain = marginal.gain(gain, rate, draft_delta), .recipient = @intCast(i), .copy = terms, .measured_cost = timed and verify.measured and c.observedCost(rows, p.n, c.frame.context, c.frame.sampled) and
            c.observedCost(@min(e.cfg.batch_rows, rows + 1), p.n, c.frame.context, c.frame.sampled) };
    }
    return null;
}

pub fn copyAhead(ptr: *anyopaque, w: *const depth.Who) bool {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    const s: *Stream = @ptrCast(@alignCast(w.stream.?));
    if (queryCopy(e,s,0.40,true,false) catch return false) |candidate|
        return candidate.tokens.len>=3 and candidate.matched>=12 and candidate.rate>=0.90;
    const copied = copyProposal(e, s) catch return false;
    return copied.len > 0;
}

pub fn probe(e: *Engine) depth.Probe {
    return .{ .ptr = e, .copy = copyAhead };
}

pub fn who(s: *Stream) depth.Who {
    return .{ .state = &s.depth, .draft_room = s.draftRoom(), .finished = s.finished, .forced = s.force.items.len > 0, .granted = s.granted, .copy_ceiling=&s.copy_ceiling, .stream = s };
}

/// Trim draft prefixes by landing probability to the shared width with the most expected tokens a ms.
pub fn allocate(e: *Engine, plans: []Plan) !void {
    const a = e.arena.allocator();
    const fixed = try a.alloc(u32, plans.len);
    const probs = try a.alloc([]const f64, plans.len);
    for (plans, fixed, probs) |p, *fx, *pr| {
        const n = p.count();
        if (p.kind == .forced or n == 0) {
            fx.* = 1 + @as(u32, @intCast(if (p.kind == .forced) n else 0));
            pr.* = &.{};
        } else if (p.kind == .copy) {
            fx.* = 1;
            pr.* = try alloc.chainProbabilities(a, &.{p.copy_rate orelse e.cfg.copy_rate}, n);
        } else {
            fx.* = 1;
            pr.* = try nodeChances(e, p.stream, n);
        }
    }
    const learner = try e.learningController();
    const counts = if (learner) |model| block: {
        const inputs = try a.alloc(adaptive.Input, plans.len);
        for (inputs, plans, fixed, probs) |*input, p, fx, pr| input.* = .{
            .origin=p.origin,.donor=&p.stream.marginal_donor,
            .observable=p.kind!=.forced and p.parents==null and p.branch_from==0 and p.lanes==null,
            .adjustable=(p.kind==.head or p.kind==.copy) and p.parents==null and p.branch_from==0 and p.lanes==null,
            .limit=@intCast(@min(5,@max(0,p.stream.draftRoom()-|1))),
            .receipt=p.receipt, .requested=p.requested, .shared_copy=p.shared_copy, .copy_band=p.copy_band, .copy_scored=p.copy_scored,
            .key = @intFromPtr(p.stream), .context = p.stream.cache_len, .fixed = fx,
            .survivals = pr, .source = if (p.kind == .copy) .copy else .head,
            .sampled = if (p.stream.sampling) |sampling| sampling.temperature > 0 else false,
            .history = if (plans.len <= adaptive.max_streams and p.kind == .head and p.held > 0 and p.parents == null and p.branch_from == 0)
                &p.stream.request_depth.load[plans.len - 1] else null,
            .warm = p.stream.rounds > 1,
        };
        if (plans.len<=adaptive.max_streams) for(plans) |p| {
            const stats=&model.work[plans.len-1];
            if (p.kind==.head) stats.head_selected+=p.count();
            if (p.kind==.copy) { if(p.shared_copy) stats.shared_selected+=p.count() else stats.local_selected+=p.count(); }
            stats.origin_history_changes += @intFromBool(p.origin_changed);
            stats.origin_contests += @intFromBool(p.origin_contest);
            stats.origin_winner_changes += @intFromBool(p.winner_changed);
            stats.origin_copy_plans += @intFromBool(model.origin_scoring and p.kind == .copy and p.copy_scored);
            switch(p.choice) { .head=>stats.head_wins+=1, .copy=>stats.copy_wins+=1, .none=>{} }
        };
        break :block try model.allocate(a, inputs, &e.cfg.shared_costs, e.rule.overheadMs(plans.len), e.cfg.batch_rows);
    } else try alloc.allocate(a, fixed, probs, &e.cfg.shared_costs, e.rule.overheadMs(plans.len), @max(plans.len, e.cfg.batch_rows));
    for (plans, probs, counts) |*p, chances, c| {
        if (p.kind == .head) p.stream.granted = c;
        if (chances.len == 0 or c == chances.len) continue;
        if (c == 0) {
            p.kind = .none;
            p.held = 0;
            p.tokens = &.{};
            p.parents = null;
        } else if (p.parents) |parents| {
            const tree = try accept.sanitizeTree(a, p.tokens, parents, c);
            p.tokens = tree.tokens;
            p.parents = tree.parents;
        } else if (p.held > 0) {
            p.held = c;
        } else p.tokens = p.tokens[0..c];
    }
    const rows = try a.alloc(Value, plans.len);
    for (rows, plans) |*r, p| {
        const pair = try a.alloc(Value, 2);
        pair[0] = trail.str(@tagName(p.kind));
        pair[1] = trail.int(p.count());
        r.* = .{ .list = pair };
    }
    try trail.event(e, &.{ f("ev", trail.str("alloc")), f("plans", .{ .list = rows }) });
}

/// The head's chance for each held draft, else chain chances from the stream's per-depth acceptance.
pub fn nodeChances(e: *Engine, s: *Stream, n: usize) ![]const f64 {
    const a = e.arena.allocator();
    if (e.cfg.node_probabilities) {
        if (e.backend.vtable.probabilities) |chances| {
            const out = try a.alloc(f64, n);
            if (try chances(e.backend.ptr, s, out)) return out;
        }
    }
    return alloc.chainProbabilities(a, try e.rule.rates(who(s)), n);
}

/// The window's rows, parents and keyed positions, ready for the backend.
pub fn build(e: *Engine, p: Plan, early: bool) !be.Window {
    const a = e.arena.allocator();
    const rows = 1 + p.count();
    const parents = try accept.rowParents(a, rows, p.parents);
    const ds = try accept.depths(a, parents);
    const positions = try a.alloc(u64, rows);
    for (positions, ds) |*x, d| x.* = p.position + 1 + d;
    return .{ .stream = p.stream, .pending = p.stream.pending.?, .held = p.held, .tokens = p.tokens, .parents = if (p.parents != null) parents else null, .positions = positions, .early = early, .prefix_only = prefixEligible(p.kind, p.parents != null) };
}

pub fn rowParents(e: *Engine, w: be.Window) ![]const i32 {
    if (w.parents) |p| return p;
    return accept.rowParents(e.arena.allocator(), w.rows(), null);
}

/// The rows' tokens: the pending one, then the drafts (held drafts as the backend read them back).
pub fn tokens(e: *Engine, w: be.Window, o: be.Verified) ![]u32 {
    const out = try e.arena.allocator().alloc(u32, w.rows());
    out[0] = w.pending;
    @memcpy(out[1 .. 1 + w.held], o.drafts[0..w.held]);
    @memcpy(out[1 + w.held ..], w.tokens);
    return out;
}

fn prefixEligible(kind: Kind, has_tree: bool) bool {
    return !has_tree and (kind == .head or kind == .copy);
}

test "only target-verified head and copy chains allow lazy samples" {
    for ([_]Kind{ .forced, .copy, .head, .fill, .none }) |kind| {
        try std.testing.expectEqual(kind == .head or kind == .copy, prefixEligible(kind, false));
        try std.testing.expect(!prefixEligible(kind, true));
    }
}

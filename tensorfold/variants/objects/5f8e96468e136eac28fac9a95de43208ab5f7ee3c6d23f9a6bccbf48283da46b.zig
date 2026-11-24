//! Bounded, model-owned history of completed continuations. This proposes
//! tokens only: the existing target verifier remains the sole acceptance rule.
//! The owner engine thread performs all accesses; no per-token locks or I/O.
const std = @import("std");
const base = @import("proposer.zig");
const Allocator = std.mem.Allocator;
pub const document_limit = 64;
pub const token_limit = 4096;
pub const context_limit = 64;
pub const extension_limit = 64;
pub const ngram = 8;
pub const bucket_count = 16384;
pub const ways = 4;
pub const Namespace = ?[32]u8;

const Entry = struct { hash: u64 = 0, epoch: u64 = 0, doc: u16 = 0, end: u16 = 0 };
const Document = struct { tokens: []u32 = &.{}, namespace: Namespace = null, epoch: u64 = 0 };
pub const Match = struct {
    tokens: [extension_limit]u32 = undefined,
    len: usize = 0,
    matched: usize = 0,
    probes: usize = 0,
};

fn sameNamespace(a: Namespace, b: Namespace) bool {
    if ((a == null) != (b == null)) return false;
    return if (a) |x| std.mem.eql(u8, &x, &b.?) else true;
}

fn key(namespace: Namespace, tail: []const u32) u64 {
    var h: u64 = 14695981039346656037;
    h = (h ^ @as(u64, @intFromBool(namespace != null))) *% 1099511628211;
    if (namespace) |ns| for (ns) |byte| { h = (h ^ byte) *% 1099511628211; };
    for (tail[tail.len - ngram ..]) |token| {
        inline for (0..4) |i| h = (h ^ ((token >> (8 * i)) & 255)) *% 1099511628211;
    }
    return h;
}

pub const Store = struct {
    gpa: Allocator,
    entries: []Entry,
    docs: [document_limit]Document = @splat(.{}),
    completed: u64 = 0,
    lookups: u64 = 0,
    hits: u64 = 0,
    stored_tokens: usize = 0,

    pub fn create(gpa: Allocator) !*Store {
        const s = try gpa.create(Store);
        errdefer gpa.destroy(s);
        const entries = try gpa.alloc(Entry, bucket_count * ways);
        @memset(entries, .{});
        s.* = .{ .gpa = gpa, .entries = entries };
        return s;
    }

    pub fn destroy(s: *Store) void {
        const gpa = s.gpa;
        for (s.docs) |doc| gpa.free(doc.tokens);
        gpa.free(s.entries);
        gpa.destroy(s);
    }

    /// Caller supplies only a successfully completed emitted reply. A short
    /// prompt suffix permits copying the beginning of a repeated response;
    /// candidates never extend into prompt-only material.
    pub fn remember(s: *Store, namespace: Namespace, prompt: []const u32, reply: []const u32) !void {
        if (reply.len == 0) return;
        const before: usize = @min(prompt.len, context_limit);
        const count = @min(reply.len, token_limit - before);
        if (before + count <= ngram) return;
        const tokens = try s.gpa.alloc(u32, before + count);
        @memcpy(tokens[0..before], prompt[prompt.len - before ..]);
        @memcpy(tokens[before..], reply[0..count]);
        const slot: usize = @intCast(s.completed % document_limit);
        s.completed += 1;
        const doc = &s.docs[slot];
        s.stored_tokens -= doc.tokens.len;
        s.gpa.free(doc.tokens);
        doc.* = .{ .tokens = tokens, .namespace = namespace, .epoch = s.completed };
        s.stored_tokens += tokens.len;
        var end = @max(before, ngram);
        while (end < tokens.len) : (end += 1) {
            const hash = key(namespace, tokens[0..end]);
            const start: usize = @intCast((hash & (bucket_count - 1)) * ways);
            var victim = start;
            for (start..start + ways) |i| {
                const e = s.entries[i];
                if (e.epoch == 0 or s.docs[e.doc].epoch != e.epoch) { victim = i; break; }
                if (e.epoch < s.entries[victim].epoch) victim = i;
            }
            s.entries[victim] = .{ .hash = hash, .epoch = doc.epoch, .doc = @intCast(slot), .end = @intCast(end) };
        }
    }

    pub fn lookup(s: *Store, namespace: Namespace, context: []const u32, max_draft: usize) Match {
        var result: Match = .{};
        if (context.len < ngram or max_draft == 0) return result;
        s.lookups += 1;
        const hash = key(namespace, context);
        const start: usize = @intCast((hash & (bucket_count - 1)) * ways);
        var newest: u64 = 0;
        for (s.entries[start..start + ways]) |e| {
            result.probes += 1;
            if (e.epoch == 0 or e.hash != hash) continue;
            const doc = s.docs[e.doc];
            if (doc.epoch != e.epoch or !sameNamespace(namespace, doc.namespace) or e.end >= doc.tokens.len) continue;
            const limit = @min(context_limit, @min(context.len, e.end));
            var matched: usize = 0;
            while (matched < limit and context[context.len - 1 - matched] == doc.tokens[e.end - 1 - matched]) : (matched += 1) {}
            if (matched < ngram or matched < result.matched or (matched == result.matched and e.epoch < newest)) continue;
            const count = @min(extension_limit, @min(max_draft, doc.tokens.len - e.end));
            @memcpy(result.tokens[0..count], doc.tokens[e.end..][0..count]);
            result.len = count;
            result.matched = matched;
            newest = e.epoch;
        }
        if (result.len > 0) s.hits += 1;
        return result;
    }
};

/// Request-owned arbitration and feedback; local lookup always remains usable.
pub const SharedLookup = struct {
    local: base.SuffixLookup,
    store: ?*Store,
    namespace: Namespace,
    out: Match = .{},
    selected_global: bool = false,
    quality: [2]@import("spec_work.zig").Quality = @splat(.{}),
    by_match: [2][3]@import("spec_work.zig").Quality = @splat(@splat(.{})),
    cached: ?struct { round:u64, width:i64, epoch:u64, length:usize, ptr:usize, local_match:i64 } = null,
    committed: ?u64 = null,
    match: i64 = 0,
    rejected: u8 = 0,
    silent: u8 = 0,
    proposed: u64 = 0,
    judged: u64 = 0,
    accepted: u64 = 0,

    pub fn matchBand(n: i64) usize {
        return if (n >= 24) 2 else if (n >= 12) 1 else 0;
    }
    fn qualityRate(s: *const SharedLookup, global: bool, n: i64, calibrated: bool) f64 {
        const index = @intFromBool(global);
        const cell = s.by_match[index][matchBand(n)];
        return if (calibrated and cell.seen >= 6) cell.rate(n) else s.quality[index].rate(n);
    }
    pub fn init(gpa: Allocator, params: base.Params, store: ?*Store, namespace: Namespace) !SharedLookup {
        return .{ .local = try base.SuffixLookup.init(gpa, params), .store = store, .namespace = namespace };
    }
    pub fn deinit(s: *SharedLookup) void { s.local.deinit(); }
    pub fn proposer(s: *SharedLookup) base.Proposer { return .{ .ptr = s, .vtable = &vtable }; }
    pub const vtable: base.Proposer.VTable = .{ .propose = proposeFn, .last_match = lastMatchFn, .observe = observeFn, .candidate=candidateFn };

    fn candidateFn(ptr:*anyopaque, context:[]const u32, q:base.Query) anyerror!base.Candidate {
        const s:*SharedLookup=@ptrCast(@alignCast(ptr));
        const epoch=if(s.store) |store| store.completed else 0;
        const reuse=if(s.cached) |c| c.round==q.round and c.width==q.width and c.epoch==epoch and
            c.length==context.len and c.ptr==@intFromPtr(context.ptr) else false;
        if (!reuse) {
            s.cached=null;
            const local=try s.local.evaluate(context,q.width,false);
            s.out=.{};
            if(s.silent==0 and q.width>0) if(s.store) |store| {
                s.out=store.lookup(s.namespace,context,@intCast(q.width));
            };
            s.cached=.{.round=q.round,.width=q.width,.epoch=epoch,.length=context.len,.ptr=@intFromPtr(context.ptr),
                .local_match=if(local.len>0) s.local.last_match else 0};
        }
        const work=@import("spec_work.zig");
        const local=s.local.out.items;
        const matched=s.cached.?.local_match;
        const local_rate=s.qualityRate(false,matched,q.calibrated);
        const global_rate=s.qualityRate(true,@intCast(s.out.matched),q.calibrated);
        const local_ok=local.len>=2 and matched>=q.minimum;
        const global_ok = s.out.len >= 2 and @as(i64, @intCast(s.out.matched)) >= q.minimum;
        const original = global_ok and (!local_ok or
            work.copyValue(global_rate, @min(q.limit, s.out.len), q.price) >
            work.copyValue(local_rate, @min(q.limit, local.len), q.price));
        var global = original;
        if (q.scorer) |sc| if (local_ok and global_ok) {
            const ln = @min(sc.limit, local.len);
            const sn = @min(sc.limit, s.out.len);
            const lv = sc.value(sc.ctx, sc.sampled, false, @intCast(matchBand(matched)), local_rate, ln, q.price);
            const sv = sc.value(sc.ctx, sc.sampled, true, @intCast(matchBand(@intCast(s.out.matched))), global_rate, sn, q.price);
            if (lv != null or sv != null) {
                const l = lv orelse work.copyValue(local_rate, ln, q.price);
                const g = sv orelse work.copyValue(global_rate, sn, q.price);
                global = if (g == l) original else g > l;
            }
        };
        var result:base.Candidate=if(global)
            .{.tokens=s.out.tokens[0..s.out.len],.matched=@intCast(s.out.matched),.rate=global_rate,.global=true}
            else .{.tokens=local,.matched=matched,.rate=local_rate,.global=false};
        result.origin_changed = global != original;
        result.local_rate=local_rate; result.shared_rate=global_rate;
        result.local_available=if(local_ok) local.len else 0;
        result.shared_available=if(s.out.matched>=q.minimum) s.out.len else 0;
        if(!q.preview) {
            s.match=result.matched; s.selected_global=global;
            if (s.committed!=q.round) {
                s.committed=q.round;
                if(s.silent>0) s.silent-=1;
                if(s.local.silent_for>0) {s.local.silent_for-=1; s.local.silenced_rounds+=1;}
                if(global and q.offer) s.proposed+=result.tokens.len else if(!global and q.offer and local.len>0) {
                    s.local.proposals+=1; s.local.proposed_tokens+=@intCast(local.len);
                }
            }
        }
        return result;
    }

    fn proposeFn(ptr: *anyopaque, context: []const u32, max_draft: i64) anyerror![]const u32 {
        const s: *SharedLookup = @ptrCast(@alignCast(ptr));
        s.cached=null;
        const local = try s.local.propose(context, max_draft);
        s.selected_global = false;
        s.match = if (local.len > 0) s.local.last_match else 0;
        if (max_draft <= 0) return local;
        if (s.silent > 0) { s.silent -= 1; return local; }
        // A strong local match needs no extra history search.
        if (local.len > 0 and s.match >= context_limit) return local;
        const store = s.store orelse return local;
        s.out = store.lookup(s.namespace, context, @intCast(max_draft));
        if (s.out.len == 0 or (local.len > 0 and s.out.matched <= s.match)) return local;
        s.selected_global = true;
        s.match = @intCast(s.out.matched);
        s.proposed += s.out.len;
        return s.out.tokens[0..s.out.len];
    }
    fn lastMatchFn(ptr: *anyopaque) i64 {
        const s: *SharedLookup = @ptrCast(@alignCast(ptr));
        return s.match;
    }
    fn observeFn(ptr: *anyopaque, proposed: i64, accepted: i64) void {
        const s: *SharedLookup = @ptrCast(@alignCast(ptr));
        s.cached=null;
        if (proposed > 0) s.by_match[@intFromBool(s.selected_global)][matchBand(s.match)].record(@intCast(proposed),@intCast(@max(0,accepted)));
        if(proposed>0) s.quality[@intFromBool(s.selected_global)].record(@intCast(proposed),@intCast(@max(0,accepted)));
        if (!s.selected_global) { s.local.observe(proposed, accepted) catch {}; return; }
        if (proposed <= 0) return;
        s.judged += @intCast(proposed);
        s.accepted += @intCast(@max(0, accepted));
        if (accepted > 0) s.rejected = 0 else s.rejected += 1;
        if (s.rejected >= 4) { s.silent = 16; s.rejected = 0; }
    }
};

test "COPY uses completed output across requests but never crosses namespace" {
    const s = try Store.create(std.testing.allocator);
    defer s.destroy();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    try s.remember(null, &prompt, &.{ 10, 11, 12, 13 });
    const found = s.lookup(null, &prompt, 3);
    try std.testing.expectEqualSlices(u32, &.{ 10, 11, 12 }, found.tokens[0..found.len]);
    try std.testing.expectEqual(@as(usize, 9), found.matched);
    try std.testing.expectEqual(@as(usize, 0), s.lookup(@splat(0), &prompt, 3).len);
    const other: [32]u8 = @splat(42);
    try s.remember(other, &prompt, &.{ 77, 78 });
    const named = s.lookup(other, &prompt, 3);
    try std.testing.expectEqualSlices(u32, &.{ 77, 78 }, named.tokens[0..named.len]);
    try std.testing.expect(found.probes <= ways);
}

test "COPY bounds retention and invalidates overwritten document entries" {
    const s = try Store.create(std.testing.allocator);
    defer s.destroy();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try s.remember(null, &prompt, &.{9});
    for (0..document_limit) |i| {
        const different = [_]u32{ 100, 101, 102, 103, 104, 105, 106, @intCast(200 + i) };
        try s.remember(null, &different, &.{999});
    }
    try std.testing.expectEqual(@as(usize, 0), s.lookup(null, &prompt, 8).len);
    var long: [token_limit + 100]u32 = undefined;
    for (&long, 0..) |*t, i| t.* = @intCast(i);
    for (0..document_limit + 1) |_| try s.remember(null, &prompt, &long);
    try std.testing.expect(s.stored_tokens <= document_limit * token_limit);
    try std.testing.expectEqual(bucket_count * ways, s.entries.len);
}

test "COPY shared proposer falls back and suppresses repeated rejected history" {
    const gpa = std.testing.allocator;
    const history = try Store.create(gpa);
    defer history.destroy();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    try history.remember(null, &prompt, &.{ 10, 11 });
    var shared = try SharedLookup.init(gpa, .{ .min_match = 4 }, history, null);
    defer shared.deinit();
    const p = shared.proposer();
    for (0..4) |_| {
        try std.testing.expectEqualSlices(u32, &.{ 10, 11 }, try p.propose(&prompt, 2));
        p.observe(2, 0);
    }
    try std.testing.expectEqual(@as(usize, 0), (try p.propose(&prompt, 2)).len);
    const local = [_]u32{ 50, 51, 52, 53, 54, 99, 50, 51, 52, 53 };
    try std.testing.expectEqualSlices(u32, &.{ 54, 99 }, try p.propose(&local, 2));
    try std.testing.expect(!shared.selected_global);
}

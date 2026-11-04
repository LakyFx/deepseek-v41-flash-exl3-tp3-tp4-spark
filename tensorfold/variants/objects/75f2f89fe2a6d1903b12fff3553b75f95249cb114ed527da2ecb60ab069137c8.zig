//! imgcache: LaneHost with the prompt cache and prompts holding images (the fake backend ignores the pictures; the
//! cache must not). Same picture: the next turn resumes; another picture at the same place: the cache stops at it.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const LaneHost = api.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;

test "image prompts resume on the same picture and stop at a different one" {
    try run();
}

fn run() !void {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            f.drop(saved);
        }
    };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    store.diagnostics = true;
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    const T: u32 = 50; // the image token
    // turn 1: text, a 4-position picture, more text
    const t1 = [_]u32{ 3, 1, 4, T, T, T, T, 1, 5, 9, 2, 6, 5, 3 };
    const red = [_]api.Image{.{ .at = 3, .tokens = 4, .bytes = "red-circle" }};
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .images = &red, .max_tokens = 6, .history_len = 12 };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    try std.testing.expectEqual(@as(?u32, 0), b1.cached);
    try std.testing.expect(store.entries.items.len >= 1); // upstream kept nothing for an image prompt
    try std.testing.expectEqual(@as(u64, 0), store.reasons.image_bypass);
    // every kept key holds the picture's key ids, never the bare image token
    for (store.entries.items) |en| for (en.tokens[3..@min(7, en.tokens.len)]) |k| try std.testing.expect(k != T and k & 0x8000_0000 != 0);

    // turn 2, same picture: the history, the reply, a tool result, a new generation prompt
    var t2: std.ArrayList(u32) = .empty;
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items[0..4]);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    const red_again = [_]api.Image{.{ .at = 3, .tokens = 4, .bytes = "red-circle" }}; // another buffer, same payload
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .images = &red_again, .max_tokens = 8, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expect(b2.cached.? >= 7); // resumed past the picture
    var history: std.ArrayList(u32) = .empty; // the reply after a restore equals a whole pass's
    defer history.deinit(gpa);
    try history.appendSlice(gpa, t2.items);
    for (b2.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }

    // the same tokens with another picture at the same place: never resumed past the picture's first position
    const blue = [_]api.Image{.{ .at = 3, .tokens = 4, .bytes = "blue-square" }};
    var b3: Box = .{};
    defer b3.tokens.deinit(gpa);
    const r3: Request = .{ .prompt = t2.items, .images = &blue, .max_tokens = 8, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(3, &r3, .{ .ctx = &b3, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b3.wait());
    try std.testing.expect(b3.cached.? <= 3);
    try std.testing.expectEqualSlices(u32, b2.tokens.items, b3.tokens.items); // the fake reads tokens only

    // and a text prompt with the bare image token ids at those positions never matches an image-keyed state
    var b4: Box = .{};
    defer b4.tokens.deinit(gpa);
    const r4: Request = .{ .prompt = t2.items, .max_tokens = 8, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(4, &r4, .{ .ctx = &b4, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b4.wait());
    try std.testing.expect(b4.cached.? <= 3);

    // LakyFx integration guard: an image-key match never crosses client namespaces.
    var isolated: Box = .{};
    defer isolated.tokens.deinit(gpa);
    const other_client: Request = .{ .prompt = t2.items, .images = &red_again,
        .max_tokens = 8, .history_len = @intCast(t2.items.len - 2),
        .cache_namespace = @as([32]u8, @splat(0x5a)) };
    try e.submit(5, &other_client, .{ .ctx = &isolated, .event = Box.event });
    try std.testing.expectEqual(Reason.length, isolated.wait());
    try std.testing.expectEqual(@as(?u32, 0), isolated.cached);
}

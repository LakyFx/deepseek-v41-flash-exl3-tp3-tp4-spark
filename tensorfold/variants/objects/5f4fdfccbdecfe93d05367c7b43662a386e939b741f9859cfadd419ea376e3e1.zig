//! Prompt ids from a request, as ``prepare_prompt`` and ``render_prompt_ids`` make them.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const messages_mod = @import("messages.zig");
const model_text = @import("model_text.zig");
const chat = @import("chat.zig");
const Server = @import("server.zig").Server;
const api = @import("engine_api");
const Value = json.Value;
const Cx = errors.Cx;

pub const Rendered = struct { ids: []const u32, history_len: usize = 0, images: []const api.Image = &.{} };

pub const isTitle = messages_mod.isTitleRequest;

/// ``render_prompt_ids``: messages normalized for this template, rendered, and a dangling ``<think>`` closed.
pub fn renderIds(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, thinking: bool, effort: ?[]const u8, generation: bool) errors.Refused![]const u32 {
    var normalized = try messages_mod.toolArguments(cx, try messages_mod.normalize(cx, messages, srv.late_system, srv.needs_user_after_tool));
    if (srv.ds41_tool_separator) normalized = try messages_mod.ds41ToolSeparator(cx, normalized);
    var problem: []const u8 = "";
    const options: model_text.RenderOptions = .{
        .tools = if (tools.len > 0) Value{ .array = @constCast(tools) } else null,
        .add_generation_prompt = generation,
        .enable_thinking = thinking,
        .reasoning_effort = if (thinking) effort else null,
        .reasoning_effort_value = if (thinking) srv.effortValue(effort) else null,
    };
    var ids = srv.text.renderIds(cx.a, normalized, options, &problem) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Template => return cx.fail(.server, "{s}", .{problem}),
    };
    if (!thinking and generation and ids.len > 0) {
        const last = try srv.text.decode(cx.a, ids[ids.len - 1 ..]);
        if (std.mem.eql(u8, @import("reply_text.zig").pyStrip(last), "<think>")) {
            const close = srv.text.encode(cx.a, "</think>", false) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Template => &[_]u32{},
            };
            if (close.len == 1) ids = try std.mem.concat(cx.a, u32, &.{ ids, close });
        }
    }
    return ids;
}

/// A request's prompt: raw text or ids for a completion, else the chat template's, with its history length.
pub fn prepare(srv: *Server, cx: *Cx, input: chat.Input, thinking: bool, effort: ?[]const u8) errors.Refused!Rendered {
    if (input.prompt) |p| switch (p) {
        .text => |t| return .{ .ids = srv.text.encode(cx.a, t, true) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Template => return cx.fail(.server, "the tokenizer cannot encode this prompt", .{}),
        } },
        .ids => |ids| return .{ .ids = ids },
    };
    const prompt = try renderIds(srv, cx, input.messages, input.tools, thinking, effort, true);
    const history = try renderIds(srv, cx, input.messages, input.tools, thinking, effort, false);
    var history_len: usize = 0;
    if (history.len > 0 and history.len < prompt.len and std.mem.eql(u32, prompt[0..history.len], history)) history_len = history.len;
    if (history_len == 0 and prompt.len > 1 and std.mem.eql(u32, history, prompt)) history_len = prompt.len - 1; // a template with no generation suffix
    if (input.images.len > 0) {
        // imgcache: the cache keys a prompt with images by its payloads (core/image_key.zig), so its history is
        // worth keeping too: the length moves from placeholder positions to the expanded spans' positions
        var r = try spans(srv, cx, prompt, input.images);
        r.history_len = expandedLen(prompt, srv.info.vision.?.token, r.images, history_len);
        return r;
    }
    return .{ .ids = prompt, .history_len = history_len };
}

/// Where position `raw` of the placeholder-rendered `ids` lands once each placeholder is its image's span.
pub fn expandedLen(ids: []const u32, token: u32, images: []const api.Image, raw: usize) usize {
    if (raw == 0) return 0;
    var k: usize = 0;
    for (ids[0..@min(raw, ids.len)]) |id| {
        if (id == token) k += 1;
    }
    if (k == 0) return raw;
    const last = images[k - 1];
    // the k-th placeholder's raw position: raw ids after it stay one for one
    var seen: usize = 0;
    var at_raw: usize = 0;
    for (ids, 0..) |id, i| if (id == token) {
        seen += 1;
        if (seen == k) {
            at_raw = i;
            break;
        }
    };
    return @as(usize, last.at) + last.tokens + (raw - at_raw - 1);
}

test "expandedLen maps placeholder positions to span positions" {
    const T: u32 = 9;
    const ids = [_]u32{ 1, T, 2, 3, T, 4 };
    const imgs = [_]api.Image{ .{ .at = 1, .tokens = 3, .bytes = "" }, .{ .at = 6, .tokens = 2, .bytes = "" } };
    // expanded: 1 T T T 2 3 T T 4
    try std.testing.expectEqual(@as(usize, 0), expandedLen(&ids, T, &imgs, 0));
    try std.testing.expectEqual(@as(usize, 1), expandedLen(&ids, T, &imgs, 1));
    try std.testing.expectEqual(@as(usize, 4), expandedLen(&ids, T, &imgs, 2));
    try std.testing.expectEqual(@as(usize, 6), expandedLen(&ids, T, &imgs, 4));
    try std.testing.expectEqual(@as(usize, 8), expandedLen(&ids, T, &imgs, 5));
    try std.testing.expectEqual(@as(usize, 9), expandedLen(&ids, T, &imgs, 6));
}

/// Each image placeholder of the rendered prompt as its image's span (the vision family's prepare: the span's length and
/// the payload the engine reads), the images in prompt order.
fn spans(srv: *Server, cx: *Cx, ids: []const u32, images: []const []const u8) errors.Refused!Rendered {
    const vision = srv.info.vision orelse return cx.refuse("this server accepts text only; image, audio and video inputs are unsupported");
    var out: std.ArrayList(u32) = .empty;
    var placed: std.ArrayList(api.Image) = .empty;
    for (ids) |id| {
        if (id != vision.token) {
            try out.append(cx.a, id);
            continue;
        }
        const k = placed.items.len;
        if (k >= images.len) return cx.refuse("image input: the prompt holds more image placeholders than images");
        var problem: []const u8 = "";
        const prep = vision.prepare(vision.ctx, cx.a, images[k], &problem) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.BadImage => return cx.refuse(problem),
        };
        try placed.append(cx.a, .{ .at = @intCast(out.items.len), .tokens = prep.tokens, .bytes = prep.payload });
        try out.appendNTimes(cx.a, vision.token, prep.tokens);
    }
    if (placed.items.len != images.len) return cx.refuse("image input: the prompt holds fewer image placeholders than images");
    return .{ .ids = out.items, .images = placed.items };
}

/// A reusable system prefix, found with a probe in place of the first user message; zero below 512 tokens.
pub fn systemPrefixLen(srv: *Server, cx: *Cx, messages: Value, tools: []const Value, prompt_ids: []const u32, thinking: bool, effort: ?[]const u8) usize {
    if (messages != .array) return 0;
    const first_user = for (messages.array, 0..) |m, i| {
        const role = m.get("role") orelse continue;
        if (role == .string and std.mem.eql(u8, role.string, "user")) break i;
    } else return 0;
    const probe = cx.a.alloc(Value, first_user + 1) catch return 0;
    @memcpy(probe[0..first_user], messages.array[0..first_user]);
    const user = json.newObject(cx.a) catch return 0;
    user.put(cx.a, "role", .{ .string = "user" }) catch return 0;
    user.put(cx.a, "content", .{ .string = "\u{2063}probe" }) catch return 0;
    probe[first_user] = .{ .object = user };
    var scratch: Cx = .{ .a = cx.a };
    const other = renderIds(srv, &scratch, .{ .array = probe }, tools, thinking, effort, true) catch return 0;
    var shared: usize = 0;
    while (shared < @min(prompt_ids.len, other.len) and prompt_ids[shared] == other[shared]) shared += 1;
    return if (shared >= 512) shared else 0;
}

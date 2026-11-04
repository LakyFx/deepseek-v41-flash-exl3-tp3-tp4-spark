//! imgcache (2026-11-03): prompt-cache keys for prompts with images.
//!
//! A kept state is keyed by its prompt's tokens, and every position of an image's span holds the same image token id,
//! so two prompts with different pictures at the same place would share a key. Upstream therefore never looked up or
//! kept a state for a prompt with images (lane_host: "a kept state is keyed by tokens: an image's are its
//! placeholder's"), and an agent that once read an image re-ran its whole prompt every step.
//!
//! Here the key is the prompt with each span's positions replaced by ids derived from the image's payload (what the
//! backend's rows come from), all with the top bit set: no vocabulary id has it, so a key id never equals a text token,
//! and two different payloads diverge at their span's first position (a 62-bit hash split over the span), where a
//! prefix match then stops. Only the cache sees these ids; the stream and the backend keep the real prompt.
const std = @import("std");
const Image = @import("lanes").stream.Image;

pub const tag: u32 = 0x8000_0000;

/// The payload's digest; the span length is folded in so a re-prepared image of another size never matches.
pub fn digest(img: Image) u64 {
    var h = std.hash.Wyhash.init(0x696d_6763_6163_6865); // "imgcache"
    h.update(img.bytes);
    h.update(std.mem.asBytes(&img.tokens));
    h.update(std.mem.asBytes(&@as(u8, @intFromBool(img.rows))));
    return h.final();
}

fn mix(x: u64) u64 {
    var z = x +% 0x9E37_79B9_7F4A_7C15;
    z = (z ^ (z >> 30)) *% 0xBF58_476D_1CE4_E5B9;
    z = (z ^ (z >> 27)) *% 0x94D0_49BB_1331_11EB;
    return z ^ (z >> 31);
}

/// The key id of position `j` of a span whose payload digest is `d`: position 0 carries the digest's low 31 bits,
/// position 1 its next 31 (so the first two positions separate payloads by 62 bits), the rest a mix of both.
pub fn id(d: u64, j: u32) u32 {
    const bits: u32 = switch (j) {
        0 => @truncate(d & 0x7fff_ffff),
        1 => @truncate((d >> 31) & 0x7fff_ffff),
        else => @truncate(mix(d ^ (@as(u64, j) << 32)) & 0x7fff_ffff),
    };
    return tag | bits;
}

/// Overwrite each span's positions inside `key` (a copy of the prompt or of a stream's context, which begins with the
/// prompt) with its image's key ids. Spans past `key` are skipped (a keep before them).
pub fn apply(key: []u32, images: []const Image, ds: []const u64) void {
    for (images, ds) |img, d| {
        const at: usize = img.at;
        if (at >= key.len) continue;
        const end = @min(key.len, at + img.tokens);
        for (key[at..end], 0..) |*k, j| k.* = id(d, @intCast(j));
    }
}

/// A gpa-owned copy of `tokens` keyed for the cache (see apply).
pub fn make(a: std.mem.Allocator, tokens: []const u32, images: []const Image, ds: []const u64) ![]u32 {
    const key = try a.dupe(u32, tokens);
    apply(key, images, ds);
    return key;
}

/// The digests of `images`, gpa-owned.
pub fn digests(a: std.mem.Allocator, images: []const Image) ![]u64 {
    const out = try a.alloc(u64, images.len);
    for (images, out) |img, *d| d.* = digest(img);
    return out;
}

test "same payload, same key; another payload diverges at its span" {
    const a = std.testing.allocator;
    const tok: u32 = 7;
    const prompt = [_]u32{ 1, 2, 3, tok, tok, tok, tok, 9, 10 };
    const red = [_]Image{.{ .at = 3, .tokens = 4, .bytes = "red-circle" }};
    const red2 = [_]Image{.{ .at = 3, .tokens = 4, .bytes = "red-circle" }};
    const blue = [_]Image{.{ .at = 3, .tokens = 4, .bytes = "blue-square" }};
    const dr = try digests(a, &red);
    defer a.free(dr);
    const dr2 = try digests(a, &red2);
    defer a.free(dr2);
    const db = try digests(a, &blue);
    defer a.free(db);
    const kr = try make(a, &prompt, &red, dr);
    defer a.free(kr);
    const kr2 = try make(a, &prompt, &red2, dr2);
    defer a.free(kr2);
    const kb = try make(a, &prompt, &blue, db);
    defer a.free(kb);
    try std.testing.expectEqualSlices(u32, kr, kr2);
    try std.testing.expectEqualSlices(u32, prompt[0..3], kr[0..3]);
    try std.testing.expectEqualSlices(u32, prompt[7..], kr[7..]);
    try std.testing.expect(kr[3] != kb[3] or kr[4] != kb[4]);
    for (kr[3..7]) |k| {
        try std.testing.expect(k & tag != 0);
        try std.testing.expect(k != tok);
    }
    try std.testing.expectEqualSlices(u32, prompt[0..3], kb[0..3]);
}

test "a longer context keeps the prompt's key and its own reply tokens" {
    const a = std.testing.allocator;
    const ctx = [_]u32{ 5, 7, 7, 6, 100, 101 }; // prompt 5 7 7 6, then two reply tokens
    const img = [_]Image{.{ .at = 1, .tokens = 2, .bytes = "x" }};
    const d = try digests(a, &img);
    defer a.free(d);
    const k = try make(a, &ctx, &img, d);
    defer a.free(k);
    try std.testing.expectEqual(@as(u32, 5), k[0]);
    try std.testing.expect(k[1] & tag != 0 and k[2] & tag != 0);
    try std.testing.expectEqualSlices(u32, &.{ 6, 100, 101 }, k[3..]);
    // a keep that stops inside the span keys only what it covers
    var short = [_]u32{ 5, 7 };
    apply(&short, &img, d);
    try std.testing.expectEqual(k[1], short[1]);
}

test "span size is part of the digest" {
    const a = Image{ .at = 0, .tokens = 4, .bytes = "same" };
    const b = Image{ .at = 0, .tokens = 5, .bytes = "same" };
    try std.testing.expect(digest(a) != digest(b));
}

// Additional LakyFx regressions for the attributed hteak issue #2 patch.
test "image keys preserve text and distinguish the changed second image and order" {
    const a = std.testing.allocator;
    const prompt = [_]u32{ 1, 7, 7, 2, 7, 7, 3 };
    const original = [_]Image{
        .{ .at = 1, .tokens = 2, .bytes = "one" },
        .{ .at = 4, .tokens = 2, .bytes = "two" },
    };
    var changed = original;
    changed[1].bytes = "different";
    var reordered = original;
    reordered[0].bytes = "two";
    reordered[1].bytes = "one";
    const d = try digests(a, &original);
    defer a.free(d);
    const dc = try digests(a, &changed);
    defer a.free(dc);
    const dr = try digests(a, &reordered);
    defer a.free(dr);
    const k = try make(a, &prompt, &original, d);
    defer a.free(k);
    const kc = try make(a, &prompt, &changed, dc);
    defer a.free(kc);
    const kr = try make(a, &prompt, &reordered, dr);
    defer a.free(kr);
    try std.testing.expectEqualSlices(u32, k[0..4], kc[0..4]);
    try std.testing.expect(!std.mem.eql(u32, k[4..6], kc[4..6]));
    try std.testing.expect(!std.mem.eql(u32, k[1..3], kr[1..3]));
    try std.testing.expectEqual(@as(u32, 1), k[0]);
    try std.testing.expectEqual(@as(u32, 2), k[3]);
    try std.testing.expectEqual(@as(u32, 3), k[6]);
    const expected = [_]u32{ 1, 7, 7, 2, 7, 7, 3 };
    try std.testing.expectEqualSlices(u32, &expected, &prompt);
    // A checkpoint before the second image must retain only its covered prefix.
    var checkpoint = [_]u32{ 1, 7, 7, 2 };
    apply(&checkpoint, &original, d);
    try std.testing.expectEqualSlices(u32, k[0..4], &checkpoint);
}

test "prepared payload and already embedded rows have different identities" {
    const raw = Image{ .at = 0, .tokens = 2, .bytes = "same", .rows = false };
    const rows = Image{ .at = 0, .tokens = 2, .bytes = "same", .rows = true };
    try std.testing.expect(digest(raw) != digest(rows));
}

fn allocationCase(a: std.mem.Allocator) !void {
    const images = [_]Image{.{ .at = 1, .tokens = 2, .bytes = "pixels" }};
    const ds = try digests(a, &images);
    defer a.free(ds);
    const key = try make(a, &.{ 1, 7, 7, 2 }, &images, ds);
    defer a.free(key);
}

test "image key construction cleans up every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCase, .{});
}

test "text-only cache key remains byte-identical" {
    const a = std.testing.allocator;
    const key = try make(a, &.{ 8, 9, 10 }, &.{}, &.{});
    defer a.free(key);
    try std.testing.expectEqualSlices(u32, &.{ 8, 9, 10 }, key);
}

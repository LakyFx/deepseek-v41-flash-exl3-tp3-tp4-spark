//! Adapted from TensorFold PR1728's immutable-append contract, Apache-2.0.
//! Our older template values have no provenance spans; no template/token rules change.
const std = @import("std");

pub fn Buffer(comptime T: type) type {
    return struct {
        const Self = @This();
        storage: []T = &.{},
        len: usize = 0,

        pub fn current(b: *const Self, slice: []const T) bool {
            return slice.ptr == b.storage.ptr and slice.len == b.len;
        }

        /// Never realloc/free: old slices, aliases and self-concats stay immutable
        /// until the render arena is destroyed. Geometric growth bounds retained copies.
        pub fn append(b: *Self, arena: std.mem.Allocator, tail: []const T) ![]const T {
            const end = std.math.add(usize, b.len, tail.len) catch return error.OutOfMemory;
            if (end > b.storage.len) {
                const cap = @max(end, std.math.mul(usize, @max(b.storage.len, 16), 2) catch end);
                const fresh = try arena.alloc(T, cap);
                @memcpy(fresh[0..b.len], b.storage[0..b.len]);
                // `tail` can alias the OLD allocation: retain it until after this copy.
                @memcpy(fresh[b.len..end], tail);
                b.storage = fresh;
            } else {
                std.mem.copyForwards(T, b.storage[b.len..end], tail);
            }
            b.len = end;
            return b.storage[0..end];
        }
    };
}

test "append retains old values, self concatenation and independent forks" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var b: Buffer(u8) = .{};
    const old = try b.append(a, "hello");
    const twice = try b.append(a, old);
    try std.testing.expectEqualStrings("hello", old);
    try std.testing.expectEqualStrings("hellohello", twice);
    try std.testing.expect(!b.current(old));
    try std.testing.expect(b.current(twice));
    _ = try b.append(a, "large append that grows storage");
    try std.testing.expectEqualStrings("hellohello", twice);
}

//! Equal collective shapes, independent sessions, and ignored padding only.
const std = @import("std");

pub const Plan = struct {
    streams: usize,
    pair: usize,
    local: usize,
    first: usize,

    pub fn init(streams: usize, pair: usize) !Plan {
        if (streams < 2 or streams > 16 or pair >= 2) return error.BadDraftSplit;
        const local = (streams + 1) / 2;
        return .{ .streams = streams, .pair = pair, .local = local, .first = pair * local };
    }

    // Padding duplicates the final valid session. A draft pass may only READ
    // absorbed rings; first-use GPU checks compare every ring before/after.
    pub fn source(p: Plan, i: usize) !usize {
        if (i >= p.local) return error.BadDraftSplit;
        return @min(p.first + i, p.streams - 1);
    }
};

test "both pairs use equal collective counts and assembled valid rows preserve every session" {
    for (2..17) |streams| {
        const a = try Plan.init(streams, 0);
        const b = try Plan.init(streams, 1);
        try std.testing.expectEqual(a.local, b.local);
        try std.testing.expect(2 * a.local <= 16);
        var assembled: [16]usize = undefined;
        for ([_]Plan{ a, b }) |p| for (0..p.local) |i| {
            assembled[p.first + i] = try p.source(i);
        };
        for (0..streams) |i| try std.testing.expectEqual(i, assembled[i]);
        if (streams % 2 == 1) try std.testing.expectEqual(streams - 1, assembled[streams]);
        try std.testing.expectError(error.BadDraftSplit, a.source(a.local));
    }
    try std.testing.expectError(error.BadDraftSplit, Plan.init(1, 0));
    try std.testing.expectError(error.BadDraftSplit, Plan.init(17, 0));
    try std.testing.expectError(error.BadDraftSplit, Plan.init(2, 2));
}

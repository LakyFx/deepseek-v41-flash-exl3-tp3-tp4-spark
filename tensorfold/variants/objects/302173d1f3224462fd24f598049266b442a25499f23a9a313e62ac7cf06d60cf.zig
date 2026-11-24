const std = @import("std");
const history = @import("copy_history.zig");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");

test "COPY target verifier rejects incorrect history and preserves serial target path" {
    const gpa = std.testing.allocator;
    for ([_]?@import("sampling.zig").Sampling{ null, .{ .seed = 42, .temperature = 0.6, .top_k = 20, .top_p = 0.95 } }) |sampling| {
        var cfg = try Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true,
            .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4,
            .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32,
            .max_streams = 8, .draft_streams = true }, 16, 15);
        defer cfg.deinit(gpa);
        const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var expected: [prompt.len + 32]u32 = undefined;
        @memcpy(expected[0..prompt.len], &prompt);
        for (prompt.len..expected.len) |i| expected[i] = fake.next(expected[0..i], sampling, i);
        var stored = expected[prompt.len..].*;
        stored[6] = (stored[6] + 1) % fake.vocab;
        const store = try history.Store.create(gpa);
        defer store.destroy();
        try store.remember(null, &prompt, &stored);
        var p = try history.SharedLookup.init(gpa, .{ .min_match = 4 }, store, null);
        defer p.deinit();
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: fake.FixedClock = .{};
        var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer engine.deinit();
        var stream = try sm.Stream.init(gpa, .{ .id = "copy-check", .prompt = &prompt,
            .max_new = 32, .eos = &.{999}, .sampling = sampling, .drafts = true,
            .proposer = p.proposer() });
        defer stream.deinit(gpa);
        try engine.addStream(&stream);
        var steps: usize = 0;
        while (engine.activeCount() > 0 and steps < 64) : (steps += 1) try engine.step();
        try std.testing.expect(stream.finished);
        try std.testing.expectEqualSlices(u32, expected[prompt.len..], stream.emitted());
        try std.testing.expect(p.judged > 0 and p.accepted < p.judged);
    }
}

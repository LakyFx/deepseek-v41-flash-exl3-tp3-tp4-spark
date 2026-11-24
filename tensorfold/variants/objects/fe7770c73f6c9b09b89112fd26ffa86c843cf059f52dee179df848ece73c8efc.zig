//! Per-request evidence for extending a verified head prefix of length three or four.
//! Owned by the stream. No prompt content or state survives request teardown.

const work = @import("spec_work.zig");

const Position = struct {
    streak: u8 = 0,
    reached: u8 = 0,
    hits: u8 = 0,
    shorter: work.Recent = .{},
    comparison_band: ?usize = null,
    samples: u8 = 0,
    extra_kept: u8 = 0,
    emitted_sum: f64 = 0,
    reference_sum: f64 = 0,
    last_feedback: u64 = 0,
    speed_votes: i8 = 0,
    last_prefix: u64 = 0,
    last_reached: u64 = 0,
    retry_after: u64 = 0,
    support_until: u64 = 0,
};

pub const Bucket = struct {
    positions: [2]Position = @splat(.{}),
    last_observed: u64 = 0,

    pub fn observe(b: *Bucket, offered: usize, kept: usize, round: u64) void {
        if (round -| b.last_observed > 256) b.* = .{};
        b.last_observed = round;
        for (&b.positions, 3..) |*p, prefix| {
            if (round -| p.last_prefix > 32 or kept < @min(offered, prefix)) p.streak = 0;
            if (kept >= prefix) {
                p.streak +|= 1;
                p.last_prefix = round;
            }
            // An earlier mismatch leaves the conditional outcome at this depth unknown.
            if (offered <= prefix or kept < prefix) continue;
            if (round -| p.last_reached > 256) {
                p.reached = 0;
                p.hits = 0;
            }
            p.last_reached = round;
            if (p.reached == 16) {
                p.reached /= 2;
                p.hits /= 2;
            }
            p.reached += 1;
            p.hits += @intFromBool(kept > prefix);
        }
    }

    pub fn adjust(b: *const Bucket, depth: usize, prior: f64, round: u64) f64 {
        if (depth < 3 or depth > 4 or round -| b.last_observed > 256) return prior;
        const p = &b.positions[depth - 3];
        if (p.reached < 3 or round -| p.last_reached > 256) return prior;
        const learned = (8 * prior + @as(f64, @floatFromInt(p.hits))) / @as(f64, @floatFromInt(8 + p.reached));
        // Positive corrections need measured whole-batch throughput evidence.
        return if (p.streak >= 2 and round -| p.last_prefix <= 32 and p.speed_votes > 0 and
            round >= p.retry_after and round <= p.support_until)
            learned
        else
            @min(learned, prior);
    }

    pub fn eligible(b: *const Bucket, depth: usize, round: u64) bool {
        if (depth < 3 or depth > 4 or round -| b.last_observed > 32) return false;
        const p = &b.positions[depth - 3];
        return p.streak >= 3 and round -| p.last_prefix <= 32 and round >= p.retry_after;
    }

    pub fn selectBand(b: *Bucket, band: usize) void {
        for (&b.positions) |*p| {
            if (p.comparison_band != band) {
                p.shorter = .{};
                p.samples = 0;
                p.extra_kept = 0;
                p.emitted_sum = 0;
                p.reference_sum = 0;
                p.speed_votes = 0;
                p.support_until = 0;
                p.retry_after = 0;
                p.comparison_band = band;
            }
        }
    }

    // Compare adjacent grants on this request at the same load and context band.
    pub fn ordinary(b: *Bucket, granted: usize, kept: usize, emitted: usize, ms: f64, round: u64, band: usize) void {
        b.selectBand(band);
        for (&b.positions, 3..) |*p, depth| {
            if (granted == depth) p.shorter.add(emitted, ms, round);
            if (granted == depth + 1 and round >= p.retry_after) {
                if (p.shorter.price(round)) |baseline|
                    _ = b.feedback(depth, kept > depth, @as(f64, @floatFromInt(emitted)) / ms, baseline, ms, round);
            }
        }
    }

    pub fn feedback(b: *Bucket, depth: usize, kept: bool, delivered_rate: f64, baseline_rate: f64, ms: f64, round: u64) bool {
        const p = &b.positions[depth - 3];
        const slow = delivered_rate < 0.98 * baseline_rate;
        if (round -| p.last_feedback > 64) {
            p.samples = 0;
            p.extra_kept = 0;
            p.emitted_sum = 0;
            p.reference_sum = 0;
        }
        p.last_feedback = round;
        p.samples += 1;
        p.extra_kept += @intFromBool(kept);
        p.emitted_sum += delivered_rate * ms;
        p.reference_sum += baseline_rate * ms;
        if (slow or !kept) p.retry_after = round +| 8;
        if (p.samples == 3) {
            if (p.extra_kept > 0 and p.emitted_sum >= p.reference_sum) {
                p.speed_votes = @min(2, p.speed_votes + 1);
                p.support_until = round +| 128;
            } else if (p.extra_kept == 0 or p.emitted_sum < 0.98 * p.reference_sum) {
                p.speed_votes = @max(-2, p.speed_votes - 1);
                p.retry_after = round +| @as(u64, if (p.speed_votes <= -2) 128 else 32);
            }
            p.samples = 0;
            p.extra_kept = 0;
            p.emitted_sum = 0;
            p.reference_sum = 0;
        }
        return slow;
    }
};

pub const History = struct {
    load: [16]Bucket = @splat(.{}),
};

pub const ProbeStats = struct {
    trials: u64 = 0,
    swapped: u64 = 0,
    kept: u64 = 0,
    timed: u64 = 0,
    slow: u64 = 0,
    emitted: u64 = 0,
    elapsed_us: u64 = 0,
    reference_millitokens: u64 = 0,
};
pub const ByLoad = [16][2]ProbeStats;

pub const Probe = struct {
    stream: usize,
    depth: usize, // zero-based index of the extra token
    swapped: bool,
    reference: f64,
    kept: ?bool = null,
};

/// Try an extra row and, when possible, a transfer from the weakest optional head row.
/// Price the actual new batch context (including graph boundaries) in both cases.
pub fn choose(c: anytype, inputs: anytype, probabilities: anytype, grants: []u32, baseline: anytype, over: f64, sampled: bool, cap: u32, reference: f64, comptime context: anytype, comptime band: anytype) ?Probe {
    var rows: u32 = 0;
    var expected: f64 = 0;
    for (inputs, grants, 0..) |input, n, i| {
        rows += input.fixed + n;
        expected += @as(f64, @floatFromInt(input.fixed));
        for (probabilities[i][0..n]) |chance| expected += chance;
    }
    const initial = expected / (c.predictedCost(rows, inputs.len, context(inputs, grants), sampled, baseline) + over);
    var best = initial * 0.98;
    var choice: ?Probe = null;
    var donor_choice: ?usize = null;
    for (inputs, 0..) |input, i| {
        const b = input.history orelse continue;
        const depth = grants[i];
        if (depth >= input.survivals.len or !b.eligible(depth, c.rounds)) continue;
        const prefix = probabilities[i][depth - 1];
        const extra = probabilities[i][depth];
        if (extra < 0.35 or extra < 0.65 * prefix) continue;
        var donor: ?usize = null;
        for (inputs, grants, 0..) |other, n, j| {
            if (i == j or other.history == null or n == 0) continue;
            if (donor == null or probabilities[j][n - 1] < probabilities[donor.?][grants[donor.?] - 1]) donor = j;
        }
        for ([_]bool{ false, true }) |swap| {
            if ((!swap and rows == cap) or (swap and donor == null)) continue;
            const new_rows = rows + @intFromBool(!swap);
            var reward = expected + extra;
            grants[i] += 1;
            if (swap) {
                grants[donor.?] -= 1;
                reward -= probabilities[donor.?][grants[donor.?]];
            }
            const new_ctx = context(inputs, grants);
            const rate = reward / (c.predictedCost(new_rows, inputs.len, new_ctx, sampled, baseline) + over);
            grants[i] -= 1;
            if (swap) grants[donor.?] += 1;
            if (band(new_ctx) == band(context(inputs, grants)) and rate > best) {
                best = rate;
                choice = .{ .stream = i, .depth = depth, .reference = reference, .swapped = swap };
                donor_choice = if (swap) donor else null;
            }
        }
    }
    if (choice) |selected| {
        grants[selected.stream] += 1;
        if (donor_choice) |j| grants[j] -= 1;
    }
    return choice;
}

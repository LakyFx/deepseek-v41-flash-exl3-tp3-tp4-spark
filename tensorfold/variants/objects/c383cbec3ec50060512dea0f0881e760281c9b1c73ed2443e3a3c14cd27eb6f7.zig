//! Selective adaptation of TensorFold PR1646 (jayleaton), Apache-2.0.
//! Metadata only: successful CUDA calls never lock, poll or synchronize.
const std = @import("std");
const Entry = struct { driver: usize = 0, context: usize = 0, refs: usize = 0, fault: c_int = 0 };
var entries: [64]Entry = @splat(.{});
var mutex: std.atomic.Mutex = .unlocked;
threadlocal var current: usize = 0;

fn lock() void { while (!mutex.tryLock()) std.Thread.yield() catch {}; }
pub fn bind(context: usize) void { current = context; }
pub fn sticky(code: c_int) bool {
    return switch (code) { 214, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719 => true, else => false };
}
pub fn retain(driver: usize, context: usize) error{OutOfMemory}!void {
    lock(); defer mutex.unlock();
    for (&entries) |*e| if (e.refs != 0 and e.driver == driver and e.context == context) { e.refs += 1; return; };
    for (&entries) |*e| if (e.refs == 0) { e.* = .{.driver=driver,.context=context,.refs=1}; return; };
    return error.OutOfMemory;
}
pub fn release(driver: usize, context: usize) void {
    lock(); defer mutex.unlock();
    for (&entries) |*e| if (e.refs != 0 and e.driver == driver and e.context == context) {
        e.refs -= 1; if (e.refs == 0) e.* = .{}; return;
    };
}
pub fn observe(driver: usize, code: c_int) void {
    if (!sticky(code)) return;
    lock(); defer mutex.unlock();
    for (&entries) |*e| if (e.refs != 0 and e.driver == driver and (current == 0 or current == e.context)) {
        if (e.fault == 0) e.fault = code;
    };
}
pub fn fault(driver: usize, context: usize) c_int {
    lock(); defer mutex.unlock();
    for (entries) |e| if (e.refs != 0 and e.driver == driver and e.context == context) return e.fault;
    return 0;
}

test "sticky execution faults persist across success and retain; OOM and invalid capture are not fatal" {
    try retain(1, 10); defer release(1, 10);
    try retain(1, 20); defer release(1, 20);
    bind(10); defer bind(0);
    for ([_]c_int{0,2,600,701,800,801,901}) |code| observe(1,code);
    try std.testing.expectEqual(@as(c_int,0),fault(1,10));
    observe(1,700);observe(1,0);
    try retain(1,10);release(1,10);
    try std.testing.expectEqual(@as(c_int,700),fault(1,10));
    try std.testing.expectEqual(@as(c_int,0),fault(1,20));
}
test "unbound asynchronous fault withdraws every context owned by affected driver only" {
    try retain(2,30);defer release(2,30);
    try retain(3,40);defer release(3,40);
    bind(0);observe(2,719);
    try std.testing.expectEqual(@as(c_int,719),fault(2,30));
    try std.testing.expectEqual(@as(c_int,0),fault(3,40));
}

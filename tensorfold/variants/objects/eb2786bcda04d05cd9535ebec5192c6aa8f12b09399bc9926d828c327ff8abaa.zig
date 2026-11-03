//! No idle/TTFT deadline: long prefills and MAX vision never count as a stall.
//! Exit callback is owned by the existing all-rank generation recovery.
const std = @import("std");

pub const Supervisor = struct {
    io: std.Io,
    owner: *anyopaque,
    fault_fn: *const fn (*anyopaque) c_int,
    fatal_fn: *const fn (*anyopaque, c_int) void,
    stopping: std.atomic.Value(bool) = .init(false),
    notified: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    pub fn start(s: *Supervisor) !void { s.thread = try std.Thread.spawn(.{}, run, .{s}); }
    pub fn stop(s: *Supervisor) void {
        s.stopping.store(true,.release);
        if (s.thread) |t| t.join();
        s.thread = null;
    }
    pub fn check(s: *Supervisor) bool {
        const code=s.fault_fn(s.owner);
        if(code==0) return false;
        if(!s.notified.swap(true,.acq_rel)) s.fatal_fn(s.owner,code);
        return true;
    }
    fn run(s: *Supervisor) void {
        while(!s.stopping.load(.acquire)) {
            if(s.check()) return;
            std.Io.sleep(s.io,.fromMilliseconds(100),.awake) catch {};
        }
    }
};

test "fault is notified once; idle is never interpreted as a failure" {
    const Fake=struct {
        code:c_int=0,notices:usize=0,
        fn fault(p:*anyopaque) c_int {const f:*@This()=@ptrCast(@alignCast(p));return f.code;}
        fn fail(p:*anyopaque,code:c_int) void {const f:*@This()=@ptrCast(@alignCast(p));std.debug.assert(code==700);f.notices+=1;}
    };
    var fake:Fake=.{};
    var s:Supervisor=.{.io=undefined,.owner=&fake,.fault_fn=Fake.fault,.fatal_fn=Fake.fail};
    for(0..100) |_| try std.testing.expect(!s.check());
    fake.code=700;try std.testing.expect(s.check());try std.testing.expect(s.check());
    try std.testing.expectEqual(@as(usize,1),fake.notices);
    s.stop();try std.testing.expect(s.thread==null);
}

test "owned monitor thread starts, latches an observed fault and joins before owner release" {
    const Fake=struct {
        code:std.atomic.Value(c_int)=.init(0),
        notices:std.atomic.Value(usize)=.init(0),
        fn fault(p:*anyopaque) c_int {const f:*@This()=@ptrCast(@alignCast(p));return f.code.load(.acquire);}
        fn fail(p:*anyopaque,code:c_int) void {const f:*@This()=@ptrCast(@alignCast(p));std.debug.assert(code==700);_=f.notices.fetchAdd(1,.acq_rel);}
    };
    var f:Fake=.{};const io=std.Io.Threaded.global_single_threaded.io();
    var s:Supervisor=.{.io=io,.owner=&f,.fault_fn=Fake.fault,.fatal_fn=Fake.fail};
    try s.start();defer s.stop();
    f.code.store(700,.release);
    for(0..20) |_| {
        if(f.notices.load(.acquire)!=0) break;
        try std.Io.sleep(io,.fromMilliseconds(20),.awake);
    }
    s.stop();try std.testing.expectEqual(@as(usize,1),f.notices.load(.acquire));
    try std.testing.expect(s.thread==null);
}

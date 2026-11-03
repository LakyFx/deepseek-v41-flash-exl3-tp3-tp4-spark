//! Optional one-time Linux rank-weight file cache advice (TensorFold PR1868 idea).
//! Only the owned rank cache fd, after GPU upload completion. Never Engram or drop_caches.
const std=@import("std");
extern "c" fn posix_fadvise(c_int,i64,i64,c_int) c_int;
pub fn enabled() !bool {
    const raw=std.c.getenv("TF404_DROP_WEIGHT_FILE_CACHE") orelse return false;
    const value=std.mem.span(raw);
    if(std.mem.eql(u8,value,"1")) return true;
    if(std.mem.eql(u8,value,"0")) return false;
    return error.Invalid404FileCacheAdvice;
}
pub fn advise(fd:c_int,callback:*const fn(c_int,i64,i64,c_int) callconv(.c) c_int) c_int {
    return callback(fd,0,0,4); // POSIX_FADV_DONTNEED: advisory, no file/data deletion.
}
pub fn loaded(fd:c_int,rank:u32) void {
    const result=advise(fd,posix_fadvise);
    std.debug.print("tf404_weight_file_cache rank={d} advice=dontneed rc={d} gpu_weights=retained engram=untouched\n",.{rank,result});
}
test "advice touches only the supplied file once; failures remain nonfatal" {
    const Fake=struct {
        var calls:usize=0;
        fn call(fd:c_int,offset:i64,length:i64,advice:c_int) callconv(.c) c_int {
            std.debug.assert(fd==81 and offset==0 and length==0 and advice==4);calls+=1;return 95;
        }
    };
    try std.testing.expectEqual(@as(c_int,95),advise(81,Fake.call));
    try std.testing.expectEqual(@as(usize,1),Fake.calls);
}

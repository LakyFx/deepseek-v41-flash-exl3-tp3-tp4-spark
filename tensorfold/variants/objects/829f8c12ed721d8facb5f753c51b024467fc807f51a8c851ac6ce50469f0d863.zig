//! Launch identity travels in the cubin; a 32-row kernel cannot silently
//! receive the production 64-row launch. Numerical variants are test-only.
const std=@import("std");
pub const Mode=enum { control,m32_exact,m64_continuous,m32_continuous };
pub var allow_numerical=false;
pub fn marker(name:[]const u8) ?Mode {
    if(std.mem.eql(u8,name,"tf_exl3max_m32_exact")) return .m32_exact;
    if(std.mem.eql(u8,name,"tf_exl3max_m64_continuous")) return .m64_continuous;
    if(std.mem.eql(u8,name,"tf_exl3max_m32_continuous")) return .m32_continuous;
    return null;
}
pub fn gridX(mode:Mode) u32 {return if(mode==.m32_exact or mode==.m32_continuous) 2 else 1;}
pub fn numerical(mode:Mode) bool {return mode==.m64_continuous or mode==.m32_continuous;}
pub fn compare(mode:Mode,reference:[]const u8,actual:[]const u8) !void {
    if(reference.len!=actual.len or reference.len%4!=0) return error.InvalidMoeOutputSize;
    if(!numerical(mode)) {
        if(!std.mem.eql(u8,reference,actual)) return error.Exl3MaxExactMismatch;
        return;
    }
    if(!allow_numerical) return error.Exl3MaxNumericalTestNotEnabled;
    // A gross-port-error guard, never a claim about model quality. The model
    // quality probe is still required even when this guard passes.
    var norm:f64=0;var delta:f64=0;
    const r=std.mem.bytesAsSlice(f32,reference);const a=std.mem.bytesAsSlice(f32,actual);
    for(r,a) |x,y| {
        if(!std.math.isFinite(x) or !std.math.isFinite(y)) return error.Exl3MaxNonfinite;
        const xf:f64=x;const yf:f64=y;norm+=xf*xf;delta+=(xf-yf)*(xf-yf);
    }
    if(delta>=1e-4*@max(norm,1e-60)) return error.Exl3MaxNumericalPortMismatch;
}
test "cubin marker pins both M32 launches" {
    try std.testing.expectEqual(Mode.m32_exact,marker("tf_exl3max_m32_exact").?);
    try std.testing.expect(marker("tf_tp4_rot_prompt")==null);
    try std.testing.expectEqual(@as(u32,2),gridX(.m32_exact));
    try std.testing.expectEqual(@as(u32,2),gridX(.m32_continuous));
    try std.testing.expectEqual(@as(u32,1),gridX(.control));
    try std.testing.expectEqual(@as(u32,1),gridX(.m64_continuous));
}
test "numerical experiment never silently bypasses exact gate" {
    const r=[_]f32{1,2};const close=[_]f32{1.00001,2.00001};const wrong=[_]f32{1,3};
    const old=allow_numerical;defer allow_numerical=old;allow_numerical=false;
    const rb=std.mem.sliceAsBytes(&r);const cb=std.mem.sliceAsBytes(&close);
    try std.testing.expectError(error.Exl3MaxExactMismatch,compare(.m32_exact,rb,cb));
    try std.testing.expectError(error.Exl3MaxNumericalTestNotEnabled,compare(.m64_continuous,rb,cb));
    allow_numerical=true;try compare(.m64_continuous,rb,cb);
    try std.testing.expectError(error.Exl3MaxNumericalPortMismatch,compare(.m64_continuous,rb,std.mem.sliceAsBytes(&wrong)));
    const bad=[_]f32{1,std.math.nan(f32)};
    try std.testing.expectError(error.Exl3MaxNonfinite,compare(.m64_continuous,rb,std.mem.sliceAsBytes(&bad)));
}

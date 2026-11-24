//! Bounded prompt-only tiling and scratch suballocation, no CUDA dependency.
const std = @import("std");
// Keep expert groups large enough for the existing 64-row matrix kernel.
// 512-row chunks would turn too many expert groups into mostly padded tiles.
pub const tile_rows: usize = 2048;
pub const threshold: usize = 4096;
pub const Tile = struct { first: usize, rows: usize, bank: usize };

pub const Iter = struct {
    n: usize,
    at: usize = 0,
    index: usize = 0,
    pub fn next(it: *Iter) ?Tile {
        if(it.at == it.n) return null;
        const left = it.n - it.at;
        // Never route a 1..63 row tail through the different decode kernels.
        const count = if(left > tile_rows and left < tile_rows + 64)
            (left + 1) / 2 else @min(tile_rows,left);
        const t: Tile = .{ .first=it.at,.rows=count,.bank=it.index%2 };
        it.at += count; it.index += 1;
        return t;
    }
};

/// Metadata is placed AFTER both banks' xg rows in the existing full-chunk
/// xg allocation. The ordinary small-chunk path cannot reach this region.
pub const Metadata = struct {
    offsets: [2][6]usize,
    sizes: [6]usize,
    first: usize,
    end: usize,
    pub fn of(cap: usize, slots: usize, d: usize, experts: usize) !Metadata {
        if(cap < threshold+64 or slots==0 or experts==0 or d==0) return error.PipeScratchCapacity;
        const p = tile_rows * slots;
        const maxu = @min(p,experts);
        const sizes: [6]usize = .{4*maxu,4,4*experts,4*maxu*tile_rows,
            8*((p+maxu*63)/64),8*((p+maxu*31)/32)};
        var out: Metadata = .{.offsets=undefined,.sizes=sizes,.first=2*p*d*2,.end=2*p*d*2};
        for(0..2) |bank| for(sizes,0..) |size,j| {
            out.end=std.mem.alignForward(usize,out.end,256);
            out.offsets[bank][j]=out.end;
            out.end+=size;
        };
        if(out.end>cap*slots*d*2) return error.PipeScratchCapacity;
        return out;
    }
};

test "PIPE covers every row once with prompt-sized tails and alternating scratch" {
    for(threshold..16385) |n| {
        var it: Iter=.{.n=n};var end:usize=0;var index:usize=0;
        while(it.next()) |t| {
            try std.testing.expectEqual(end,t.first);
            try std.testing.expect(t.rows>=64 and t.rows<=tile_rows);
            try std.testing.expectEqual(index%2,t.bank);
            end+=t.rows;index+=1;
        }
        try std.testing.expectEqual(n,end);
    }
}

test "PIPE metadata is disjoint from both data banks and stays inside existing memory" {
    for([_]usize{4608,6144,8192}) |cap| for([_]usize{5,9}) |slots| {
        const m=try Metadata.of(cap,slots,5120,257);
        var at=2*tile_rows*slots*5120*2;
        for(m.offsets) |bank| for(bank,m.sizes) |offset,size| {
            try std.testing.expect(offset>=at and offset%256==0);
            at=offset+size;
        };
        try std.testing.expectEqual(m.end,at);
        try std.testing.expect(at<=cap*slots*5120*2);
    };
    try std.testing.expectError(error.PipeScratchCapacity,Metadata.of(threshold,9,5120,257));
}

test "PIPE enqueue dependency schedule allows exchange overlap without recycling live banks" {
    // Main: G0,G1,D0,G2,D1,...,Dlast. Communication: C0,C1,... .
    // C_i waits G_i; D_i waits C_i. A bank's G_(i+2) follows D_i.
    for(2..33) |count| {
        var g:[33]usize=undefined;var c:[33]usize=undefined;var d:[33]usize=undefined;
        var main:usize=0;var comm:usize=0;
        for(0..count) |i| {
            main+=2;g[i]=main;
            comm=@max(comm,g[i])+3;c[i]=comm;
            if(i>0) {main=@max(main,c[i-1])+2;d[i-1]=main;}
        }
        main=@max(main,c[count-1])+2;d[count-1]=main;
        for(0..count) |i| {
            try std.testing.expect(c[i]>g[i] and d[i]>c[i]);
            if(i>=2) try std.testing.expect(g[i]>d[i-2]);
        }
        try std.testing.expect(g[1]<c[0]); // genuine independent work interval
    }
}

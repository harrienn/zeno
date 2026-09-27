const std = @import("std");
const Io = std.Io;
const zeno = @import("zeno");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    try stdout_writer.print("Zeno v{d}.{d}.{d} — Jev-style structured decision engine on DiffusionGemma\n", .{
        zeno.version.major,
        zeno.version.minor,
        zeno.version.patch,
    });
    try stdout_writer.print("Default canvas length: {d} tokens | Vocab size: {d}\n", .{
        zeno.ModelConfig.canvas_length,
        zeno.ModelConfig.vocab_size,
    });

    try stdout_writer.flush();
}

test "main module compilation test" {
    try std.testing.expect(zeno.version.major == 0);
}

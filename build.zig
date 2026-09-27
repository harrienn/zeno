const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Zeno core library module
    const zeno_mod = b.addModule("zeno", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Main CLI executable
    const exe = b.addExecutable(.{
        .name = "zeno",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zeno", .module = zeno_mod },
            },
        }),
    });

    b.installArtifact(exe);

    // Run command: `zig build run`
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    const run_step = b.step("run", "Run the Zeno CLI executable");
    run_step.dependOn(&run_cmd.step);

    // Module unit tests
    const mod_tests = b.addTest(.{
        .root_module = zeno_mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    // Executable unit tests
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    // Test step: `zig build test`
    const test_step = b.step("test", "Run library and executable unit tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}

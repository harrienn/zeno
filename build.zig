const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const test_model = b.option(bool, "test-model", "Run heavy full-model forward pass tests (default: false)") orelse false;

    const build_options = b.addOptions();
    build_options.addOption(bool, "test_model", test_model);

    // Zeno core library module
    const zeno_mod = b.addModule("zeno", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    zeno_mod.addImport("build_options", build_options.createModule());

    // Main CLI executable
    const exe = b.addExecutable(.{
        .name = "zeno",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zeno", .module = zeno_mod },
                .{ .name = "build_options", .module = build_options.createModule() },
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
    const test_step = b.step("test", "Run unit tests (pass -Dtest-model=true for full 26B model evaluation)");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    // Dedicated step for full model integration tests: `zig build test-model`
    const model_options = b.addOptions();
    model_options.addOption(bool, "test_model", true);

    const model_test_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    model_test_mod.addImport("build_options", model_options.createModule());

    const model_tests = b.addTest(.{
        .root_module = model_test_mod,
    });
    const run_model_tests = b.addRunArtifact(model_tests);
    const test_model_step = b.step("test-model", "Run full-model forward pass tests on local GGUF");
    test_model_step.dependOn(&run_model_tests.step);
}

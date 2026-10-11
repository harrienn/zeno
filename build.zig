const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const enable_cuda = b.option(bool, "cuda", "Enable CUDA backend (default: false)") orelse false;
    const cuda_arch = b.option([]const u8, "cuda-arch", "CUDA target compute architecture (default: sm_86)") orelse "sm_86";
    const test_model = b.option(bool, "test-model", "Run heavy full-model forward pass tests (default: false)") orelse false;

    const build_options = b.addOptions();
    build_options.addOption(bool, "test_model", test_model);
    build_options.addOption(bool, "enable_cuda", enable_cuda);
    build_options.addOption([]const u8, "cuda_arch", cuda_arch);

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

    // Dedicated step for full model integration tests: `zig build test-model`
    const model_options = b.addOptions();
    model_options.addOption(bool, "test_model", true);
    model_options.addOption(bool, "enable_cuda", enable_cuda);
    model_options.addOption([]const u8, "cuda_arch", cuda_arch);

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

    if (enable_cuda) {
        const nvcc_path = b.findProgram(.{
            .names = &.{
                "/usr/local/cuda-13.3/bin/nvcc",
                "/usr/local/cuda/bin/nvcc",
                "nvcc",
            },
        }) orelse "nvcc";

        const cuda_bin_dir = std.fs.path.dirname(nvcc_path) orelse "/usr/local/cuda/bin";
        const cuda_root = std.fs.path.dirname(cuda_bin_dir) orelse "/usr/local/cuda";
        const cuda_lib_dir = b.fmt("{s}/lib64", .{cuda_root});

        const arch_flag = b.fmt("-arch={s}", .{cuda_arch});

        const nvcc_cmd = b.addSystemCommand(&.{
            nvcc_path,
            "-c",
            "-O3",
            "-Xcompiler",
            "-fPIC",
            arch_flag,
            "-Isrc/cuda",
            "src/cuda/zeno_cuda.cu",
            "-o",
        });
        const cuda_obj = nvcc_cmd.addOutputFileArg("zeno_cuda.o");

        zeno_mod.addObjectFile(cuda_obj);
        zeno_mod.addLibraryPath(.{ .cwd_relative = cuda_lib_dir });
        zeno_mod.linkSystemLibrary("cudart", .{});
        zeno_mod.link_libc = true;

        model_tests.root_module.addObjectFile(cuda_obj);
        model_tests.root_module.addLibraryPath(.{ .cwd_relative = cuda_lib_dir });
        model_tests.root_module.linkSystemLibrary("cudart", .{});
        model_tests.root_module.link_libc = true;
    }

    b.installArtifact(exe);

    // Run command: `zig build run`
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    const run_step = b.step("run", "Run the Zeno CLI executable");
    run_step.dependOn(&run_cmd.step);

    // Test step: `zig build test`
    const test_step = b.step("test", "Run unit tests (pass -Dtest-model=true for full 26B model evaluation)");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    const test_model_step = b.step("test-model", "Run full-model forward pass tests on local GGUF");
    test_model_step.dependOn(&run_model_tests.step);
}

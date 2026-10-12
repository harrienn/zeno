//! Zeno — System One HTTP Decision API Server.
//!
//! Implements the Jev-compatible /v1/systemone HTTP REST API contract
//! specified in specs/decision-api.md using zero external dependencies.

const std = @import("std");
const root = @import("root.zig");
const decision = @import("decision.zig");
const engine_mod = @import("engine.zig");
const ops = @import("ops.zig");

const DecisionEngine = engine_mod.DecisionEngine;
const DecisionRequest = decision.DecisionRequest;
const DecisionResponse = decision.DecisionResponse;
const SchemaType = decision.SchemaType;
const CandidateSpec = decision.CandidateSpec;
const linux = std.os.linux;

/// Wire candidate object inside request JSON schema.candidates.
pub const WireCandidate = struct {
    token: []const u8,
    label: []const u8,
};

/// Wire schema object inside request JSON.
pub const WireSchema = struct {
    type: []const u8,
    candidates: []const WireCandidate,
};

/// Wire options object inside request JSON.
pub const WireOptions = struct {
    step_cap: ?u32 = null,
    temperature: ?f32 = null,
    entropy_threshold: ?f32 = null,
    canvas_length: ?usize = null,
};

/// Full wire JSON request body for POST /v1/systemone.
pub const WireDecisionRequest = struct {
    query: []const u8,
    context: ?[]const u8 = null,
    schema: WireSchema,
    options: ?WireOptions = null,
};

/// Parses an IPv4 string "A.B.C.D" into a network-ordered u32 for sockaddr.in.
pub fn parseIp4(str: []const u8) !u32 {
    var parts: [4]u8 = undefined;
    var idx: usize = 0;
    var it = std.mem.splitScalar(u8, str, '.');
    while (it.next()) |part| {
        if (idx >= 4) return error.InvalidIp;
        parts[idx] = try std.fmt.parseInt(u8, part, 10);
        idx += 1;
    }
    if (idx != 4) return error.InvalidIp;
    return @as(u32, parts[0]) |
        (@as(u32, parts[1]) << 8) |
        (@as(u32, parts[2]) << 16) |
        (@as(u32, parts[3]) << 24);
}

/// HTTP API Server for Zeno System One Decision Engine.
pub const Server = struct {
    pub const max_active_connections: usize = 64;

    allocator: std.mem.Allocator,
    engine: *DecisionEngine,
    host: []const u8,
    port: u16,
    listen_fd: ?i32 = null,
    should_stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    active_connections: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    pub fn init(allocator: std.mem.Allocator, engine: *DecisionEngine, host: []const u8, port: u16) Server {
        return .{
            .allocator = allocator,
            .engine = engine,
            .host = host,
            .port = port,
            .listen_fd = null,
            .should_stop = std.atomic.Value(bool).init(false),
            .active_connections = std.atomic.Value(usize).init(0),
        };
    }

    pub fn deinit(self: *Server) void {
        self.stop();
    }

    pub fn stop(self: *Server) void {
        self.should_stop.store(true, .release);
        if (self.listen_fd) |fd| {
            _ = linux.close(fd);
            self.listen_fd = null;
        }

        // Wait cleanly for all active connections to drain (R21-02)
        while (self.active_connections.load(.acquire) > 0) {
            const ts = std.posix.timespec{ .sec = 0, .nsec = 10_000_000 };
            _ = std.posix.system.nanosleep(&ts, null);
        }
    }

    /// Binds and listens on the configured host:port.
    pub fn listen(self: *Server) !void {
        const rc = linux.socket(linux.AF.INET, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, 0);
        if (linux.errno(rc) != .SUCCESS) {
            return error.SocketCreationFailed;
        }
        const fd: i32 = @intCast(rc);
        errdefer _ = linux.close(fd);

        const one: i32 = 1;
        const opt_rc = linux.setsockopt(fd, linux.SOL.SOCKET, linux.SO.REUSEADDR, @ptrCast(&one), @sizeOf(i32));
        if (linux.errno(opt_rc) != .SUCCESS) {
            return error.SetSockOptFailed;
        }

        const ip_u32 = try parseIp4(self.host);
        var addr: linux.sockaddr.in = .{
            .family = linux.AF.INET,
            .port = std.mem.nativeToBig(u16, self.port),
            .addr = ip_u32,
            .zero = [8]u8{ 0, 0, 0, 0, 0, 0, 0, 0 },
        };

        const bind_rc = linux.bind(fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.in));
        if (linux.errno(bind_rc) != .SUCCESS) {
            return error.BindFailed;
        }

        if (self.port == 0) {
            var bound_addr: linux.sockaddr.in = undefined;
            var bound_len: u32 = @sizeOf(linux.sockaddr.in);
            const gs_rc = linux.getsockname(fd, @ptrCast(&bound_addr), &bound_len);
            if (linux.errno(gs_rc) == .SUCCESS) {
                self.port = std.mem.bigToNative(u16, bound_addr.port);
            }
        }

        const listen_rc = linux.listen(fd, 128);
        if (linux.errno(listen_rc) != .SUCCESS) {
            return error.ListenFailed;
        }

        self.listen_fd = fd;
    }

    /// Runs the accept and request dispatch loop.
    pub fn serve(self: *Server) !void {
        const listen_fd = self.listen_fd orelse return error.NotListening;

        while (!self.should_stop.load(.acquire)) {
            const client_rc = linux.accept4(listen_fd, null, null, linux.SOCK.CLOEXEC);
            const err = linux.errno(client_rc);
            if (err != .SUCCESS) {
                if (self.should_stop.load(.acquire)) break;
                // Transient interrupt or connection reset
                if (err == .INTR or err == .CONNABORTED or err == .AGAIN) continue;
                return error.AcceptFailed;
            }

            const client_fd: i32 = @intCast(client_rc);

            if (self.should_stop.load(.acquire)) {
                _ = linux.close(client_fd);
                break;
            }

            // Atomically reserve a connection slot before spawning a worker (R21-01)
            var cur_conns = self.active_connections.load(.monotonic);
            const reserved = while (true) {
                if (cur_conns >= max_active_connections) break false;
                if (self.active_connections.cmpxchgWeak(cur_conns, cur_conns + 1, .acquire, .monotonic)) |actual| {
                    cur_conns = actual;
                } else {
                    break true;
                }
            };

            if (!reserved) {
                self.sendError(client_fd, 503, "service_unavailable", "Server connection capacity reached");
                _ = linux.shutdown(client_fd, linux.SHUT.WR);
                _ = linux.close(client_fd);
                continue;
            }

            // Configure socket read/write timeouts (15s) to prevent slowloris / stalled clients (R6-02)
            const tv = linux.timeval{ .sec = 15, .usec = 0 };
            _ = linux.setsockopt(client_fd, linux.SOL.SOCKET, linux.SO.RCVTIMEO, @ptrCast(&tv), @sizeOf(linux.timeval));
            _ = linux.setsockopt(client_fd, linux.SOL.SOCKET, linux.SO.SNDTIMEO, @ptrCast(&tv), @sizeOf(linux.timeval));

            const thread = std.Thread.spawn(.{}, workerConnectionHandler, .{ self, client_fd }) catch {
                _ = self.active_connections.fetchSub(1, .release);
                self.sendError(client_fd, 503, "service_unavailable", "Server busy: unable to spawn connection worker");
                _ = linux.shutdown(client_fd, linux.SHUT.WR);
                _ = linux.close(client_fd);
                continue;
            };
            thread.detach();
        }
    }

    fn workerConnectionHandler(self: *Server, client_fd: i32) void {
        defer _ = self.active_connections.fetchSub(1, .release);
        self.handleClient(client_fd);
    }

    const ReadResult = enum {
        ok,
        eof,
        idle_timeout,
        deadline_timeout,
        read_error,
    };

    fn getMonotonicMs() i64 {
        var ts: linux.timespec = undefined;
        _ = linux.clock_gettime(.MONOTONIC, &ts);
        return @as(i64, ts.sec) * 1000 + @divTrunc(ts.nsec, 1_000_000);
    }

    /// Returns the remaining deadline in milliseconds (<= 30s), or null if the 30s absolute deadline has expired (R9-01).
    fn getRemainingDeadlineMs(start_ms: i64) ?i64 {
        const elapsed_ms = getMonotonicMs() - start_ms;
        if (elapsed_ms >= 30_000) return null;
        return 30_000 - elapsed_ms;
    }

    /// Reads up to dest.len bytes with strict idle timeout (15s) and capped absolute deadline (30s) (R8-01, R9-01, R21-02).
    fn readChunkWithDeadline(
        self: ?*Server,
        client_fd: i32,
        dest: []u8,
        start_ms: i64,
        out_read: *usize,
    ) ReadResult {
        var pfd = [1]linux.pollfd{.{
            .fd = client_fd,
            .events = linux.POLL.IN,
            .revents = 0,
        }};

        const poll_start_ms = getMonotonicMs();

        while (true) {
            if (self) |s| {
                if (s.should_stop.load(.acquire)) return .read_error;
            }

            const remaining_ms = getRemainingDeadlineMs(start_ms) orelse return .deadline_timeout;
            // Poll in bounded slices of at most 250ms to check should_stop promptly (R21-02)
            const wait_ms: i32 = @intCast(@min(@as(i64, 250), @min(@as(i64, 15_000), remaining_ms)));

            const p_rc = linux.poll(&pfd, 1, wait_ms);
            const p_err = linux.errno(p_rc);
            if (p_err == .INTR) continue; // Recompute monotonic time and remaining deadline (R9-01)
            if (p_err != .SUCCESS) return .read_error;
            if (p_rc == 0) {
                // Poll slice timed out without readable data
                if (self) |s| {
                    if (s.should_stop.load(.acquire)) return .read_error;
                }
                if (getRemainingDeadlineMs(start_ms) == null) {
                    return .deadline_timeout;
                }
                if (getMonotonicMs() - poll_start_ms >= 15_000) {
                    return .idle_timeout;
                }
                continue;
            }
            break;
        }

        while (true) {
            if (getRemainingDeadlineMs(start_ms) == null) {
                return .deadline_timeout;
            }

            const r_rc = linux.read(client_fd, dest.ptr, dest.len);
            const r_err = linux.errno(r_rc);
            if (r_err == .INTR) continue; // Recheck deadline after interruption (R9-01)
            if (r_err != .SUCCESS) {
                if (r_err == .AGAIN) {
                    if (getRemainingDeadlineMs(start_ms) == null) return .deadline_timeout;
                    return .idle_timeout;
                }
                return .read_error;
            }
            if (r_rc == 0) return .eof;
            out_read.* = r_rc;
            return .ok;
        }
    }

    /// Processes a single client connection.
    fn handleClient(self: *Server, client_fd: i32) void {
        defer {
            _ = linux.shutdown(client_fd, linux.SHUT.WR);
            _ = linux.close(client_fd);
        }

        const start_ms = getMonotonicMs();
        var buf: [65536]u8 = undefined;
        var total_read: usize = 0;

        // Read request header (bounded by 15s idle timeout and strict 30s overall deadline)
        while (total_read < buf.len) {
            var chunk_len: usize = 0;
            const res = readChunkWithDeadline(self, client_fd, buf[total_read..], start_ms, &chunk_len);
            switch (res) {
                .ok => {
                    total_read += chunk_len;
                    if (std.mem.indexOf(u8, buf[0..total_read], "\r\n\r\n") != null) {
                        break;
                    }
                },
                .eof => break,
                .idle_timeout => return, // Idle socket closes silently
                .deadline_timeout => {
                    self.sendError(client_fd, 408, "request_timeout", "Request read timed out (exceeded 30s deadline)");
                    return;
                },
                .read_error => return,
            }
        }

        const req_data = buf[0..total_read];
        const header_end = std.mem.indexOf(u8, req_data, "\r\n\r\n") orelse {
            self.sendError(client_fd, 400, "bad_request", "Request header exceeds buffer or malformed");
            return;
        };

        const headers_part = req_data[0..header_end];
        var body_part = req_data[header_end + 4 ..];

        // Parse Request-Line: METHOD PATH HTTP/VERSION
        var line_it = std.mem.splitSequence(u8, headers_part, "\r\n");
        const req_line = line_it.next() orelse {
            self.sendError(client_fd, 400, "bad_request", "Empty request line");
            return;
        };

        var token_it = std.mem.splitScalar(u8, req_line, ' ');
        const method = token_it.next() orelse "";
        const path = token_it.next() orelse "";
        const version = token_it.next() orelse "";

        if (method.len == 0 or path.len == 0 or version.len == 0 or token_it.next() != null) {
            self.sendError(client_fd, 400, "bad_request", "Malformed request line: expected 'METHOD PATH HTTP/VERSION'");
            return;
        }

        if (!std.mem.eql(u8, version, "HTTP/1.1") and !std.mem.eql(u8, version, "HTTP/1.0")) {
            self.sendError(client_fd, 505, "http_version_not_supported", "Only HTTP/1.0 and HTTP/1.1 are supported");
            return;
        }

        if (path[0] != '/') {
            self.sendError(client_fd, 400, "bad_request", "Request path must be an absolute path starting with '/'");
            return;
        }

        // Parse headers
        var content_length: ?usize = null;
        var has_transfer_encoding = false;

        while (line_it.next()) |header_line| {
            if (header_line.len == 0) continue;

            if (std.ascii.startsWithIgnoreCase(header_line, "transfer-encoding:")) {
                has_transfer_encoding = true;
            } else if (std.ascii.startsWithIgnoreCase(header_line, "content-length:")) {
                if (content_length != null) {
                    // RFC 7230 §3.3.2: Duplicate Content-Length header is a smuggling vulnerability
                    self.sendError(client_fd, 400, "bad_request", "Duplicate Content-Length header is not permitted");
                    return;
                }
                const val_str = std.mem.trim(u8, header_line["content-length:".len..], " \t");
                content_length = std.fmt.parseInt(usize, val_str, 10) catch {
                    self.sendError(client_fd, 400, "bad_request", "Invalid Content-Length header value");
                    return;
                };
            }
        }

        if (has_transfer_encoding) {
            self.sendError(client_fd, 501, "not_implemented", "Transfer-Encoding is not supported");
            return;
        }

        // Route: GET/HEAD /health (R7-06)
        if (std.mem.eql(u8, path, "/health")) {
            const is_get = std.mem.eql(u8, method, "GET");
            const is_head = std.mem.eql(u8, method, "HEAD");
            if (is_get or is_head) {
                self.sendResponse(client_fd, 200, "{\"status\":\"ok\"}\n", is_head);
                return;
            } else {
                self.sendError(client_fd, 405, "method_not_allowed", "Only GET and HEAD methods are allowed for /health");
                return;
            }
        }

        // Only POST /v1/systemone is supported for decisions
        if (!std.mem.eql(u8, path, "/v1/systemone")) {
            self.sendError(client_fd, 404, "not_found", "Path not found. Decision API is at POST /v1/systemone");
            return;
        }

        if (!std.mem.eql(u8, method, "POST")) {
            self.sendError(client_fd, 405, "method_not_allowed", "Only POST method is allowed for /v1/systemone");
            return;
        }

        const cl = content_length orelse {
            self.sendError(client_fd, 411, "length_required", "Content-Length header is required");
            return;
        };

        // Bounded request body cap: 64 KB is generous for decision requests and bounds allocations (R7-05)
        if (cl > 64 * 1024) {
            self.sendError(client_fd, 413, "payload_too_large", "Request payload exceeds 64 KB limit");
            return;
        }

        // Read remaining body if not fully received
        var full_body: []const u8 = undefined;
        var heap_body: ?[]u8 = null;
        defer if (heap_body) |hb| self.allocator.free(hb);

        if (body_part.len >= cl) {
            full_body = body_part[0..cl];
        } else {
            const body_buf = self.allocator.alloc(u8, cl) catch {
                self.sendError(client_fd, 500, "internal_error", "Memory allocation failed");
                return;
            };
            heap_body = body_buf;
            @memcpy(body_buf[0..body_part.len], body_part);
            var body_read = body_part.len;

            while (body_read < cl) {
                var chunk_len: usize = 0;
                const res = readChunkWithDeadline(self, client_fd, body_buf[body_read..], start_ms, &chunk_len);
                switch (res) {
                    .ok => {
                        body_read += chunk_len;
                    },
                    .eof => break,
                    .idle_timeout, .deadline_timeout => {
                        self.sendError(client_fd, 408, "request_timeout", "Request body read timed out (exceeded deadline)");
                        return;
                    },
                    .read_error => {
                        self.sendError(client_fd, 400, "read_error", "Failed to read full request body");
                        return;
                    },
                }
            }

            if (body_read < cl) {
                self.sendError(client_fd, 400, "incomplete_body", "Client closed connection before sending complete body");
                return;
            }
            full_body = body_buf;
        }

        self.handleDecisionRequest(client_fd, full_body);
    }

    /// Parses JSON decision request, executes decision engine, and writes JSON response.
    fn handleDecisionRequest(self: *Server, client_fd: i32, json_body: []const u8) void {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const alloc = arena.allocator();

        const parsed = std.json.parseFromSlice(WireDecisionRequest, alloc, json_body, .{
            .ignore_unknown_fields = true,
        }) catch |err| {
            var err_msg_buf: [128]u8 = undefined;
            const msg = std.fmt.bufPrint(&err_msg_buf, "Malformed JSON request body: {s}", .{@errorName(err)}) catch "Malformed JSON";
            self.sendError(client_fd, 400, "invalid_json", msg);
            return;
        };
        defer parsed.deinit();

        const wire_req = parsed.value;

        // Parse schema type
        const schema_type: SchemaType = if (std.mem.eql(u8, wire_req.schema.type, "boolean"))
            .boolean
        else if (std.mem.eql(u8, wire_req.schema.type, "choice"))
            .choice
        else if (std.mem.eql(u8, wire_req.schema.type, "score"))
            .score
        else {
            self.sendError(client_fd, 400, "invalid_schema_type", "Schema type must be 'boolean', 'choice', or 'score'");
            return;
        };

        // Convert candidate specs
        const cand_specs = alloc.alloc(CandidateSpec, wire_req.schema.candidates.len) catch {
            self.sendError(client_fd, 500, "internal_error", "Allocation failed");
            return;
        };
        for (wire_req.schema.candidates, 0..) |c, i| {
            cand_specs[i] = .{
                .token = c.token,
                .label = c.label,
            };
        }

        // Build DecisionRequest
        const dec_req = DecisionRequest{
            .query = wire_req.query,
            .context = wire_req.context,
            .schema_type = schema_type,
            .candidates = cand_specs,
            .step_cap = if (wire_req.options) |o| o.step_cap orelse 1 else 1,
            .temperature = if (wire_req.options) |o| o.temperature orelse 1.0 else 1.0,
            .entropy_threshold = if (wire_req.options) |o| o.entropy_threshold else null,
            .canvas_length = if (wire_req.options) |o| o.canvas_length else null,
        };

        // Execute decision engine
        const resp = self.engine.decide(alloc, dec_req) catch |err| {
            switch (err) {
                error.EmptyQuery => self.sendError(client_fd, 400, "empty_query", "Query string must not be empty"),
                error.TooFewCandidates => self.sendError(client_fd, 400, "too_few_candidates", "At least 2 candidate options are required"),
                error.TooManyCandidates => self.sendError(client_fd, 400, "too_many_candidates", "At most 32 candidate options are supported"),
                error.DuplicateCandidateToken => self.sendError(client_fd, 400, "duplicate_candidate_token", "Candidate options contain duplicate token IDs"),
                error.DuplicateCandidateLabel => self.sendError(client_fd, 400, "duplicate_candidate_label", "Candidate options contain duplicate display labels"),
                error.UnsupportedStepCap => self.sendError(client_fd, 400, "unsupported_step_cap", "step_cap must be 1 in single-step decision v0"),
                error.InvalidTemperature => self.sendError(client_fd, 400, "invalid_temperature", "Temperature must be a finite positive number"),
                error.InvalidEntropyThreshold => self.sendError(client_fd, 400, "invalid_entropy_threshold", "Entropy threshold must be between 0.0 and 1.0"),
                error.InvalidCanvasLength => self.sendError(client_fd, 400, "invalid_canvas_length", "Canvas length must be a multiple of 16 between 16 and 256"),
                error.TokenNotFound => self.sendError(client_fd, 400, "token_not_found", "A candidate token string does not exist in model vocabulary"),
                error.MultiTokenCandidateNotAllowed => self.sendError(client_fd, 400, "multi_token_candidate", "Candidate token string resolves to multiple tokens; single-token code required"),
                error.CanvasOverflow => self.sendError(client_fd, 400, "canvas_overflow", "Prompt exceeds configured canvas length"),
                else => {
                    var err_msg_buf: [128]u8 = undefined;
                    const msg = std.fmt.bufPrint(&err_msg_buf, "Decision engine execution error: {s}", .{@errorName(err)}) catch "Internal server error";
                    self.sendError(client_fd, 500, "internal_error", msg);
                },
            }
            return;
        };

        // Serialize DecisionResponse JSON conforming to specs/decision-api.md §3.2
        const resp_json = serializeDecisionResponse(alloc, resp) catch {
            self.sendError(client_fd, 500, "internal_error", "Failed to serialize response JSON");
            return;
        };

        self.sendJson(client_fd, 200, resp_json);
    }

    /// Writes an RFC 8259 compliant escaped JSON string (including surrounding quotes).
    pub fn writeEscapedJsonString(list: *std.ArrayList(u8), allocator: std.mem.Allocator, str: []const u8) !void {
        try list.append(allocator, '"');
        for (str) |c| {
            switch (c) {
                '"' => try list.appendSlice(allocator, "\\\""),
                '\\' => try list.appendSlice(allocator, "\\\\"),
                '\n' => try list.appendSlice(allocator, "\\n"),
                '\r' => try list.appendSlice(allocator, "\\r"),
                '\t' => try list.appendSlice(allocator, "\\t"),
                0x00...0x08, 0x0B, 0x0C, 0x0E...0x1F => {
                    var hex_buf: [6]u8 = undefined;
                    const hex = std.fmt.bufPrint(&hex_buf, "\\u{x:0>4}", .{c}) catch "\\u0000";
                    try list.appendSlice(allocator, hex);
                },
                else => try list.append(allocator, c),
            }
        }
        try list.append(allocator, '"');
    }

    /// Serializes a complete DecisionResponse to a formatted JSON string with full RFC 8259 escaping (R6-01).
    pub fn serializeDecisionResponse(allocator: std.mem.Allocator, resp: DecisionResponse) ![]const u8 {
        var list: std.ArrayList(u8) = .empty;
        errdefer list.deinit(allocator);

        try list.appendSlice(allocator, "{\n  \"decision\": ");
        try writeEscapedJsonString(&list, allocator, resp.decision);

        try list.print(allocator,
            \\,
            \\  "decision_probability": {d:.6},
            \\  "entropy_nats": {d:.6},
            \\  "normalized_entropy": {d:.6},
            \\  "uncertain": {},
            \\  "steps_executed": {d},
            \\  "canvas_tokens_used": {d},
            \\  "candidates": [
            \\
        , .{
            resp.decision_probability,
            resp.entropy_nats,
            resp.normalized_entropy,
            resp.uncertain,
            resp.steps_executed,
            resp.canvas_tokens_used,
        });

        for (resp.candidates, 0..) |c, i| {
            const comma = if (i + 1 < resp.candidates.len) "," else "";
            try list.appendSlice(allocator, "    {\n      \"label\": ");
            try writeEscapedJsonString(&list, allocator, c.label);
            try list.print(allocator,
                \\,
                \\      "token_id": {d},
                \\      "probability": {d:.6},
                \\      "raw_logit": {d:.6}
                \\    }}{s}
                \\
            , .{
                c.token_id,
                c.probability,
                c.raw_logit,
                comma,
            });
        }

        try list.appendSlice(allocator, "  ]\n}\n");
        return list.toOwnedSlice(allocator);
    }

    /// Sends a JSON response with HTTP headers.
    fn sendJson(self: *Server, client_fd: i32, status_code: u16, json_body: []const u8) void {
        self.sendResponse(client_fd, status_code, json_body, false);
    }

    /// Sends an HTTP JSON response, omitting the response body if is_head is true (R7-06).
    fn sendResponse(self: *Server, client_fd: i32, status_code: u16, json_body: []const u8, is_head: bool) void {
        _ = self;
        const status_text = switch (status_code) {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            405 => "Method Not Allowed",
            408 => "Request Timeout",
            411 => "Length Required",
            413 => "Payload Too Large",
            500 => "Internal Server Error",
            501 => "Not Implemented",
            505 => "HTTP Version Not Supported",
            else => "Response",
        };

        var header_buf: [512]u8 = undefined;
        const header = std.fmt.bufPrint(&header_buf,
            "HTTP/1.1 {d} {s}\r\n" ++
                "Content-Type: application/json\r\n" ++
                "Content-Length: {d}\r\n" ++
                "Connection: close\r\n" ++
                "\r\n",
            .{ status_code, status_text, json_body.len },
        ) catch return;

        writeAll(client_fd, header);
        if (!is_head) {
            writeAll(client_fd, json_body);
        }
    }

    /// Sends a structured JSON error response conforming to specs/decision-api.md §5 with RFC 8259 escaping.
    fn sendError(self: *Server, client_fd: i32, status_code: u16, code: []const u8, message: []const u8) void {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const alloc = arena.allocator();

        var list: std.ArrayList(u8) = .empty;
        list.appendSlice(alloc, "{\n  \"error\": {\n    \"code\": ") catch return;
        writeEscapedJsonString(&list, alloc, code) catch return;
        list.appendSlice(alloc, ",\n    \"message\": ") catch return;
        writeEscapedJsonString(&list, alloc, message) catch return;
        list.appendSlice(alloc, "\n  }\n}\n") catch return;

        self.sendJson(client_fd, status_code, list.items);
    }

    fn writeAll(client_fd: i32, bytes: []const u8) void {
        var written: usize = 0;
        while (written < bytes.len) {
            const w_rc = linux.write(client_fd, bytes[written..].ptr, bytes.len - written);
            const err = linux.errno(w_rc);
            if (err != .SUCCESS) {
                if (err == .INTR) continue;
                return;
            }
            if (w_rc == 0) return;
            written += w_rc;
        }
    }
};

test "parseIp4 valid and invalid addresses" {
    const localhost = try parseIp4("127.0.0.1");
    try std.testing.expectEqual(@as(u32, 0x0100007F), localhost);

    const zero = try parseIp4("0.0.0.0");
    try std.testing.expectEqual(@as(u32, 0), zero);

    const custom = try parseIp4("192.168.1.10");
    try std.testing.expectEqual(@as(u32, 0x0A01A8C0), custom);

    try std.testing.expectError(error.InvalidIp, parseIp4("127.0.0"));
    try std.testing.expectError(error.InvalidIp, parseIp4("127.0.0.1.5"));
}

test "WireDecisionRequest JSON deserialization" {
    const allocator = std.testing.allocator;
    const json_str =
        \\{
        \\  "query": "Is database healthy?",
        \\  "context": "Connection pools normal",
        \\  "schema": {
        \\    "type": "boolean",
        \\    "candidates": [
        \\      {"token": "Yes", "label": "Yes"},
        \\      {"token": "No", "label": "No"}
        \\    ]
        \\  },
        \\  "options": {
        \\    "step_cap": 1,
        \\    "temperature": 1.0,
        \\    "entropy_threshold": 0.5
        \\  }
        \\}
    ;

    const parsed = try std.json.parseFromSlice(WireDecisionRequest, allocator, json_str, .{});
    defer parsed.deinit();

    const req = parsed.value;
    try std.testing.expectEqualStrings("Is database healthy?", req.query);
    try std.testing.expectEqualStrings("Connection pools normal", req.context.?);
    try std.testing.expectEqualStrings("boolean", req.schema.type);
    try std.testing.expectEqual(@as(usize, 2), req.schema.candidates.len);
    try std.testing.expectEqualStrings("Yes", req.schema.candidates[0].token);
    try std.testing.expectEqualStrings("No", req.schema.candidates[1].token);
    try std.testing.expectEqual(@as(?u32, 1), req.options.?.step_cap);
    try std.testing.expectEqual(@as(?f32, 1.0), req.options.?.temperature);
    try std.testing.expectEqual(@as(?f32, 0.5), req.options.?.entropy_threshold);
}

test "serializeDecisionResponse formatted JSON output" {
    const allocator = std.testing.allocator;
    const candidates = [_]decision.CandidateResult{
        .{ .label = "Yes", .token_id = 10784, .probability = 0.95, .raw_logit = 15.2 },
        .{ .label = "No", .token_id = 3771, .probability = 0.05, .raw_logit = 10.1 },
    };
    const resp = DecisionResponse{
        .decision = "Yes",
        .decision_probability = 0.95,
        .entropy_nats = 0.198,
        .normalized_entropy = 0.285,
        .uncertain = false,
        .candidates = &candidates,
        .steps_executed = 1,
        .canvas_tokens_used = 70,
    };

    const json = try Server.serializeDecisionResponse(allocator, resp);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"decision\": \"Yes\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"decision_probability\": 0.950000") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"uncertain\": false") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"label\": \"Yes\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"token_id\": 10784") != null);
}

test "serializeDecisionResponse with special characters in label and decision (R6-01)" {
    const allocator = std.testing.allocator;
    const special_label = "Option \"A\" \\ [backslash] \n newline \t tab";
    const candidates = [_]decision.CandidateResult{
        .{ .label = special_label, .token_id = 100, .probability = 0.8, .raw_logit = 12.0 },
        .{ .label = "Normal B", .token_id = 200, .probability = 0.2, .raw_logit = 10.0 },
    };
    const resp = DecisionResponse{
        .decision = special_label,
        .decision_probability = 0.8,
        .entropy_nats = 0.5,
        .normalized_entropy = 0.72,
        .uncertain = false,
        .candidates = &candidates,
        .steps_executed = 1,
        .canvas_tokens_used = 40,
    };

    const json = try Server.serializeDecisionResponse(allocator, resp);
    defer allocator.free(json);

    // Verify it is strictly valid JSON by parsing it with std.json
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();

    const root_obj = parsed.value.object;
    try std.testing.expectEqualStrings(special_label, root_obj.get("decision").?.string);

    const cands_arr = root_obj.get("candidates").?.array;
    try std.testing.expectEqualStrings(special_label, cands_arr.items[0].object.get("label").?.string);
}

test "Server GET /health over loopback TCP" {
    const allocator = std.testing.allocator;
    var server_inst = Server.init(allocator, undefined, "127.0.0.1", 0);
    try server_inst.listen();
    defer server_inst.deinit();

    const thread = try std.Thread.spawn(.{}, struct {
        fn run(s: *Server) void {
            s.serve() catch {};
        }
    }.run, .{&server_inst});

    // Connect client
    const rc = linux.socket(linux.AF.INET, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, 0);
    try std.testing.expect(linux.errno(rc) == .SUCCESS);
    const client_fd: i32 = @intCast(rc);
    defer _ = linux.close(client_fd);

    var addr: linux.sockaddr.in = .{
        .family = linux.AF.INET,
        .port = std.mem.nativeToBig(u16, server_inst.port),
        .addr = try parseIp4("127.0.0.1"),
        .zero = [8]u8{ 0, 0, 0, 0, 0, 0, 0, 0 },
    };
    const c_rc = linux.connect(client_fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.in));
    try std.testing.expect(linux.errno(c_rc) == .SUCCESS);

    const req = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    _ = linux.write(client_fd, req.ptr, req.len);

    var resp_buf: [1024]u8 = undefined;
    var total_read: usize = 0;
    while (total_read < resp_buf.len) {
        const r_rc = linux.read(client_fd, resp_buf[total_read..].ptr, resp_buf.len - total_read);
        if (linux.errno(r_rc) != .SUCCESS or r_rc == 0) break;
        total_read += r_rc;
    }

    const resp_text = resp_buf[0..total_read];
    try std.testing.expect(std.mem.indexOf(u8, resp_text, "200 OK") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp_text, "{\"status\":\"ok\"}") != null);

    server_inst.stop();
    // Wake up accept by sending one dummy connection to allow thread to exit cleanly
    const wake_rc = linux.socket(linux.AF.INET, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, 0);
    if (linux.errno(wake_rc) == .SUCCESS) {
        const wake_fd: i32 = @intCast(wake_rc);
        _ = linux.connect(wake_fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.in));
        _ = linux.close(wake_fd);
    }
    thread.join();
}

test "Server concurrent /health response while active connection is open (F4)" {
    const allocator = std.testing.allocator;
    var server_inst = Server.init(allocator, undefined, "127.0.0.1", 0);
    try server_inst.listen();
    defer server_inst.deinit();

    const thread = try std.Thread.spawn(.{}, struct {
        fn run(s: *Server) void {
            s.serve() catch {};
        }
    }.run, .{&server_inst});

    var addr: linux.sockaddr.in = .{
        .family = linux.AF.INET,
        .port = std.mem.nativeToBig(u16, server_inst.port),
        .addr = try parseIp4("127.0.0.1"),
        .zero = [8]u8{ 0, 0, 0, 0, 0, 0, 0, 0 },
    };

    // Client 1 connects and holds connection open (simulating active connection)
    const rc1 = linux.socket(linux.AF.INET, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, 0);
    try std.testing.expect(linux.errno(rc1) == .SUCCESS);
    const client1_fd: i32 = @intCast(rc1);

    const c1_rc = linux.connect(client1_fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.in));
    try std.testing.expect(linux.errno(c1_rc) == .SUCCESS);

    // Send partial line so client 1 stays active in read loop
    const partial_data = "POST /v1/";
    _ = linux.write(client1_fd, partial_data.ptr, partial_data.len);

    // Give server worker thread a few ms to pick up client 1
    const ts_wait = std.posix.timespec{ .sec = 0, .nsec = 20_000_000 };
    _ = std.posix.system.nanosleep(&ts_wait, null);

    // Verify active_connections is at least 1
    try std.testing.expect(server_inst.active_connections.load(.acquire) >= 1);

    // Client 2 connects to GET /health concurrently
    const rc2 = linux.socket(linux.AF.INET, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, 0);
    try std.testing.expect(linux.errno(rc2) == .SUCCESS);
    const client2_fd: i32 = @intCast(rc2);
    defer _ = linux.close(client2_fd);

    const c2_rc = linux.connect(client2_fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.in));
    try std.testing.expect(linux.errno(c2_rc) == .SUCCESS);

    const health_req = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    const start_time = Server.getMonotonicMs();
    _ = linux.write(client2_fd, health_req.ptr, health_req.len);

    var resp_buf: [1024]u8 = undefined;
    var total_read: usize = 0;
    while (total_read < resp_buf.len) {
        const r_rc = linux.read(client2_fd, resp_buf[total_read..].ptr, resp_buf.len - total_read);
        if (linux.errno(r_rc) != .SUCCESS or r_rc == 0) break;
        total_read += r_rc;
    }
    const elapsed_ms = Server.getMonotonicMs() - start_time;

    const resp_text = resp_buf[0..total_read];
    try std.testing.expect(std.mem.indexOf(u8, resp_text, "200 OK") != null);
    try std.testing.expect(std.mem.indexOf(u8, resp_text, "{\"status\":\"ok\"}") != null);
    // /health must respond immediately (< 500ms), proving it was NOT blocked by client 1
    try std.testing.expect(elapsed_ms < 500.0);

    // Now close client 1 cleanly
    _ = linux.close(client1_fd);

    server_inst.stop();
    const wake_rc = linux.socket(linux.AF.INET, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, 0);
    if (linux.errno(wake_rc) == .SUCCESS) {
        const wake_fd: i32 = @intCast(wake_rc);
        _ = linux.connect(wake_fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.in));
        _ = linux.close(wake_fd);
    }
    thread.join();
}

test "readChunkWithDeadline immediate return on expired deadline (R8-01)" {
    var dest_buf: [16]u8 = undefined;
    var bytes_read: usize = 0;
    const fake_past_start = Server.getMonotonicMs() - 35_000;
    const status = Server.readChunkWithDeadline(null, 0, &dest_buf, fake_past_start, &bytes_read);
    try std.testing.expectEqual(Server.ReadResult.deadline_timeout, status);
}

test "readChunkWithDeadline caps poll wait to remaining deadline and returns deadline_timeout (R8-01)" {
    var sv: [2]i32 = undefined;
    const sp_rc = linux.socketpair(linux.AF.UNIX, linux.SOCK.STREAM, 0, &sv);
    if (linux.errno(sp_rc) != .SUCCESS) return;
    defer {
        _ = linux.close(sv[0]);
        _ = linux.close(sv[1]);
    }

    var dest_buf: [16]u8 = undefined;
    var bytes_read: usize = 0;
    // Set start time to 29.95s ago (only 50ms remaining before 30s deadline)
    const near_deadline_start = Server.getMonotonicMs() - 29_950;
    const status = Server.readChunkWithDeadline(null, sv[0], &dest_buf, near_deadline_start, &bytes_read);
    try std.testing.expectEqual(Server.ReadResult.deadline_timeout, status);
}

test "getRemainingDeadlineMs bounds and expiration (R9-01)" {
    const now = Server.getMonotonicMs();
    // Fresh start: remaining should be approximately 30_000ms
    const rem_fresh = Server.getRemainingDeadlineMs(now);
    try std.testing.expect(rem_fresh != null);
    try std.testing.expect(rem_fresh.? <= 30_000 and rem_fresh.? >= 29_900);

    // Expired start: 30s or more ago
    const expired_start = now - 30_000;
    try std.testing.expectEqual(@as(?i64, null), Server.getRemainingDeadlineMs(expired_start));

    const way_past_start = now - 35_000;
    try std.testing.expectEqual(@as(?i64, null), Server.getRemainingDeadlineMs(way_past_start));
}


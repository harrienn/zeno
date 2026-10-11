#!/usr/bin/env python3
"""
Zeno — HTTP Decision API Server Integration Test Suite
Tests /health and /v1/systemone against the specs/decision-api.md contract,
including framing robustness, RFC 8259 escaping, and canvas overflow limits.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8999
MODEL_PATH = "models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def wait_for_server(timeout=60):
    url = f"{BASE_URL}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("status") == "ok":
                        return True
        except Exception:
            time.sleep(0.5)
    return False


def test_request(method, path, body=None, expected_status=200):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"} if body is not None else {}
    data = json.dumps(body).encode("utf-8") if body is not None else None

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
            status = resp.status
            return status, json.loads(content) if content else {}
    except urllib.error.HTTPError as e:
        content = e.read().decode("utf-8")
        return e.code, json.loads(content) if content else {}


def send_raw_http(raw_bytes):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((SERVER_HOST, SERVER_PORT))
    s.sendall(raw_bytes)
    resp = b""
    s.settimeout(5.0)
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            resp += chunk
    except Exception:
        pass
    s.close()
    return resp.decode("utf-8", errors="replace")


def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"Starting Zeno API server on {BASE_URL}...")
    server_proc = subprocess.Popen(
        ["./zig-out/bin/zeno", "serve", MODEL_PATH, "--host", SERVER_HOST, "--port", str(SERVER_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        print("Waiting for server to become ready...")
        if not wait_for_server(60):
            print("ERROR: Server failed to start within timeout.")
            if server_proc.poll() is not None:
                stdout, stderr = server_proc.communicate()
                print("STDOUT:", stdout)
                print("STDERR:", stderr)
            sys.exit(1)
        print("✓ Server is ready.")

        # Test 1: GET /health
        print("\n--- Test 1: GET /health ---")
        status, data = test_request("GET", "/health")
        assert status == 200, f"Expected 200, got {status}"
        assert data.get("status") == "ok", f"Expected ok status, got {data}"
        print(f"✓ GET /health returned 200 OK: {data}")

        # Test 2: GET /v1/systemone (405 Method Not Allowed)
        print("\n--- Test 2: GET /v1/systemone (Method Not Allowed) ---")
        status, data = test_request("GET", "/v1/systemone")
        assert status == 405, f"Expected 405, got {status}"
        assert "error" in data and data["error"].get("code") == "method_not_allowed"
        print(f"✓ Correctly returned 405: {data}")

        # Test 3: POST /unknown (404 Not Found)
        print("\n--- Test 3: POST /unknown (Not Found) ---")
        status, data = test_request("POST", "/unknown", body={"query": "test"})
        assert status == 404, f"Expected 404, got {status}"
        assert "error" in data and data["error"].get("code") == "not_found"
        print(f"✓ Correctly returned 404: {data}")

        # Test 4: Schema validation: unsupported step_cap > 1 (400 Bad Request)
        print("\n--- Test 4: Schema validation (step_cap > 1) ---")
        status, data = test_request("POST", "/v1/systemone", body={
            "query": "Is system healthy?",
            "schema": {
                "type": "boolean",
                "candidates": [
                    {"token": "Yes", "label": "Yes"},
                    {"token": "No", "label": "No"}
                ]
            },
            "options": {"step_cap": 2}
        })
        assert status == 400, f"Expected 400, got {status}"
        assert "error" in data and data["error"].get("code") == "unsupported_step_cap"
        print(f"✓ Correctly returned 400 for step_cap > 1: {data}")

        # Test 5: Schema validation: fewer than 2 candidates
        print("\n--- Test 5: Schema validation (too few candidates) ---")
        status, data = test_request("POST", "/v1/systemone", body={
            "query": "Is system healthy?",
            "schema": {
                "type": "boolean",
                "candidates": [{"token": "Yes", "label": "Yes"}]
            }
        })
        assert status == 400, f"Expected 400, got {status}"
        assert "error" in data and data["error"].get("code") == "too_few_candidates"
        print(f"✓ Correctly returned 400 for too few candidates: {data}")

        # Test 6: Raw HTTP framing: Malformed request line (R6-06)
        print("\n--- Test 6: Malformed request line (400 Bad Request) ---")
        raw_resp = send_raw_http(b"POST /v1/systemone\r\nHost: 127.0.0.1\r\n\r\n")
        assert "400 Bad Request" in raw_resp, f"Expected 400, got: {raw_resp[:100]}"
        assert "bad_request" in raw_resp
        print("✓ Correctly rejected 2-token request line with 400 Bad Request")

        # Test 7: Raw HTTP framing: Duplicate Content-Length (R6-06)
        print("\n--- Test 7: Duplicate Content-Length rejection (400 Bad Request) ---")
        raw_resp = send_raw_http(b"POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 10\r\nContent-Length: 20\r\n\r\n{}")
        assert "400 Bad Request" in raw_resp, f"Expected 400, got: {raw_resp[:100]}"
        assert "Duplicate Content-Length" in raw_resp
        print("✓ Correctly rejected duplicate Content-Length with 400 Bad Request")

        # Test 8: Raw HTTP framing: Transfer-Encoding rejection (R6-06, R9-02)
        print("\n--- Test 8: Transfer-Encoding rejection (501 Not Implemented) ---")
        raw_resp = send_raw_http(b"POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n")
        assert "501 Not Implemented" in raw_resp, f"Expected 501, got: {raw_resp[:100]}"
        assert "Transfer-Encoding is not supported" in raw_resp, f"Expected error message, got: {raw_resp[:200]}"
        print("✓ Correctly rejected chunked transfer encoding with 501 Not Implemented")

        raw_resp_gzip = send_raw_http(b"POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: gzip\r\n\r\n")
        assert "501 Not Implemented" in raw_resp_gzip, f"Expected 501, got: {raw_resp_gzip[:100]}"
        assert "Transfer-Encoding is not supported" in raw_resp_gzip, f"Expected error message, got: {raw_resp_gzip[:200]}"
        print("✓ Correctly rejected non-chunked (gzip) transfer encoding with 501 Not Implemented")

        # Test 9: Prompt size bounding: Canvas Overflow (R6-03)
        print("\n--- Test 9: Prompt size bounding (Canvas Overflow) ---")
        oversized_query = "This is an extremely long question " * 200  # generates > 768 tokens
        status, data = test_request("POST", "/v1/systemone", body={
            "query": oversized_query,
            "schema": {
                "type": "boolean",
                "candidates": [
                    {"token": "Yes", "label": "Yes"},
                    {"token": "No", "label": "No"}
                ]
            }
        })
        assert status == 400, f"Expected 400, got {status}"
        assert "error" in data and data["error"].get("code") == "canvas_overflow"
        print(f"✓ Correctly rejected oversized prompt with canvas_overflow: {data}")

        # Test 10: Live Decision Query with Escaped Labels (R6-01)
        print("\n--- Test 10: Live Decision Query with Escaped Labels (POST /v1/systemone) ---")
        escaped_label_yes = 'Yes "Primary" \\ verified'
        escaped_label_no = 'No "Alternate" \n second'
        query_payload = {
            "query": "Is the system healthy and operational?",
            "schema": {
                "type": "boolean",
                "candidates": [
                    {"token": "Yes", "label": escaped_label_yes},
                    {"token": "No", "label": escaped_label_no}
                ]
            },
            "options": {
                "step_cap": 1,
                "temperature": 1.0,
                "entropy_threshold": 0.5
            }
        }

        start_time = time.time()
        status, data = test_request("POST", "/v1/systemone", body=query_payload)
        elapsed = time.time() - start_time
        print(f"Decision request completed in {elapsed:.2f}s with status {status}")

        assert status == 200, f"Expected 200, got {status}: {data}"
        print("Response received:")
        print(json.dumps(data, indent=2))

        assert data.get("decision") in [escaped_label_yes, escaped_label_no], f"Expected decision in candidates, got {data.get('decision')}"
        # Semantic threshold assertion (independent of model distribution)
        expected_uncertain = (data.get("normalized_entropy", 0.0) >= 0.5)
        assert data.get("uncertain") == expected_uncertain, f"Expected uncertain={expected_uncertain}, got {data.get('uncertain')}"
        assert len(data.get("candidates", [])) == 2, f"Expected 2 candidates, got {len(data.get('candidates', []))}"
        assert data["candidates"][0]["label"] == escaped_label_yes
        assert data["candidates"][1]["label"] == escaped_label_no
        print(f"✓ Decision '{data.get('decision')}' properly serialized and parsed with escaped labels (prob {data.get('decision_probability'):.4f}, uncertain={data.get('uncertain')})!")

        # Test 11: HTTP HEAD /health returns headers with empty body (R7-06)
        print("\n--- Test 11: HEAD /health returns empty body ---")
        raw_head_resp = send_raw_http(b"HEAD /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        assert "200 OK" in raw_head_resp, f"Expected 200 OK, got: {raw_head_resp[:100]}"
        assert "Content-Length: 16" in raw_head_resp, f"Expected Content-Length: 16 header"
        parts = raw_head_resp.split("\r\n\r\n", 1)
        assert len(parts) == 2 and parts[1] == "", f"Expected empty body for HEAD, got: {parts[1]}"
        print("✓ Correctly returned 200 OK headers with no response body for HEAD /health")

        # Test 12: Wire error codes and native validation parity (R7-02)
        print("\n--- Test 12: Wire error code verification ---")
        # 12a: empty_query
        status, data = test_request("POST", "/v1/systemone", body={
            "query": "",
            "schema": {"type": "boolean", "candidates": [{"token": "Yes", "label": "Yes"}, {"token": "No", "label": "No"}]}
        })
        assert status == 400 and data.get("error", {}).get("code") == "empty_query", f"Expected empty_query, got: {data}"

        # 12b: duplicate_candidate_label
        status, data = test_request("POST", "/v1/systemone", body={
            "query": "Is it valid?",
            "schema": {"type": "boolean", "candidates": [{"token": "Yes", "label": "Same"}, {"token": "No", "label": "Same"}]}
        })
        assert status == 400 and data.get("error", {}).get("code") == "duplicate_candidate_label", f"Expected duplicate_candidate_label, got: {data}"

        # 12c: invalid_schema_type
        status, data = test_request("POST", "/v1/systemone", body={
            "query": "Is it valid?",
            "schema": {"type": "unknown_type", "candidates": [{"token": "Yes", "label": "Yes"}, {"token": "No", "label": "No"}]}
        })
        assert status == 400 and data.get("error", {}).get("code") == "invalid_schema_type", f"Expected invalid_schema_type, got: {data}"

        # 12d: http_version_not_supported
        raw_resp = send_raw_http(b"GET /health HTTP/2.0\r\nHost: 127.0.0.1\r\n\r\n")
        assert "505 HTTP Version Not Supported" in raw_resp, f"Expected 505, got: {raw_resp[:100]}"
        assert "http_version_not_supported" in raw_resp, f"Expected code http_version_not_supported"
        print("✓ Successfully verified wire error codes: empty_query, duplicate_candidate_label, invalid_schema_type, http_version_not_supported")

        # Test 13: Oversized candidate label within HTTP limit (<64 KB) but >16 KiB prompt preflight (R8-02)
        print("\n--- Test 13: Oversized candidate label preflight rejection (R8-02) ---")
        huge_label = "L" * 17000  # 17 KB label, total body ~18 KB (< 64 KB HTTP cap)
        status, data = test_request("POST", "/v1/systemone", body={
            "query": "Is it valid?",
            "schema": {
                "type": "boolean",
                "candidates": [
                    {"token": "Yes", "label": huge_label},
                    {"token": "No", "label": "No"}
                ]
            }
        })
        assert status == 400 and data.get("error", {}).get("code") == "canvas_overflow", f"Expected canvas_overflow, got status {status}: {data}"
        print(f"✓ Correctly rejected oversized candidate label before formatting/tokenizing: {data}")

        # Test 14: Client disconnects during body transmission
        print("\n--- Test 14: Incomplete body handling ---")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((SERVER_HOST, SERVER_PORT))
        s.sendall(b"POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 100\r\n\r\n12345")
        s.shutdown(socket.SHUT_WR)
        resp_data = b""
        s.settimeout(5.0)
        try:
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                resp_data += chunk
        except Exception:
            pass
        s.close()
        resp_str = resp_data.decode("utf-8", errors="replace")
        assert "400 Bad Request" in resp_str, f"Expected 400, got: {resp_str[:100]}"
        assert "incomplete_body" in resp_str, f"Expected incomplete_body in response"
        print("✓ Correctly returned 400 Bad Request with incomplete_body on truncated client transmission")

        print("\n=======================================================")
        print(" ALL HTTP SERVER INTEGRATION TESTS PASSED (14/14)")
        print("=======================================================")

    finally:
        print("\nTerminating Zeno server process...")
        server_proc.send_signal(signal.SIGINT)
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
        print("Server shutdown complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Zeno — HTTP Decision API Server Integration Test
Tests the /health and /v1/systemone endpoints against the specs/decision-api.md contract.
"""

import json
import os
import signal
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

        # Test 4: Validation error: unsupported step_cap > 1 (400 Bad Request)
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

        # Test 5: Validation error: fewer than 2 candidates
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

        # Test 6: Real Decision Query: POST /v1/systemone
        print("\n--- Test 6: Live Decision Query (POST /v1/systemone) ---")
        query_payload = {
            "query": "Is the system healthy and operational?",
            "schema": {
                "type": "boolean",
                "candidates": [
                    {"token": "Yes", "label": "Yes"},
                    {"token": "No", "label": "No"}
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

        assert data.get("decision") == "Yes", f"Expected decision 'Yes', got {data.get('decision')}"
        assert data.get("steps_executed") == 1, f"Expected 1 step, got {data.get('steps_executed')}"
        assert data.get("uncertain") is False, f"Expected uncertain=False, got {data.get('uncertain')}"
        assert len(data.get("candidates", [])) == 2, f"Expected 2 candidates, got {len(data.get('candidates', []))}"
        assert data["candidates"][0]["label"] == "Yes"
        assert data["candidates"][1]["label"] == "No"
        print(f"✓ Decision matches expected golden judgment ('Yes') with prob {data.get('decision_probability'):.4f}!")

        print("\n=======================================================")
        print(" ALL HTTP SERVER INTEGRATION TESTS PASSED (6/6)")
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

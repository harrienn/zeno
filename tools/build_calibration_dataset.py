#!/usr/bin/env python3
"""Generates the 50-example System One calibration benchmark dataset:
specs/fixtures/calibration_benchmark.json
"""

import json
from pathlib import Path

DATASET = {
    "schema": "zeno.dataset/v1",
    "name": "system_one_decisions_v1",
    "description": "50-example diverse labeled benchmark for System One decision calibration, ECE, and adaptive step dynamics",
    "examples": []
}

# ---------------------------------------------------------------------------
# 1. 20 Boolean Examples (10 Yes, 10 No)
# ---------------------------------------------------------------------------
boolean_cases = [
    # Clear YES cases
    ("bool_01", "infra_health", "Is the database service healthy?", 
     "All replica nodes report 0 replication lag. Heartbeat latency is 1.2ms. CPU utilization is 22%.", "Yes"),
    ("bool_02", "security_auth", "Did this login attempt originate from a suspicious location?",
     "User alice logged in from an IP address in Pyongyang 10 minutes after logging in from Chicago.", "Yes"),
    ("bool_03", "payment_fraud", "Should this credit card transaction be flagged for immediate manual review?",
     "A brand new user account attempted to purchase five $1,000 gift cards at 03:14 UTC with a proxy IP.", "Yes"),
    ("bool_04", "incident_detection", "Does this log indicate an active service outage?",
     "HTTP 500 error rate spiked to 98.4% across all payment gateway endpoints for the last 5 minutes.", "Yes"),
    ("bool_05", "capacity_planning", "Is the storage volume exceeding the critical capacity threshold of 90%?",
     "Volume /dev/nvme0n1p1 reports 482 GB used out of 500 GB total capacity (96.4% utilization).", "Yes"),
    ("bool_06", "network_connectivity", "Is there a network packet loss issue between region us-east and us-west?",
     "Ping probes report 42% packet loss and round-trip time jitter exceeding 850ms.", "Yes"),
    ("bool_07", "sla_monitoring", "Did the service breach the customer 99.9% uptime SLA this month?",
     "Total service downtime recorded during the calendar month was 142 minutes (99.67% uptime).", "Yes"),
    ("bool_08", "deployment_gate", "Should the automated canary deployment be rolled back?",
     "Canary version v2.4.1 has an uncaught NullPointerException rate of 14% on customer requests.", "Yes"),
    ("bool_09", "data_integrity", "Does this database table have orphan foreign key records?",
     "SELECT COUNT(*) FROM order_items WHERE order_id NOT IN (SELECT id FROM orders) returned 1,428.", "Yes"),
    ("bool_10", "resource_leak", "Is this worker process exhibiting signs of a memory leak?",
     "Resident set size (RSS) has monotonically increased from 200MB to 7.8GB over 48 hours without releasing.", "Yes"),

    # Clear NO cases
    ("bool_11", "infra_health", "Is the web server fleet experiencing an outage?",
     "All 48 web instances are responding to health checks with HTTP 200 OK. P95 latency is 18ms.", "No"),
    ("bool_12", "security_auth", "Is this SSH session unauthorized?",
     "Session authenticated via hardware YubiKey FIDO2 token matching on-call engineer Bob's public key.", "No"),
    ("bool_13", "payment_fraud", "Is this routine monthly subscription payment fraudulent?",
     "Regular recurring billing of $14.99 for user account active for 4 years from home ISP address.", "No"),
    ("bool_14", "incident_detection", "Does this single timeout warning require waking up the on-call engineer?",
     "One background analytics batch query timed out after 60 seconds. User traffic is completely unaffected.", "No"),
    ("bool_15", "capacity_planning", "Is the Redis memory cache out of memory?",
     "Redis reports 1.2GB used out of 16GB allocated (7.5% memory usage, 0 evictions).", "No"),
    ("bool_16", "network_connectivity", "Is DNS resolution failing in the production cluster?",
     "Internal CoreDNS latency is 0.4ms across 100,000 queries with 0 SERVFAIL errors.", "No"),
    ("bool_17", "sla_monitoring", "Did the batch invoicing pipeline miss its midnight deadline?",
     "Daily invoicing batch job completed successfully at 23:41 UTC, 19 minutes ahead of schedule.", "No"),
    ("bool_18", "deployment_gate", "Is the release binary corrupted?",
     "SHA-256 checksum of downloaded artifact exactly matches release registry signature.", "No"),
    ("bool_19", "data_integrity", "Are there corrupted rows in the audit log partition?",
     "Cryptographic HMAC verification passed on all 4,500,000 audit log rows.", "No"),
    ("bool_20", "resource_leak", "Are file descriptors being leaked by the connection pool?",
     "Open file descriptor count is steady at 64 out of 65,536 allowed over 7 days.", "No")
]

for qid, cat, q, ctx, gt in boolean_cases:
    DATASET["examples"].append({
        "id": qid,
        "category": cat,
        "query": q,
        "context": ctx,
        "schema_type": "boolean",
        "candidates": [
            {"token": "Yes", "label": "Yes"},
            {"token": "No", "label": "No"}
        ],
        "ground_truth": gt
    })

# ---------------------------------------------------------------------------
# 2. 15 Multi-Class Choice Examples (Severity, Triage, Routing)
# ---------------------------------------------------------------------------
choice_cases = [
    # Severity triage: Low, Medium, High, Critical
    ("choice_01", "incident_severity", "What is the severity of this production incident?",
     "Customer login is completely down globally. Authentication database master has crashed with disk corruption.",
     [{"token": "Low", "label": "Low"}, {"token": "Medium", "label": "Medium"}, {"token": "High", "label": "High"}, {"token": "Critical", "label": "Critical"}], "Critical"),
    ("choice_02", "incident_severity", "What is the severity of this production incident?",
     "Secondary search autocomplete results are delayed by 2 seconds in one regional zone. Main search works fine.",
     [{"token": "Low", "label": "Low"}, {"token": "Medium", "label": "Medium"}, {"token": "High", "label": "High"}, {"token": "Critical", "label": "Critical"}], "Low"),
    ("choice_03", "incident_severity", "What is the severity of this production incident?",
     "Checkout flow error rate is 25% for customers paying with PayPal. Credit card and debit payments work normally.",
     [{"token": "Low", "label": "Low"}, {"token": "Medium", "label": "Medium"}, {"token": "High", "label": "High"}, {"token": "Critical", "label": "Critical"}], "High"),
    ("choice_04", "incident_severity", "What is the severity of this production incident?",
     "Internal admin dashboard for metric viewing is rendering charts slowly. No customer impact.",
     [{"token": "Low", "label": "Low"}, {"token": "Medium", "label": "Medium"}, {"token": "High", "label": "High"}, {"token": "Critical", "label": "Critical"}], "Medium"),
    ("choice_05", "incident_severity", "What is the severity of this production incident?",
     "A minor typo was noticed in the footer copyright year of the public documentation site.",
     [{"token": "Low", "label": "Low"}, {"token": "Medium", "label": "Medium"}, {"token": "High", "label": "High"}, {"token": "Critical", "label": "Critical"}], "Low"),
    ("choice_06", "incident_severity", "What is the severity of this production incident?",
     "All API traffic to European region is dropping 100% of packets due to fiber cut at Frankfurt datacenter.",
     [{"token": "Low", "label": "Low"}, {"token": "Medium", "label": "Medium"}, {"token": "High", "label": "High"}, {"token": "Critical", "label": "Critical"}], "Critical"),

    # Triage domain: Low, Medium, High (Maintenance priority)
    ("choice_07", "maintenance_priority", "Assign maintenance priority for this hardware alert.",
     "Server rack fan unit 3 is running at 70% speed. Temperature is normal at 24C.",
     [{"token": "Low", "label": "Low"}, {"token": "Medium", "label": "Medium"}, {"token": "High", "label": "High"}], "Low"),
    ("choice_08", "maintenance_priority", "Assign maintenance priority for this hardware alert.",
     "Core switch primary power supply failed. Redundant secondary power supply is active and carrying full load.",
     [{"token": "Low", "label": "Low"}, {"token": "Medium", "label": "Medium"}, {"token": "High", "label": "High"}], "Medium"),
    ("choice_09", "maintenance_priority", "Assign maintenance priority for this hardware alert.",
     "RAID controller reports 2 out of 3 parity disks have sustained unrecoverable read errors.",
     [{"token": "Low", "label": "Low"}, {"token": "Medium", "label": "Medium"}, {"token": "High", "label": "High"}], "High"),
    ("choice_10", "maintenance_priority", "Assign maintenance priority for this hardware alert.",
     "Routine air filter replacement scheduled for cooling intake in server hall B.",
     [{"token": "Low", "label": "Low"}, {"token": "Medium", "label": "Medium"}, {"token": "High", "label": "High"}], "Low"),
    ("choice_11", "maintenance_priority", "Assign maintenance priority for this hardware alert.",
     "Chiller coolant loop pressure dropping rapidly; core temperature rising 2 degrees per minute.",
     [{"token": "Low", "label": "Low"}, {"token": "Medium", "label": "Medium"}, {"token": "High", "label": "High"}], "High"),

    # Single-token choice letter: A, B, C with explicit option mapping in context
    ("choice_12", "failure_domain", "Which subsystem is the root cause of this failure?",
     "Queries: SELECT * FROM users WHERE id=? timed out after 30s. Connection pool exhausted with 500 active locks. Options: A) Database, B) Network, C) Compute.",
     [{"token": "A", "label": "A"}, {"token": "B", "label": "B"}, {"token": "C", "label": "C"}], "A"),
    ("choice_13", "failure_domain", "Which subsystem is the root cause of this failure?",
     "BGP route flap between autonomous systems AS15169 and AS3356 causing 80ms route loop and packet drop. Options: A) Database, B) Network, C) Compute.",
     [{"token": "A", "label": "A"}, {"token": "B", "label": "B"}, {"token": "C", "label": "C"}], "B"),
    ("choice_14", "support_routing", "Which team should this support ticket be routed to?",
     "Invoice #9921 charged customer $450 instead of contracted $350 monthly tier. Options: A) Authentication, B) Infrastructure, C) Billing.",
     [{"token": "A", "label": "A"}, {"token": "B", "label": "B"}, {"token": "C", "label": "C"}], "C"),
    ("choice_15", "support_routing", "Which team should this support ticket be routed to?",
     "Customer cannot reset password because OTP SMS is not arriving on international phone numbers. Options: A) Authentication, B) Infrastructure, C) Billing.",
     [{"token": "A", "label": "A"}, {"token": "B", "label": "B"}, {"token": "C", "label": "C"}], "A")
]

for qid, cat, q, ctx, cands, gt in choice_cases:
    DATASET["examples"].append({
        "id": qid,
        "category": cat,
        "query": q,
        "context": ctx,
        "schema_type": "choice",
        "candidates": cands,
        "ground_truth": gt
    })

# ---------------------------------------------------------------------------
# 3. 15 Ordinal Score Examples (Scale 1 to 5)
# ---------------------------------------------------------------------------
score_candidates = [
    {"token": "1", "label": "1"},
    {"token": "2", "label": "2"},
    {"token": "3", "label": "3"},
    {"token": "4", "label": "4"},
    {"token": "5", "label": "5"}
]

score_cases = [
    ("score_01", "customer_satisfaction", "Rate customer satisfaction on a scale from 1 to 5.",
     "The customer said: 'Absolute perfection! Everything worked seamlessly, support answered in 10 seconds. Best software experience ever.'", "5"),
    ("score_02", "customer_satisfaction", "Rate customer satisfaction on a scale from 1 to 5.",
     "The customer said: 'Terrible! Total disaster. System deleted my project files and support refused to help. Demanding an immediate full refund!'", "1"),
    ("score_03", "customer_satisfaction", "Rate customer satisfaction on a scale from 1 to 5.",
     "The customer said: 'Product is good overall and does the job. A few minor UI quirks on mobile, but desktop experience is solid.'", "4"),
    ("score_04", "customer_satisfaction", "Rate customer satisfaction on a scale from 1 to 5.",
     "The customer said: 'Average service. Neither impressive nor broken. Just okay.'", "3"),
    ("score_05", "customer_satisfaction", "Rate customer satisfaction on a scale from 1 to 5.",
     "The customer said: 'Quite frustrating experience. Multiple crashes during export, had to restart 3 times before it worked.'", "2"),
    ("score_06", "customer_satisfaction", "Rate customer satisfaction on a scale from 1 to 5.",
     "The customer said: 'Exceeded all my expectations. Fast, intuitive, elegant design, 10/10 would recommend to everyone.'", "5"),
    ("score_07", "customer_satisfaction", "Rate customer satisfaction on a scale from 1 to 5.",
     "The customer said: 'Unusable garbage. Does not open on Linux at all despite documentation claiming full support.'", "1"),

    ("score_08", "urgency_rating", "Rate the urgency of this infrastructure alert from 1 (lowest) to 5 (highest).",
     "Routine weekly backup completed successfully with 0 warnings.", "1"),
    ("score_09", "urgency_rating", "Rate the urgency of this infrastructure alert from 1 (lowest) to 5 (highest).",
     "Production primary database disk is at 99.8% capacity with 200MB remaining and write failures beginning.", "5"),
    ("score_10", "urgency_rating", "Rate the urgency of this infrastructure alert from 1 (lowest) to 5 (highest).",
     "TLS certificate for primary domain will expire in 2 days and automatic renewal failed.", "4"),
    ("score_11", "urgency_rating", "Rate the urgency of this infrastructure alert from 1 (lowest) to 5 (highest).",
     "Non-critical dev environment nightly snapshot took 45 minutes instead of typical 30 minutes.", "2"),
    ("score_12", "urgency_rating", "Rate the urgency of this infrastructure alert from 1 (lowest) to 5 (highest).",
     "Staging environment API gateway has elevated 502 Bad Gateway responses.", "3"),

    ("score_13", "quality_rating", "Rate the quality of this code snippet from 1 (poor) to 5 (excellent).",
     "Functions are well-documented, clean error handling, 100% unit test coverage, zero memory leaks.", "5"),
    ("score_14", "quality_rating", "Rate the quality of this code snippet from 1 (poor) to 5 (excellent).",
     "All variables named a, b, c; nested loops with complexity O(N^4); hardcoded passwords in plaintext.", "1"),
    ("score_15", "quality_rating", "Rate the quality of this code snippet from 1 (poor) to 5 (excellent).",
     "Reasonable modular structure, passes basic tests, but lacks docstrings and edge-case boundary checks.", "3")
]

for qid, cat, q, ctx, gt in score_cases:
    DATASET["examples"].append({
        "id": qid,
        "category": cat,
        "query": q,
        "context": ctx,
        "schema_type": "score",
        "candidates": score_candidates,
        "ground_truth": gt
    })

out_path = Path("specs/fixtures/calibration_benchmark.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(DATASET, f, indent=2)

print(f"Successfully generated {len(DATASET['examples'])} benchmark examples in {out_path}")

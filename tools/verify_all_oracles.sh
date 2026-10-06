#!/usr/bin/env bash
set -euo pipefail

# Zeno — Automated End-to-End Reference Oracle Verification Suite
# Verifies all golden fixtures in specs/fixtures/*.json against llama.cpp oracle harness.

MODEL_PATH="${1:-models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf}"
LLAMA_EVAL="/hdd/Projects/llama.cpp/build/bin/llama-diffusion-gemma-eval"
ZENO_BIN="./zig-out/bin/zeno"

echo "================================================================="
echo " Zeno <-> llama.cpp Reference Oracle Verification Suite"
echo "================================================================="
echo "Model:       ${MODEL_PATH}"
echo "Oracle:      ${LLAMA_EVAL}"
echo "Engine:      ${ZENO_BIN}"
echo

if [ ! -f "${MODEL_PATH}" ]; then
    echo "Error: Model file not found at ${MODEL_PATH}" >&2
    exit 1
fi

if [ ! -x "${LLAMA_EVAL}" ]; then
    echo "Error: llama-diffusion-gemma-eval not found or not executable at ${LLAMA_EVAL}" >&2
    exit 1
fi

FIXTURES=(
    "specs/fixtures/system_health_boolean.json"
    "specs/fixtures/incident_severity_multiclass.json"
    "specs/fixtures/boundary_256_incident_triage.json"
    "specs/fixtures/maintenance_priority_uncertain.json"
    "specs/fixtures/satisfaction_score_5class.json"
)

TOTAL=${#FIXTURES[@]}
PASSED=0

TMP_DIR=$(mktemp -d /tmp/zeno_verify_XXXXXX)
trap 'rm -rf "${TMP_DIR}"' EXIT

for FIXTURE in "${FIXTURES[@]}"; do
    NAME=$(basename "${FIXTURE}" .json)
    echo "-----------------------------------------------------------------"
    echo "Evaluating Fixture: ${NAME} (${FIXTURE})"
    echo "-----------------------------------------------------------------"

    P_FILE="${TMP_DIR}/${NAME}_p.i32"
    C_FILE="${TMP_DIR}/${NAME}_c.i32"
    OUT_GPU="${TMP_DIR}/${NAME}_gpu.bin"

    # 1. Export exact prompt and canvas token sequences via Zeno
    "${ZENO_BIN}" dump-fixture "${MODEL_PATH}" "${FIXTURE}" "${P_FILE}" "${C_FILE}"

    # 2. Execute reference oracle forward pass via llama.cpp
    echo "Running llama-diffusion-gemma-eval (CUDA offloaded NGL=99)..."
    NGL=99 "${LLAMA_EVAL}" "${MODEL_PATH}" "${P_FILE}" "${C_FILE}" "${OUT_GPU}" > /dev/null 2>&1

    # 3. Verify exact parity and tolerances against fixture record
    python3 tools/verify_oracle_parity.py "${OUT_GPU}" "${FIXTURE}"
    echo "✓ Fixture '${NAME}' passed reference parity check."
    echo
    PASSED=$((PASSED + 1))
done

echo "================================================================="
echo " Verification Complete: ${PASSED}/${TOTAL} Fixtures Passed Successfully"
echo "================================================================="

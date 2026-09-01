#!/usr/bin/env bash
set -Eeuo pipefail

# Unattended, fail-fast executor for the R3 D12/D13 formal run.
# This file orchestrates only frozen runner commands.  It never reads or
# interprets F1, CPD, gain, passline, or acceptance values.

umask 077

REPO_ROOT="/home/lmy/iot-device-classification"
PYTHON_BIN="/home/lmy/anaconda3/envs/iotcls/bin/python"
RUNNER_REL="code/scripts/analysis/unsw_test1.py"
RUNNER="$REPO_ROOT/$RUNNER_REL"
LAUNCHER_REL="code/scripts/analysis/run_unsw_test1_server.sh"
DISCUSSION_PATH="$REPO_ROOT/docs/CROSS_LINE_DISCUSSION_20260830.md"
AUTHORIZED_COMMIT="f29486baa67eba098065dcd41338f6ea40d13e2b"

A_ROOT="/tmp/unsw_test1-R3-A"
B_ROOT="/tmp/unsw_test1-R3-B"
CONTROL_ROOT="/tmp/iotcls-unsw-test1-R3-control"
CANONICAL_ROOT="$REPO_ROOT/results/unsw_test1"
MPL_ROOT="/tmp/iotcls-unsw-test1-mpl"

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR="$MPL_ROOT"

# The experiment has no network path.  Clear inherited proxy configuration in
# both common casings before Python imports optional model packages.
for proxy_name in \
    ALL_PROXY HTTPS_PROXY HTTP_PROXY FTP_PROXY NO_PROXY \
    all_proxy https_proxy http_proxy ftp_proxy no_proxy \
    BUNDLE_HTTPS_PROXY BUNDLE_HTTP_PROXY DOCKER_HTTPS_PROXY DOCKER_HTTP_PROXY \
    REQUESTS_CA_BUNDLE CURL_CA_BUNDLE SSL_CERT_FILE PIP_PROXY \
    NPM_CONFIG_PROXY NPM_CONFIG_HTTP_PROXY NPM_CONFIG_HTTPS_PROXY \
    npm_config_proxy npm_config_http_proxy npm_config_https_proxy \
    WS_PROXY WSS_PROXY ws_proxy wss_proxy \
    YARN_HTTP_PROXY YARN_HTTPS_PROXY \
    CODEX_NETWORK_PROXY_ACTIVE CODEX_NETWORK_PROXY_BROKERED_CREDENTIALS \
    CODEX_NETWORK_PROXY_CREDENTIAL_BROKER_ACTIVE; do
    unset "$proxy_name" || true
done

phase="preflight"
child_pids=()

log() {
    printf '%s phase=%s %s\n' "$(date --iso-8601=seconds)" "$phase" "$*"
}

die() {
    log "status=FAIL reason=$*"
    exit 1
}

terminate_children() {
    local pid
    for pid in "${child_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${child_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    child_pids=()
}

record_exit() {
    local rc=$?
    terminate_children
    if [[ -d "$CONTROL_ROOT" ]]; then
        {
            printf 'exit_code=%s\n' "$rc"
            printf 'final_phase=%s\n' "$phase"
            printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
            printf 'execution_head=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
            printf 'authorized_commit=%s\n' "$AUTHORIZED_COMMIT"
            printf 'launcher_commit=%s\n' "${D12_R3_LAUNCHER_COMMIT:-unset}"
            printf 'launcher_sha256=%s\n' "${D12_R3_LAUNCHER_SHA256:-unset}"
            printf 'execution_head_expected=%s\n' "${D12_R3_EXECUTION_HEAD:-unset}"
        } > "$CONTROL_ROOT/final_status.env"
    fi
    if [[ "$rc" -eq 0 ]]; then
        log "status=COMPLETE"
    else
        log "status=FAIL exit_code=$rc"
    fi
}

verify_proxy_environment_cleared() {
    local proxy_name
    for proxy_name in \
        ALL_PROXY HTTPS_PROXY HTTP_PROXY FTP_PROXY NO_PROXY \
        all_proxy https_proxy http_proxy ftp_proxy no_proxy \
        BUNDLE_HTTPS_PROXY BUNDLE_HTTP_PROXY DOCKER_HTTPS_PROXY DOCKER_HTTP_PROXY \
        REQUESTS_CA_BUNDLE CURL_CA_BUNDLE SSL_CERT_FILE PIP_PROXY \
        NPM_CONFIG_PROXY NPM_CONFIG_HTTP_PROXY NPM_CONFIG_HTTPS_PROXY \
        npm_config_proxy npm_config_http_proxy npm_config_https_proxy \
        WS_PROXY WSS_PROXY ws_proxy wss_proxy \
        YARN_HTTP_PROXY YARN_HTTPS_PROXY \
        CODEX_NETWORK_PROXY_ACTIVE CODEX_NETWORK_PROXY_BROKERED_CREDENTIALS \
        CODEX_NETWORK_PROXY_CREDENTIAL_BROKER_ACTIVE; do
        if printenv "$proxy_name" >/dev/null 2>&1; then
            die "proxy_variable_still_set_$proxy_name"
        fi
    done
}

verify_ip_sockets_blocked() {
    "$PYTHON_BIN" -c '
import socket

for family, name in ((socket.AF_INET, "AF_INET"), (socket.AF_INET6, "AF_INET6")):
    handle = None
    try:
        handle = socket.socket(family, socket.SOCK_STREAM)
    except OSError:
        continue
    finally:
        if handle is not None:
            handle.close()
    raise SystemExit(f"network isolation missing: {name} socket creation succeeded")
'
}

verify_r3_authorization_block() {
    "$PYTHON_BIN" -c '
from pathlib import Path
import sys

discussion, implementation, launcher_commit, launcher_sha256 = sys.argv[1:]
expected = (
    "R3_RUN_AUTHORIZED\n"
    f"implementation_commit: {implementation}\n"
    f"launcher_commit: {launcher_commit}\n"
    f"launcher_sha256: {launcher_sha256}"
)
if expected not in Path(discussion).read_text(encoding="utf-8"):
    raise SystemExit("exact R3_RUN_AUTHORIZED block is absent")
' "$DISCUSSION_PATH" "$AUTHORIZED_COMMIT" \
        "$D12_R3_LAUNCHER_COMMIT" "$D12_R3_LAUNCHER_SHA256"
}

preflight() {
    local authorization_scope="$1"
    [[ -n "${D12_R3_LAUNCHER_COMMIT:-}" ]] || die "D12_R3_LAUNCHER_COMMIT_missing"
    [[ -n "${D12_R3_LAUNCHER_SHA256:-}" ]] || die "D12_R3_LAUNCHER_SHA256_missing"
    [[ -n "${D12_R3_EXECUTION_HEAD:-}" ]] || die "D12_R3_EXECUTION_HEAD_missing"
    [[ "$D12_R3_LAUNCHER_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "launcher_commit_not_full_hash"
    [[ "$D12_R3_LAUNCHER_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "launcher_sha256_not_full_hash"
    [[ "$D12_R3_EXECUTION_HEAD" =~ ^[0-9a-f]{40}$ ]] || die "execution_head_not_full_hash"

    verify_proxy_environment_cleared
    verify_ip_sockets_blocked

    local current_hash committed_hash
    current_hash="$(sha256sum "$REPO_ROOT/$LAUNCHER_REL" | awk '{print $1}')"
    committed_hash="$(
        git -C "$REPO_ROOT" show "$D12_R3_LAUNCHER_COMMIT:$LAUNCHER_REL" \
            | sha256sum | awk '{print $1}'
    )"
    [[ "$current_hash" == "$D12_R3_LAUNCHER_SHA256" ]] || die "launcher_worktree_hash_mismatch"
    [[ "$committed_hash" == "$D12_R3_LAUNCHER_SHA256" ]] || die "launcher_commit_hash_mismatch"
    [[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$D12_R3_EXECUTION_HEAD" ]] \
        || die "execution_head_mismatch"

    if [[ "$authorization_scope" == "formal" ]]; then
        verify_r3_authorization_block
    elif [[ "$authorization_scope" != "review" ]]; then
        die "unknown_authorization_scope_$authorization_scope"
    fi

    [[ -x "$PYTHON_BIN" ]] || die "canonical_python_missing"
    [[ -f "$RUNNER" ]] || die "runner_missing"
    # GNU nproc honors OMP_NUM_THREADS in this environment; use the online
    # host count so the frozen per-process OMP cap cannot falsify this gate.
    [[ "$(getconf _NPROCESSORS_ONLN)" -ge 24 ]] || die "fewer_than_24_cpus"

    local available_mem_kb available_disk_kb
    available_mem_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    available_disk_kb="$(df -Pk /tmp | awk 'NR == 2 {print $4}')"
    [[ "$available_mem_kb" -ge 52428800 ]] || die "less_than_50GiB_memory_available"
    [[ "$available_disk_kb" -ge 104857600 ]] || die "less_than_100GiB_tmp_available"

    local feature_files meta_files
    feature_files="$(find "$REPO_ROOT/results/unsw_features_full" -maxdepth 1 -type f -name 'features_day_*.csv' | wc -l)"
    meta_files="$(find "$REPO_ROOT/results/unsw_features_full" -maxdepth 1 -type f -name 'features_day_*.run_meta.json' | wc -l)"
    [[ "$feature_files" -eq 20 ]] || die "feature_file_count_$feature_files"
    [[ "$meta_files" -eq 20 ]] || die "run_meta_file_count_$meta_files"
    [[ -f "$REPO_ROOT/dataset/unsw/device_mac_map.csv" ]] || die "mac_map_missing"

    local target
    for target in "$A_ROOT" "$B_ROOT" "$CONTROL_ROOT" "$CANONICAL_ROOT"; do
        [[ ! -e "$target" ]] || die "preexisting_target_$target"
    done

    "$PYTHON_BIN" -c \
        "import sys; sys.path.insert(0, '$REPO_ROOT/code/scripts/analysis'); import unsw_test1 as u; u.runtime_provenance(u.validate_run_authorization('$AUTHORIZED_COMMIT'))"

    log "status=PASS scope=$authorization_scope execution_head=$(git -C "$REPO_ROOT" rev-parse HEAD) launcher_sha256=$current_hash network_isolation=AF_UNIX_only"
}

run_arm() {
    local arm="$1"
    local root="$2"
    local shard_index completed rc
    local shards=()

    phase="${arm}_shards"
    child_pids=()
    for shard_index in 0 1 2 3 4 5; do
        shards+=("$root/shard-$shard_index")
        "$PYTHON_BIN" "$RUNNER" \
            --feature-root results/unsw_features_full \
            --mac-map dataset/unsw/device_mac_map.csv \
            --output-root "$root/shard-$shard_index" \
            --n-jobs 4 \
            --shard-index "$shard_index" \
            --shard-count 6 \
            --authorized-commit "$AUTHORIZED_COMMIT" &
        child_pids+=("$!")
        log "status=START shard=$shard_index pid=$!"
    done

    completed=0
    while [[ "$completed" -lt 6 ]]; do
        if wait -n; then
            completed=$((completed + 1))
            log "status=SHARD_EXIT_ZERO completed=$completed/6"
        else
            rc=$?
            log "status=SHARD_EXIT_NONZERO exit_code=$rc action=terminate_remaining"
            terminate_children
            return "$rc"
        fi
    done
    child_pids=()

    phase="${arm}_merge"
    log "status=START"
    "$PYTHON_BIN" "$RUNNER" \
        --merge-shards "${shards[@]}" \
        --output-root "$root/packet"
    log "status=PASS"
}

main() {
    cd "$REPO_ROOT"
    case "${1:-}" in
        --review-preflight-only)
            [[ "$#" -eq 1 ]] || die "review_preflight_argument_count"
            preflight "review"
            return 0
            ;;
        --preflight-only)
            [[ "$#" -eq 1 ]] || die "formal_preflight_argument_count"
            preflight "formal"
            return 0
            ;;
        "")
            [[ "$#" -eq 0 ]] || die "formal_argument_count"
            preflight "formal"
            ;;
        *)
            die "unknown_argument_$1"
            ;;
    esac

    mkdir -p "$CONTROL_ROOT"
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap record_exit EXIT
    {
        printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'execution_head=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
        printf 'authorized_commit=%s\n' "$AUTHORIZED_COMMIT"
        printf 'launcher_commit=%s\n' "$D12_R3_LAUNCHER_COMMIT"
        printf 'launcher_sha256=%s\n' "$D12_R3_LAUNCHER_SHA256"
        printf 'network_policy=IPAddressDeny_any_RestrictAddressFamilies_AF_UNIX_proxy_variables_cleared\n'
        printf 'restart_policy=none\n'
    } > "$CONTROL_ROOT/launch_record.env"

    run_arm "A" "$A_ROOT"
    run_arm "B" "$B_ROOT"

    phase="compare"
    log "status=START"
    "$PYTHON_BIN" "$RUNNER" \
        --compare-runs "$A_ROOT/packet" "$B_ROOT/packet"
    log "status=PASS"

    phase="publish"
    log "status=START"
    "$PYTHON_BIN" "$RUNNER" \
        --publish-canonical "$A_ROOT/packet" "$B_ROOT/packet" \
        --output-root results/unsw_test1
    log "status=PASS"

    phase="complete"
}

main "$@"

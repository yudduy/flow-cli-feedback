---
name: mithril-flow
description: >
  Best practices for using the Mithril Flow CLI and SDK to provision GPUs, run training jobs,
  manage instances, and debug workloads. Use this skill whenever the user is working with
  `flow` commands, Mithril GPU instances, Flow SDK, batch jobs, distributed training,
  InfiniBand clusters, or any task involving `flow submit`, `flow instance`, `flow ssh`,
  `flow logs`, `flow dev`, or `flow volume`. Also trigger when the user is debugging
  a stuck or provisioning instance, choosing between GPU types (A100, H100, H200, B200),
  checking job status, optimizing cost, or orchestrating multi-node training on Mithril.
---

# Mithril Flow — Complete Reference

## MANDATORY: Cost-Optimized Provisioning

**NEVER hardcode a GPU type or price.** Spot prices swing 100x+ within hours (A100: $0.01 → $32/hr same day). Always check live pricing, pick the cheapest that works, and bid smart.

### The Workflow (agents MUST follow for every provisioning)

```bash
# Step 1: Get live pricing as JSON
PRICING=$(flow pricing --json)

# Step 2: Parse top_rows — already sorted cheapest first
# Structure: top_rows[].{gpu, region, current_fmt, cap_med_by_count.{N: bid}, avail_counts[]}
echo "$PRICING" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for row in data['top_rows']:
    gpu = row['gpu']
    region = row['region']
    price = row['current_fmt']
    counts = row['avail_counts']
    bids = row['cap_med_by_count']
    print(f'{gpu:6s} {region:20s} {price:>8s}  counts={counts}  bid_caps={bids}')
"

# Step 3: Pick cheapest GPU that meets VRAM requirements (see table below)
# Step 4: Use cap_med_by_count as bid cap — this is the recommended medium bid
# Step 5: Provision

flow instance create \
  -i <instance_type> \
  -r <region> \
  -m <cap_med_bid> \
  -n <name> \
  --json
```

### JSON Structure Reference

`flow pricing --json` returns:

```json
{
  "top_rows": [                          // ← USE THIS — sorted cheapest first
    {
      "gpu": "b200",
      "region": "us-central5-a",
      "current_fmt": "$0.01",            // current spot price per GPU-hour
      "cap_med_by_count": {              // recommended bid caps by GPU count
        "1": 0.01,                       // ← use as -m flag value
        "8": 0.01
      },
      "avail_counts": [1, 8]            // available configurations
    }
  ],
  "raw_listings": [                      // all instances with exact availability
    {
      "region": "...",
      "gpu_type": "...",
      "gpu_count": 8,
      "price_per_hour": 1.0,
      "instance_type": "h100-80gb.sxm.8x",
      "available_quantity": 38
    }
  ],
  "recommendations_by_size": {           // low/med/high bid recommendations
    "h100": {
      "us-central2-a": {
        "8": { "low": 1.15, "med": 1.1, "high": 1.25 }
      }
    }
  }
}
```

### Decision Logic

1. Parse `top_rows` — it's already sorted cheapest first
2. For each option, check: does this GPU meet my VRAM requirements? (see table)
3. Check `avail_counts` — does the GPU count I need exist?
4. First match = cheapest valid option
5. Set `-m` to `cap_med_by_count[gpu_count]` — the platform's recommended bid
6. Set `-r` to the `region` from the same row
7. If nothing meets requirements or everything is too expensive → tell user, don't provision

### Minimum VRAM Requirements (for GPU filtering)

| Workload | Min VRAM per GPU | Min GPUs | Acceptable GPUs |
|---|---|---|---|
| Fine-tuning <=7B | 40 GB | 1 | Any (A100, H100, H200, B200) |
| Fine-tuning 7-30B | 80 GB | 1 | Any (all have >=80GB) |
| Fine-tuning 30-70B | 80 GB | 4-8 | 4xa100, 8xa100, 8xh100, 8xb200 |
| Training from scratch | 80 GB | 8 | 8xa100, 8xh100, 8xb200 |
| Long context >32k tokens | 141+ GB | 1+ | H200, B200 |
| Inference / experiments | 80 GB | 1 | Any |

**Any GPU in "Acceptable" works. Always pick whichever is cheapest RIGHT NOW.**

### Concrete Example

```bash
# Agent needs 1 GPU with 80GB VRAM for a 13B fine-tune experiment
PRICING=$(flow pricing --json)

# Parse: top_rows shows B200 at $0.01, H100 at $1.00, A100 at $4.00
# All have >= 80GB VRAM → all qualify
# B200 is cheapest → use B200
# cap_med_by_count["1"] = 0.01 → bid $0.01
# region = us-central5-a

flow instance create -i 1xb200 -r us-central5-a -m 0.01 -n my-experiment --json
# Saved: $3.99/hr vs A100, $0.99/hr vs H100
```

### Price Volatility Patterns (observed)

- Prices change in real-time — auction model, not fixed pricing
- Overnight / weekends tend to be cheapest
- Afternoon weekdays (US time) tend to be most expensive
- Same GPU can swing $0.01 to $32+ within one day
- **B200 can be cheaper than A100** — never assume newer = more expensive
- Always re-check pricing before each new provision, even in the same session

---

## CRITICAL: SSH Key Configuration

**Verify your SSH key is set up correctly BEFORE provisioning.** This is the #2 agent failure mode.

```bash
# Check: the default key MUST show Local ✓ AND Platform ✓
flow ssh-key list

# If the default key shows Local: - (missing locally), SSH will silently fail.
# Fix: set a key that exists locally as default in ~/.flow/config.yaml
```

The SSH key used for provisioning is set in `~/.flow/config.yaml` under `ssh_keys`. If this key doesn't have a local private key, SSH auth will timeout even though port 22 is open.

---

## CRITICAL: SSH Readiness Race Condition

**`status=running` does NOT mean SSH is ready.** This is the #1 agent failure mode.

Mithril flips status to `running` when the VM gets a network interface and public IP — but SSH daemon may not be listening yet. The sequence after `status=running`:

1. Container runtime still initializing (CUDA drivers registering)
2. NVIDIA container toolkit injecting GPU devices
3. `sshd` starting but not yet bound to port 22
4. SSH key injection into `~/.ssh/authorized_keys` completing

**`started_at` is also unreliable** — it gets set around IP assignment, still ahead of sshd.

### The ONLY reliable signal is a successful SSH connection.

```bash
# CORRECT: Poll for actual SSH readiness (use this pattern ALWAYS)
for i in $(seq 1 20); do
    if timeout 15 flow ssh <name> -- "echo READY" 2>/dev/null; then
        echo "SSH ready after ~$((i * 15))s"
        break
    fi
    echo "Attempt $i/20: SSH not ready, waiting 15s..."
    sleep 15
done

# WRONG: Assuming SSH works because status=running
flow status <name>  # shows "running" — does NOT mean SSH works
flow ssh <name> -- "nvidia-smi"  # may timeout for 2-3 min after status=running
```

### While waiting for SSH, use startup logs (different path, no SSH needed)

```bash
timeout 30 flow logs <name> --source startup 2>&1
```

This reads from the provider's cloud-init log API, not SSH.

### `flow ssh` does NOT proxy — it connects directly

`flow ssh --json <name>` reveals the raw SSH command. It connects directly to the instance's public IP on port 22. There is no Mithril proxy layer. If port 22 is unreachable from your network, `flow ssh` will also fail.

**Diagnostic when SSH fails even after readiness wait:**

```bash
# Check what flow ssh actually does
flow ssh --json <name>
# Test raw connectivity to the IP
timeout 10 nc -zv <ip> 22
# If nc fails, it's a network/firewall issue, not a flow CLI issue
```

---

## Mental Model

Flow has two modes:

| Mode | Command | Best for |
|---|---|---|
| **Infrastructure** | `flow instance create` | Long-running clusters, full SSH control, custom environments |
| **Research** *(preview)* | `flow submit` / `flow dev` | Quick batch jobs, rapid iteration, reproducible experiments |

The underlying stack: **your intent -> Flow Execution Layer -> Mithril GPU auction -> InfiniBand cluster**

GPUs are procured via a **blind second-price auction** — you set a max bid cap and Mithril finds the cheapest available node at or below it. You often pay significantly less than your cap.

---

## GPU Hardware Reference

### Instance Types Available on Mithril

| Instance name | Flow `-i` flag | GPUs | GPU VRAM | CPU cores | RAM | NVMe storage | Interconnect |
|---|---|---|---|---|---|---|---|
| 1x A100 80GB SXM | `1xa100` | 1 x A100 SXM | 80 GB | 26 | 190 GB | 1.75 TB | Ethernet |
| 2x A100 80GB SXM | `2xa100` | 2 x A100 SXM | 80 GB | 52 | 380 GB | 3.5 TB | Ethernet |
| 4x A100 80GB SXM | `4xa100` | 4 x A100 SXM | 80 GB | 102 | 760 GB | 7 TB | Ethernet |
| 8x A100 80GB SXM | `8xa100` | 8 x A100 SXM | 80 GB | 204 | 1,520 GB | 14 TB | **InfiniBand** |
| 8x H100 80GB SXM | `8xh100` | 8 x H100 SXM | 80 GB | 104+ | 1,800+ GB | 6.4+ TB | **InfiniBand** |
| 8x B200 192GB SXM | `8xb200` | 8 x B200 SXM | 192 GB | 232 | 3,584 GB | 2.8 TB | **InfiniBand** |

> **H200 note:** Available via custom short/long-term reservations only — not in the spot pool. Contact support@mithril.ai.

### GPU Selection Decision Tree

```
What matters most?
|
+-- Lowest cost / prototyping / fast spinup
|   -> 1xa100 or 2xa100
|
+-- Fine-tuning < 30B params
|   -> 1xa100 or 2xa100
|
+-- Fine-tuning 30-70B params (full precision)
|   -> 4xa100 or 8xa100
|
+-- Training from scratch (any large model)
|   -> 8xh100  <- sweet spot for training economics
|
+-- Model doesn't fit in 80 GB / long context (> 32k tokens)
|   -> H200 (contact support for reservation)
|
+-- Bleeding-edge throughput / FP4 inference / >100B models
|   -> 8xb200
|
+-- Multiple isolated workloads on one GPU (MIG)
    -> Any A100 or H100 (both support MIG)
```

### Key GPU facts

- **H100 vs A100:** ~2.5-4x faster training; total cost often *lower* on H100 for long runs (finish faster).
- **H200:** Same compute as H100 but 141 GB HBM3e (76% more VRAM). For models that overflow 80 GB or long-context (>32k tokens).
- **B200:** ~2-3x faster than H100 training; 192 GB VRAM; FP4 support. Requires `Ubuntu24.04 2025.10.10` image (CUDA 13.0).

### VM Images

Always specify your image. Default (and recommended) is `Ubuntu24.04 2025.10.10`.

| Image | NVIDIA Driver | CUDA | Notes |
|---|---|---|---|
| `Ubuntu24.04 2025.10.10` | 580.95.05 | 13.0 | **Recommended.** Required for B200. |
| `Ubuntu24.04 2025.08.01` | 570.86.10 | 12.8 | Cannot execute binaries from persistent storage |
| `Ubuntu22.04 2024.07.24` | 535.183.01 | 12.2 | Legacy only |

---

## Instance Lifecycle & Provisioning

### Instance state reference

| Status | `started_at` | Meaning |
|---|---|---|
| `running` | `null` | Provisioning — image pull / startup scripts |
| `running` | `<timestamp>` | Nominally up — SSH *may* be available (see race condition above) |
| `starting` | — | Allocating hardware |
| `preempting` | — | Spot instance being reclaimed |
| `cancelled` | — | Stopped (intentional or preempted) |
| `completed` | — | Job finished cleanly |
| `failed` | — | Startup or workload error |

**Do not cancel and resubmit during provisioning.** Standard images take 3-10 min. Custom/devel images take longer.

### Check readiness (non-SSH)

```bash
# One-time check
flow status --json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for i in data:
    print(i.get('name','?'), '| status=', i.get('status'), '| started=', i.get('started_at'))
"

# Live watch (auto-refreshes)
flow instance list -w
```

### Faster startup tip

Reduce RAM for up to 25% faster provisioning (spot only). Pricing is per GPU-hour — RAM doesn't change cost.

```bash
flow instance create -i 8xh100 --ram 500   # vs default 1800+ GB
```

---

## Getting Logs

`flow logs` opens a persistent SSH tunnel and blocks indefinitely in scripts.

```bash
# ALWAYS use timeout in scripts/agents
timeout 20 flow logs <name> 2>&1 || echo "done"

# Startup logs (uses cloud-init API, NOT SSH — works before SSH is ready)
flow logs <name> --source startup

# Cloud-init log via SSH (requires SSH ready)
flow ssh <name> -- "tail -100 /var/log/cloud-init-output.log"

# Non-blocking stdout snapshot
flow ssh <name> -- "tail -50 ~/output.log 2>&1"
```

> `flow logs` and `flow ssh` hang if SSH isn't ready. Confirm with the polling pattern above first.

---

## SSH Access

```bash
# One-off command
flow ssh <name> -- "nvidia-smi"

# GPU utilization summary
flow ssh <name> -- \
  "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv"

# Runs across ALL nodes in parallel on multi-node cluster
flow ssh <name> -- "hostname && nvidia-smi"

# Interactive shell
flow ssh <name>

# Get raw SSH connection params (useful for debugging)
flow ssh --json <name>

# Skip readiness wait (use cached endpoint)
flow ssh --fast <name>
```

> **Never use `ssh ubuntu@<ip>` directly** — the IP may not be reachable from your network. Always use `flow ssh`.

---

## Cancelling Instances

```bash
# Non-interactive (ALWAYS use this in scripts/agents)
echo y | flow cancel <name>

# Wildcard pattern
flow instance cancel -n "dev-*"

# Regex
flow instance cancel --regex "segtb-tb-vs-.*"

# Always check first
flow instance list
```

---

## Common Workflows

### Multi-node H100 cluster
```bash
flow instance create -i 8xh100 -N 20   # 20 nodes = 160 H100s
```

### Batch training job (Research mode)
```bash
flow submit "python train.py" -i 8xh100
flow submit "python train.py" -i 8xa100   # cheaper for smaller runs
```

### Distributed training via SDK (32 GPUs)
```python
import flow

task = flow.run(
    command="torchrun --nproc_per_node=8 train.py",
    instance_type="8xa100",
    num_instances=4,       # 4 nodes x 8 GPUs = 32 total
    env={"NCCL_DEBUG": "INFO"}
)
```

### Interactive dev loop
```bash
flow dev   # sub-5-second iteration after initial VM config
```

### Mount S3 + persistent volumes
```python
task = flow.run(
    "python analyze.py",
    gpu="a100",
    mounts={
        "/datasets": "s3://ml-bucket/imagenet",
        "/models": "volume://pretrained-models"
    }
)
```

### Persistent storage
```bash
flow volume create -s 10000 -i file   # 10 TB
```

### Run an existing SLURM script
```bash
flow submit job.slurm   # handles #SBATCH directives natively
```

### task.yaml for reproducible jobs
```yaml
workdir: .
resources:
  infra: mithril
  accelerators: H100:8
config:
  mithril:
    max_price_per_hour: 12.00
```

---

## Cost & Bid Management

- Pricing is **per GPU-hour**. RAM doesn't affect price.
- Second-price blind auction — you set a cap, pay market-clearing price (often lower).
- Spot jobs can be preempted; Flow auto-heals and migrates tasks.
- **Prices are volatile.** ALWAYS run `flow pricing` before provisioning. Never rely on cached/memorized prices.

```bash
# Check live prices (MANDATORY before provisioning)
flow pricing         # 7-day history, win rates, recommended bid caps
flow availability    # current snapshot

# Set max price per node (not per GPU)
flow instance create -i 8xh100 -m 10.0   # -m = max price per hour per node
```

**DO NOT use the table below for pricing decisions — it's a rough reference only. Run `flow pricing` for real numbers.**

| GPU | 7-day observed range | Notes |
|---|---|---|
| A100 80GB | $0.01–$32.00/GPU-hr | Most volatile. Can be cheapest or most expensive. |
| H100 80GB | $1.00–$10.00/GPU-hr | 8x only. Stable floor. |
| H200 141GB | $1.00+/GPU-hr | 8x only. Limited availability. |
| B200 192GB | $0.01–$25.00/GPU-hr | Can be absurdly cheap. Fastest GPU. Requires CUDA 13.0 image. |

---

## Multi-Node Distributed Training

InfiniBand is auto-configured on all 8x instances. No manual setup needed.

```python
task = flow.run(
    command="torchrun --nproc_per_node=8 --nnodes=4 train.py",
    instance_type="8xh100",
    num_instances=4,
    env={
        "NCCL_DEBUG": "WARN",               # INFO for debugging, WARN for production
        "NCCL_IB_DISABLE": "0",             # Keep InfiniBand on (default)
        "TORCH_DISTRIBUTED_DEBUG": "DETAIL", # For debugging hangs
    }
)
```

**Rules:**
- Use `8x` instance types for multi-node — InfiniBand only on 8-GPU nodes.
- Mithril groups nodes into IB-connected clusters automatically.
- For >8 nodes, keep `NCCL_DEBUG=WARN` to avoid log flooding.
- Code is synced to all nodes via `--upload-code` (default).

---

## Storage Reference

| Type | Path | Survives termination? | Cost |
|---|---|---|---|
| Ephemeral NVMe SSD | `/mnt/local` | No | Free |
| Persistent volume | `volume://name` | Yes | Charged |
| S3 mount | any path | Yes (remote) | S3 rates |

```bash
flow volume create -s 5000 -i file                             # create 5TB volume
flow instance create -i 8xh100 --volume pretrained:/models     # mount it
```

---

## .flowignore

Flow auto-uploads your working directory at job submission. Always have a `.flowignore`:

```
__pycache__/
*.pyc
.git/
data/
checkpoints/
*.egg-info/
.venv/
node_modules/
```

---

## Debugging Checklist (Agent-Friendly)

Follow this sequence exactly. Do NOT skip steps or assume.

**Step 1 — Check status and started_at:**
```bash
flow status <name> 2>&1
```

**Step 2 — If `started_at` is null:** Wait. Do NOT cancel. Poll every 30s for up to 10 min.

**Step 3 — If `started_at` is set, poll for SSH readiness:**
```bash
for i in $(seq 1 12); do
    if timeout 15 flow ssh <name> -- "echo READY" 2>/dev/null; then
        echo "SSH ready"; break
    fi
    sleep 15
done
```

**Step 4 — If SSH never comes up (3+ min after started_at):**
```bash
# Check raw connectivity
flow ssh --json <name>  # get the IP
timeout 5 nc -zv <ip> 22  # test port reachability
# If nc fails: network/firewall issue. Tell user to check from their terminal.
# If nc succeeds but flow ssh fails: check SSH key with flow ssh-key list
```

**Step 5 — SSH works, check GPU:**
```bash
flow ssh <name> -- "nvidia-smi"
```

**Step 6 — Cancel wrong job:**
```bash
echo y | flow cancel <name>
```

**Step 7 — Instance is preempting:** Flow auto-heals. Wait for restart.

**Step 8 — NCCL / distributed hang:**
```bash
flow submit "python train.py" -i 8xh100 --env NCCL_DEBUG=INFO
```

---

## Key Gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `flow logs` hangs in script | Persistent SSH tunnel | `timeout N flow logs ...` |
| `ssh ubuntu@<ip>` times out | No proxy; direct connection | Use `flow ssh <n>` |
| `status=running`, SSH times out | SSH not ready yet (race condition) | Poll with SSH readiness loop (see above) |
| `status=running`, `started_at=null` | Image pull / startup in progress | Wait 5-10 min, don't cancel |
| SSH times out for 3+ min after started_at | Network/firewall blocking port 22 | `nc -zv <ip> 22` to diagnose |
| Cancel prompt hangs in script | Interactive `[y/N]` | `echo y \| flow cancel ...` |
| Duplicate job in list | Submitted twice | Check `flow instance list` before submitting |
| NCCL hangs at init | IB misconfiguration | `NCCL_DEBUG=INFO`, `NCCL_IB_DISABLE=0` |
| B200 driver errors | Needs CUDA 13.0+ | Use `Ubuntu24.04 2025.10.10` image |

---

## Installation

```bash
# Recommended (uv) — pin to Python 3.13 for stability
uv tool install --python 3.13 flow-compute
flow setup

# Alternative (pipx)
pipx install flow-compute
flow setup
```

Requires Python 3.10+. Tested stable on 3.13. Python 3.14 may have issues with SSH error handling code.

**PATH setup:** If installed via uv, ensure `~/.local/bin` is on PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"  # add to ~/.zshrc
```

**Verify installation:**
```bash
flow --version   # should show 0.3.x+
flow health      # checks connectivity, auth, state sync
flow ssh-key list  # verify default key has Local + Platform checkmarks
```

---

## Further Reading

- Instance specs: https://docs.mithril.ai/compute-and-storage/instance-types-and-specifications
- Flow quickstart: https://docs.mithril.ai/cli-and-sdk/quickstart
- CLI reference: https://docs.mithril.ai/cli-and-sdk/cli-reference
- Issue tracker: https://github.com/mithrilcompute/flow
- Support: support@mithril.ai

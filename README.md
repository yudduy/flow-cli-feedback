# Flow CLI: Why It Wins for AI-Agent GPU Workflows

User feedback from building autonomous AI agent workflows (Claude Code, Codex) on Mithril GPU infrastructure.

## Context

I evaluated both CLIs for automating GPU provisioning and training jobs via AI coding agents:
- **`flow`** — Python-based, 30+ commands, full platform CLI
- **`mithril-cli`** (mcli) — Rust-based, focused instance management, delegates to flow for everything else

Both report the same version (`flow, version 0.3.2`). They share identity — mcli is a native frontend to the same platform.

## Accurate Command Comparison

### mithril-cli (Rust) — what it covers natively

| Command | Features |
|---|---|
| `instance create` | `--json`, `--watch`, `--dry-run`, `--wait` |
| `instance list` | `--json`, `--state` filter, `--all` |
| `instance delete` | Cancel/delete instances |
| `instance info` | `--json`, detailed instance view |
| `instance list-types` | `--json`, `--verbose`, region filter |
| `ssh` | Interactive picker, `--show`, `--no-wait`, multi-node |
| `k8s` | list, info, ssh, update-kubeconfig |
| `[ARGS]...` | **Pass-through to flow CLI for anything else** |

**mcli does instance management well.** It has `--json` output, watch mode, dry-run, and a nice interactive SSH picker. For humans doing simple provisioning, it's solid.

### flow (Python) — the full surface

| Category | Commands |
|---|---|
| **Compute** | `instance`, `grab`, `dev`, `serve`, `reserve` |
| **Jobs** | `submit`, `status`, `logs`, `cancel`, `ssh` |
| **Resources** | `volume`, `ssh-key`, `k8s`, `cluster` (SLURM overlay) |
| **Tools** | `pricing`, `availability`, `ask` (AI-powered), `jupyter`, `colab`, `ports`, `deploy` |
| **More** | `docs`, `upload-code`, `health`, `mount`, `slurm`, `claude`, `init`, `example`, `tutorial` |

**Flow commands that have no mcli equivalent:**
`submit`, `status`, `logs`, `cancel`, `grab`, `dev`, `serve`, `reserve`, `volume`, `ssh-key`, `cluster`, `pricing`, `availability`, `ask`, `jupyter`, `colab`, `ports`, `deploy`, `health`, `mount`, `slurm`, `claude`

That's 22 command groups mcli doesn't have natively (it delegates them via pass-through).

## Why Flow Wins for AI Agents

Agents don't provision one instance and SSH in. They run full lifecycles: provision → wait → deploy → monitor → adjust → teardown. This needs:

### 1. Job lifecycle commands
```
flow submit "python train.py" -i 8xh100    # run job
flow status --json                          # poll
flow logs <name> -f                         # monitor
flow cancel <name>                          # teardown
```
mcli has none of these natively. An agent using mcli would pass-through to flow anyway — so why not just use flow directly?

### 2. Cost intelligence
```
flow pricing --gpu h100                     # market prices
flow availability --gpu b200                # what's available now
flow ask "cheapest H100 in us-west?"        # AI-powered recommendations
```
An autonomous agent can make cost-optimal decisions without human intervention. mcli can't do this.

### 3. Machine-readable output everywhere
Both CLIs have `--json` on instance commands. But flow also has `--json` on `status`, `ssh`, `availability`, and more. Agents parse JSON — the broader the coverage, the more the agent can do without screen-scraping.

### 4. Non-interactive operation
```bash
echo y | flow cancel <name>     # no interactive prompt
flow status --json              # no pretty tables to parse
timeout 20 flow logs <name>     # bounded, won't hang
```

### 5. End-to-end without the web console
With flow, this entire agent workflow works:
```
1. flow availability --json          # check GPU availability
2. flow instance create -i 8xh100   # provision
3. flow status --json                # poll until running
4. flow ssh <name> -- "echo READY"   # poll until SSH ready
5. flow ssh <name> -- "nvidia-smi"   # verify GPUs
6. flow submit "python train.py"     # run the job
7. timeout 20 flow logs <name>       # monitor output
8. echo y | flow cancel <name>       # teardown
```
With mcli, steps 1, 6, 7 require either pass-through to flow or dropping to the web console. The agent can't stay in its loop.

## The Agent Skill I Built

See [`skill/SKILL.md`](./skill/SKILL.md) — a complete Claude Code skill for Flow covering:
- **SSH readiness race condition** — `status=running` does NOT mean SSH is ready (the #1 agent failure mode)
- **GPU selection decision tree** — which instance type for which workload
- **Agent-friendly debugging checklist** — step-by-step, no assumptions
- **Cost/bid management** — auction mechanics, typical price ranges
- **Multi-node distributed training** — InfiniBand, NCCL, torchrun patterns
- **All gotchas encountered** — with symptoms, causes, and fixes

This skill enables fully autonomous GPU workflows. It couldn't be built around mcli alone — there isn't enough native surface for the job lifecycle.

## One Gap: SSH Proxy

`flow ssh` connects directly to the instance's public IP on port 22. No relay/proxy. Port 22 was blocked from my network (confirmed via `nc -zv`). Everything else worked (status, cancel, submit — all HTTP API).

A relay/proxy layer for SSH would make flow work from any network without firewall exceptions. This is the single biggest blocker I hit.

## Bottom Line

**mcli is good for what it does** — fast Rust binary, nice interactive SSH picker, solid instance management with `--json`. For humans doing quick provisioning, it works.

**flow is what agents need** — 30+ commands covering the full lifecycle. Submit, monitor, logs, pricing, volumes, ports, deploy. The surface area makes autonomous GPU workflows possible.

Since mcli delegates to flow for everything beyond instance management anyway, investing in flow as the primary CLI makes sense. It's already the engine — mcli is the dashboard.

# Mithril CLI Feedback: AI-Agent GPU Workflows

User feedback from building autonomous AI agent workflows (Claude Code, Codex) on Mithril GPU infrastructure.

## Context

I used both CLIs for automating GPU provisioning and training jobs via AI coding agents:
- **`mithril-cli`** (mcli) — Rust binary. Native instance management + pass-through to flow for everything else.
- **`flow`** — Python CLI/SDK. The full platform engine (30+ commands).

Both report `flow, version 0.3.2`. mithril-cli delegates to flow for any command it doesn't handle natively — so **mcli is a superset**: everything flow can do, mcli can do via pass-through, plus native Rust implementations for the common operations.

## What mithril-cli Covers Natively (Rust)

| Command | Features |
|---|---|
| `instance create` | `--json`, `--watch`, `--dry-run`, `--wait` |
| `instance list` | `--json`, `--state` filter, `--all` |
| `instance delete` | Cancel/delete instances |
| `instance info` | `--json`, detailed instance view |
| `instance list-types` | `--json`, `--verbose`, region filter |
| `ssh` | Interactive picker, `--show`, `--no-wait`, multi-node |
| `k8s` | list, info, ssh, update-kubeconfig |

These are the hot-path operations — what you do most often. Fast native Rust, good `--json` output, nice interactive SSH picker.

## What Comes Via Flow Pass-Through

Everything else — and it's a lot:

| Category | Commands (via `mithril-cli <args>`) |
|---|---|
| **Jobs** | `submit`, `status`, `logs`, `cancel` |
| **Compute** | `grab`, `dev`, `serve`, `reserve` |
| **Resources** | `volume`, `ssh-key`, `cluster` (SLURM overlay) |
| **Tools** | `pricing`, `availability`, `ask` (AI-powered), `jupyter`, `colab`, `ports`, `deploy` |
| **Ops** | `health`, `mount`, `upload-code`, `slurm`, `claude` |

This is what makes the full agent lifecycle possible — submit, monitor, logs, cost intelligence, storage, port management.

## What Works Well for AI Agents

The combination of native mcli + flow pass-through gives agents everything they need:

```bash
# Full autonomous agent workflow — all via mithril-cli
mithril-cli instance list-types --json       # check what's available (native Rust)
mithril-cli instance create -i 8xh100 --json # provision (native Rust)
mithril-cli instance list --json --state running  # poll status (native Rust)
mithril-cli ssh <name> -- "echo READY"       # poll SSH readiness (native Rust)
mithril-cli ssh <name> -- "nvidia-smi"       # verify GPUs (native Rust)
mithril-cli submit "python train.py"         # run job (flow pass-through)
mithril-cli logs <name>                      # monitor (flow pass-through)
mithril-cli cancel <name>                    # teardown (flow pass-through)
```

**What makes it agent-friendly:**
- `--json` output on native commands (and flow commands that support it)
- Non-interactive operation (`echo y | mithril-cli cancel <name>`)
- Full lifecycle without dropping to web console
- Pass-through means no capability gap vs flow

## The Agent Skill I Built

See [`skill/SKILL.md`](./skill/SKILL.md) — a Claude Code skill for autonomous GPU workflows covering:
- **SSH readiness race condition** — `status=running` ≠ SSH ready (the #1 agent failure mode)
- **GPU selection decision tree** — which instance type for which workload
- **Agent-friendly debugging checklist** — step-by-step, no assumptions
- **Cost/bid management** — auction mechanics, typical price ranges
- **Multi-node distributed training** — InfiniBand, NCCL, torchrun patterns
- **Gotcha table** — symptoms, causes, fixes for every issue I hit

This skill was written against flow commands but works identically via mithril-cli pass-through.

## One Blocker: SSH Proxy

`flow ssh` / `mithril-cli ssh` connects directly to the instance's public IP on port 22. No relay or proxy. Port 22 was blocked from my network (confirmed via `nc -zv <ip> 22`). Everything HTTP-based worked fine (status, cancel, submit).

**This is the single biggest blocker I hit.** A relay/proxy layer for SSH would make Mithril work from any network — corporate firewalls, university networks, coffee shops — without requiring port 22 outbound.

## Suggestions

1. **SSH relay/proxy** — #1 priority. Blocked port 22 = blocked SSH = blocked logs, blocked commands, blocked everything interactive. An HTTPS-tunneled relay would fix this universally.

2. **`--json` on pass-through commands** — Native mcli commands have great `--json` support. Ensuring flow pass-through commands also consistently support `--json` would help agents parse everything uniformly.

3. **Non-interactive defaults for agents** — Some commands prompt for confirmation (cancel). A `--yes` or `--non-interactive` flag across the board would help automated workflows.

4. **Startup readiness signal** — `status=running` with `started_at` set doesn't mean SSH is ready. An explicit `ssh_ready` field in status JSON would eliminate the polling loop every agent has to implement.

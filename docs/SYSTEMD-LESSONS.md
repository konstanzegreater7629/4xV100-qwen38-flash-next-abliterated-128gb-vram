# Two systemd traps that cost us real debugging time

Both were discovered the hard way on this box. They are not vLLM-specific.

## Trap 1 — `Environment=` accumulates; comments don't unset anything

`Environment=` directives from the **main unit and all drop-ins accumulate**;
for the same variable, the last assignment wins. A **commented-out line in a
drop-in does nothing** — if the main unit sets

```ini
[Service]
Environment=NCCL_P2P_LEVEL=NVL
```

and a drop-in later contains

```ini
#Environment=NCCL_P2P_LEVEL=NVL   # "turned off" — NOT!
```

…the effective environment **still has `NCCL_P2P_LEVEL=NVL`**. We intended to
test P2P through the PLX bridge (after enabling `iommu=pt`), wrote the
comment, and unknowingly kept running restricted — until an audit caught it.

**How to actually change it:** assign the variable explicitly in a drop-in
(systemd has no unset for `Environment=`; an empty value sets it empty, which
some programs treat as unset but NCCL does not), or edit the main unit. Then
`systemctl daemon-reload` and restart — and **re-measure** before keeping it.

**How to see the effective config:**

```bash
systemctl cat your.service        # merged unit + drop-ins, in order
systemctl show your.service -p Environment
tr '\0' '\n' < /proc/$(systemctl show -p MainPID --value your.service)/environ | grep NCCL
```

The last one is ground truth: what the running process actually has.

## Trap 2 — the running process is the truth, not the unit file

At E0 we audited the system and found `vllm-flash.service`'s `ExecStart` was
**stale** relative to the live server: missing `--api-key`, missing MTP flags,
missing tool-calling — while the actual process (started long ago, flags since
forgotten) had all of them. On any restart (`Restart=always` makes this a
*when*, not an *if*) the box would have booted the wrong, unauthenticated
configuration.

**Rule:** before touching a long-running service, snapshot what it is *actually
doing*:

```bash
ps -o args= -p $(systemctl show -p MainPID --value your.service)
```

Compare with `systemctl cat`. If they diverge, the process is the spec —
transcribe it into config *first*, restart, verify identical behavior.

This is also why all our launchers read the API key from a single
`EnvironmentFile` with mode 600 instead of hardcoding it in the unit: one
source of truth, rotatable, never in git.

#!/usr/bin/env python3
"""Run a list of independent training jobs across all GPUs of one machine (a rented vast.ai box).

    python tools/vast/runner.py run    tools/vast/jobs_deferred.toml [--per-gpu 3] [--threads 5]
    python tools/vast/runner.py plan   tools/vast/jobs_deferred.toml [--fake-gpus 8x24] [--speed 1.1]
    python tools/vast/runner.py status [--state-dir runs/vast]

Scheduling: longest job first (est_min); each job goes to the GPU with the least committed memory
that still fits it: at most --per-gpu jobs per GPU, the sum of the jobs' vram_gb plus a per-process
CUDA context allowance stays under the card's memory minus --vram-reserve-gb, and the GPU's actual
free memory (nvidia-smi) covers the job. Smaller jobs backfill gaps. Host RAM is budgeted the same
way (ram_gb), and nothing launches while MemAvailable is under --ram-reserve-gb.

Each job runs from the repo root in its own process group with CUDA_VISIBLE_DEVICES set to its GPU,
thread pools capped to --threads (OMP/MKL/OpenBLAS/numba/torch-inductor), and pinned to its own
CPU cores when enough are available. Output: <state-dir>/logs/<job>.log; state per job in
<state-dir>/state/<job>.json (pending/running/done/failed). Re-running `run` skips jobs already done,
so the runner is safe to restart. A job that dies of an out-of-memory error (exit 137/-9, or a CUDA
OOM in its log) is retried once alone on a GPU.

First Ctrl-C (or SIGTERM): stop launching, let running jobs finish. Second: kill them.
Stdlib only; needs Python >= 3.11 (tomllib) -- use the project venv's python.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CTX_GB = 0.6          # per-process CUDA context + allocator slack, added to every job's vram_gb
OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|"
                    r"cudaErrorMemoryAllocation|MemoryError")


# ----------------------------------------------------------------------------------------- jobs
@dataclass
class Job:
    id: str
    name: str
    cmd: list[str]
    est_min: float
    vram_gb: float
    ram_gb: float
    exclusive: bool = False
    attempts: int = 0


def load_jobs(path: Path) -> list[Job]:
    with open(path, "rb") as f:
        spec = tomllib.load(f)
    dflt = spec.get("defaults", {})
    jobs: list[Job] = []
    for j in spec.get("job", []):
        seeds = j.get("seeds")
        for s in (seeds if seeds is not None else [None]):
            cmd = j["cmd"] if s is None else j["cmd"].replace("{seed}", str(s))
            if "{seed}" in cmd:
                raise SystemExit(f"job {j['name']}: cmd has {{seed}} but no seeds list")
            jobs.append(Job(
                id=j["name"] + ("" if s is None else f"-s{s}"), name=j["name"], cmd=shlex.split(cmd),
                est_min=float(j["est_min"]), vram_gb=float(j["vram_gb"]),
                ram_gb=float(j.get("ram_gb", dflt.get("ram_gb", 6.0))),
                exclusive=bool(j.get("exclusive", dflt.get("exclusive", False)))))
    ids = [j.id for j in jobs]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise SystemExit(f"duplicate job ids: {sorted(dup)}")
    return jobs


# --------------------------------------------------------------------------------------- machine
@dataclass
class Gpu:
    index: int
    total_gb: float
    name: str = ""
    jobs: list[str] = field(default_factory=list)       # running job ids
    committed_gb: float = 0.0                           # sum of (vram_gb + CTX_GB) of running jobs
    exclusive: bool = False                             # an exclusive job is running


def detect_gpus(fake: str | None) -> list[Gpu]:
    if fake:                                            # "8x24" -> 8 GPUs of 24 GB
        n, gb = fake.lower().split("x")
        return [Gpu(i, float(gb), "fake") for i in range(int(n))]
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,name",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    gpus = []
    for line in out.stdout.strip().splitlines():
        idx, mem, name = [x.strip() for x in line.split(",", 2)]
        gpus.append(Gpu(int(idx), float(mem) / 1024, name))
    return gpus


def gpu_free_gb() -> dict[int, float]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20, check=True)
        return {int(a): float(b) / 1024 for a, b in
                (l.split(",") for l in out.stdout.strip().splitlines())}
    except Exception:
        return {}


def gpu_proc_mem_mb() -> dict[int, float]:
    """pid -> used GPU memory (best effort; containers often hide pids)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20, check=True)
        res = {}
        for l in out.stdout.strip().splitlines():
            a, b = l.split(",")
            res[int(a)] = float(b)
        return res
    except Exception:
        return {}


def effective_cpus() -> list[int]:
    cpus = sorted(os.sched_getaffinity(0))
    try:                                                # cgroup v2 CPU quota (docker --cpus)
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            cpus = cpus[:max(1, int(q) // int(p))]
    except Exception:
        pass
    return cpus


def mem_gb() -> tuple[float, float]:
    """(MemTotal, MemAvailable) in GB, honouring a cgroup v2 memory limit if one is set."""
    info = {}
    for l in Path("/proc/meminfo").read_text().splitlines():
        k, v = l.split(":")
        info[k] = float(v.split()[0]) / 1024 / 1024
    total, avail = info["MemTotal"], info["MemAvailable"]
    try:
        lim = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if lim != "max":
            used = float(Path("/sys/fs/cgroup/memory.current").read_text()) / 2**30
            total = min(total, float(lim) / 2**30)
            avail = min(avail, total - used)
    except Exception:
        pass
    return total, avail


# ------------------------------------------------------------------------------------- placement
def pick_gpu(job: Job, gpus: list[Gpu], per_gpu: int, reserve_gb: float,
             free_gb: dict[int, float] | None) -> Gpu | None:
    best = None
    for g in gpus:
        if g.exclusive or len(g.jobs) >= per_gpu:
            continue
        if job.exclusive and g.jobs:
            continue
        need = job.vram_gb + CTX_GB
        if g.committed_gb + need > g.total_gb - reserve_gb:
            continue
        if free_gb is not None and g.index in free_gb and free_gb[g.index] < need:
            continue
        if best is None or (g.committed_gb, len(g.jobs)) < (best.committed_gb, len(best.jobs)):
            best = g
    return best


def check_fits(jobs: list[Job], gpus: list[Gpu], reserve_gb: float) -> None:
    cap = max(g.total_gb for g in gpus) - reserve_gb
    bad = [j.id for j in jobs if j.vram_gb + CTX_GB > cap]
    if bad:
        raise SystemExit(f"jobs larger than any GPU ({cap:.1f} GB usable): {bad}")


# ------------------------------------------------------------------------------------------ plan
def cmd_plan(a) -> None:
    jobs = load_jobs(Path(a.jobs))
    gpus = detect_gpus(a.fake_gpus)
    check_fits(jobs, gpus, a.vram_reserve_gb)
    t, events, pending = 0.0, [], sorted(jobs, key=lambda j: -j.est_min)
    timeline: dict[int, list[tuple[float, float, str]]] = {g.index: [] for g in gpus}
    while pending or events:
        launched = True
        while launched:
            launched = False
            for j in list(pending):
                g = pick_gpu(j, gpus, a.per_gpu, a.vram_reserve_gb, None)
                if g is None:
                    continue
                # jobs sharing a GPU slow each other down: model the k-th co-runner with a factor
                dur = j.est_min / a.speed * (1 + a.share_penalty * len(g.jobs))
                g.jobs.append(j.id); g.committed_gb += j.vram_gb + CTX_GB; g.exclusive |= j.exclusive
                heapq.heappush(events, (t + dur, j.id, g.index, j.vram_gb, j.exclusive))
                timeline[g.index].append((t, t + dur, j.id))
                pending.remove(j); launched = True
        if not events:
            break
        t, jid, gi, v, ex = heapq.heappop(events)
        g = next(x for x in gpus if x.index == gi)
        g.jobs.remove(jid); g.committed_gb -= v + CTX_GB
        if ex:
            g.exclusive = False
    if pending:
        print("never placed:", [j.id for j in pending])
    gpu_min = sum(j.est_min for j in jobs) / a.speed
    print(f"{len(jobs)} jobs, {gpu_min / 60:.1f} job-hours at speed {a.speed}; "
          f"{len(gpus)} GPUs x {a.per_gpu} slots, share penalty {a.share_penalty}")
    print(f"predicted wall clock: {t / 60:.2f} h")
    if a.verbose:
        for gi, items in timeline.items():
            print(f"GPU {gi}: " + ", ".join(f"{jid}[{s:.0f}-{e:.0f}]" for s, e, jid in items))


# ------------------------------------------------------------------------------------------- run
@dataclass
class Running:
    job: Job
    gpu: Gpu
    proc: subprocess.Popen
    start: float
    cores: list[int]
    log: Path
    peak_mb: float = 0.0


class Runner:
    def __init__(self, a):
        self.a = a
        self.state_dir = Path(a.state_dir)
        (self.state_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.state_dir / "state").mkdir(parents=True, exist_ok=True)
        self.jobs = load_jobs(Path(a.jobs))
        if a.only:
            pat = re.compile(a.only)
            self.jobs = [j for j in self.jobs if pat.search(j.id)]
        self.gpus = detect_gpus(a.fake_gpus)
        if a.gpus:
            keep = {int(x) for x in a.gpus.split(",")}
            self.gpus = [g for g in self.gpus if g.index in keep]
        check_fits(self.jobs, self.gpus, a.vram_reserve_gb)
        self.cpus = effective_cpus()
        slots = len(self.gpus) * a.per_gpu
        self.threads = a.threads or max(2, min(8, len(self.cpus) // max(1, slots)))
        self.free_cores = list(self.cpus)
        self.python = a.python or str(REPO / ".venv" / "bin" / "python")
        self.running: dict[int, Running] = {}
        self.stopping = False
        self.t0 = time.time()
        self.counts = {"done": 0, "failed": 0}

    # -- state files
    def state_path(self, j: Job) -> Path:
        return self.state_dir / "state" / f"{j.id}.json"

    def read_state(self, j: Job) -> dict:
        p = self.state_path(j)
        return json.loads(p.read_text()) if p.exists() else {}

    def write_state(self, j: Job, **kw) -> None:
        st = self.read_state(j)
        st.update(kw)
        tmp = self.state_path(j).with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1))
        tmp.replace(self.state_path(j))

    # -- launching
    def env_for(self, j: Job, g: Gpu) -> dict:
        t = str(self.threads)
        env = os.environ.copy()
        env.update({
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(g.index),
            "OMP_NUM_THREADS": t, "MKL_NUM_THREADS": t, "OPENBLAS_NUM_THREADS": t,
            "NUMBA_NUM_THREADS": t, "POLARS_MAX_THREADS": t, "RAYON_NUM_THREADS": t,
            "TORCHINDUCTOR_COMPILE_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "BB_RUNNER_JOB": j.id,
        })
        return env

    def launch(self, j: Job, g: Gpu) -> None:
        cores = []
        if len(self.free_cores) >= self.threads:
            cores, self.free_cores = self.free_cores[:self.threads], self.free_cores[self.threads:]
        cmd = [self.python if c == "python" and i == 0 else c for i, c in enumerate(j.cmd)]
        log = self.state_dir / "logs" / f"{j.id}.log"
        j.attempts += 1
        with open(log, "a") as lf:
            lf.write(f"\n==== {time.strftime('%Y-%m-%dT%H:%M:%S')} attempt {j.attempts} gpu {g.index} "
                     f"cores {cores[:1]}..{cores[-1:]} threads {self.threads}\n$ {shlex.join(cmd)}\n")
            lf.flush()
            pre = (lambda: os.sched_setaffinity(0, cores)) if cores else None
            proc = subprocess.Popen(cmd, cwd=REPO, env=self.env_for(j, g), stdout=lf,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True, preexec_fn=pre)
        g.jobs.append(j.id); g.committed_gb += j.vram_gb + CTX_GB; g.exclusive |= j.exclusive
        self.running[proc.pid] = Running(j, g, proc, time.time(), cores, log)
        self.write_state(j, status="running", gpu=g.index, pid=proc.pid, attempts=j.attempts,
                         started=time.strftime("%Y-%m-%dT%H:%M:%S"), cmd=shlex.join(cmd),
                         est_min=j.est_min, vram_gb=j.vram_gb)
        print(f"[{self.clock()}] start {j.id} on GPU {g.index} (attempt {j.attempts}, "
              f"{len(g.jobs)} on GPU, {g.committed_gb:.1f} GB committed)", flush=True)

    def finish(self, r: Running, rc: int) -> Job | None:
        """Record a finished job; returns it if it should be retried."""
        g, j = r.gpu, r.job
        g.jobs.remove(j.id); g.committed_gb -= j.vram_gb + CTX_GB
        if j.exclusive:
            g.exclusive = False
        self.free_cores = sorted(self.free_cores + r.cores)
        wall = (time.time() - r.start) / 60
        tail = ""
        try:
            with open(r.log, "rb") as f:
                f.seek(max(0, os.path.getsize(r.log) - 20000))
                tail = f.read().decode(errors="replace")
        except OSError:
            pass
        oom = rc in (137, -9) or bool(OOM_RE.search(tail))
        status = "done" if rc == 0 else "failed"
        self.write_state(j, status=status, rc=rc, oom=oom, wall_min=round(wall, 2),
                         finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                         observed_peak_vram_mb=r.peak_mb or None)
        print(f"[{self.clock()}] {status} {j.id} rc={rc} {wall:.1f} min (est {j.est_min:.0f})"
              + (" OOM" if oom else ""), flush=True)
        if rc != 0 and oom and j.attempts <= self.a.retries and not self.stopping:
            j.exclusive = True
            print(f"[{self.clock()}] retry {j.id} alone on a GPU", flush=True)
            return j
        self.counts[status] += 1
        return None

    def clock(self) -> str:
        m = int((time.time() - self.t0) // 60)
        return f"{m // 60:d}:{m % 60:02d}"

    # -- main loop
    def run(self) -> int:
        a = self.a
        pending = []
        for j in self.jobs:
            st = self.read_state(j).get("status")
            if st == "done" or (st == "failed" and not a.retry_failed):
                self.counts["done" if st == "done" else "failed"] += 1
                continue
            pending.append(j)
        pending.sort(key=lambda j: -j.est_min)
        print(f"{len(self.jobs)} jobs: {len(pending)} to run, {self.counts['done']} done, "
              f"{self.counts['failed']} failed earlier; {len(self.gpus)} GPUs "
              f"({', '.join(f'{g.index}:{g.name} {g.total_gb:.0f}G' for g in self.gpus)}), "
              f"{a.per_gpu}/GPU, {self.threads} threads/job, {len(self.cpus)} CPUs", flush=True)
        if a.dry_run:
            for j in pending:
                print(f"  {j.id:28s} est {j.est_min:5.0f} min  vram {j.vram_gb:4.1f} GB  "
                      f"{'excl ' if j.exclusive else ''}{shlex.join(j.cmd)}")
            return 0

        def on_signal(sig, frm):
            if not self.stopping:
                self.stopping = True
                print(f"\n[{self.clock()}] stopping: no new launches; signal again to kill "
                      f"{len(self.running)} running jobs", flush=True)
            else:
                for r in self.running.values():
                    try:
                        os.killpg(r.proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

        last_status, idle_since = 0.0, None
        while (pending and not self.stopping) or self.running:
            for pid, r in list(self.running.items()):
                rc = r.proc.poll()
                if rc is None:
                    continue
                del self.running[pid]
                again = self.finish(r, rc)
                if again is not None:
                    pending.insert(0, again)
            if not self.stopping and pending:
                total, avail = mem_gb()
                committed_ram = sum(r.job.ram_gb for r in self.running.values())
                free = None if a.fake_gpus else gpu_free_gb()
                for j in list(pending):
                    if avail < a.ram_reserve_gb + j.ram_gb or \
                            committed_ram + j.ram_gb > total - a.ram_reserve_gb:
                        continue
                    g = pick_gpu(j, self.gpus, a.per_gpu, a.vram_reserve_gb, free)
                    if g is None:
                        continue
                    self.launch(j, g)
                    pending.remove(j)
                    committed_ram += j.ram_gb
                    if free is not None:
                        free[g.index] = free.get(g.index, 0) - (j.vram_gb + CTX_GB)
                    if a.stagger:
                        time.sleep(a.stagger)
                    avail -= j.ram_gb
            if pending and not self.running and not self.stopping:
                # nothing running and nothing fits: report why instead of waiting forever
                idle_since = idle_since or time.time()
                if time.time() - idle_since > 60:
                    total, avail = mem_gb()
                    j = pending[0]
                    print(f"[{self.clock()}] STUCK: cannot place {j.id} (vram {j.vram_gb} GB, ram "
                          f"{j.ram_gb} GB); host RAM total {total:.1f} GB, available {avail:.1f} GB, "
                          f"--ram-reserve-gb {a.ram_reserve_gb}; GPU free "
                          f"{gpu_free_gb() if not a.fake_gpus else 'n/a'}", flush=True)
                    break
            else:
                idle_since = None
            if time.time() - last_status >= a.status_every:
                last_status = time.time()
                mem = gpu_proc_mem_mb()
                for r in self.running.values():
                    r.peak_mb = max(r.peak_mb, mem.get(r.proc.pid, 0.0))
                self.write_summary(pending)
            time.sleep(a.poll)
        self.write_summary(pending)
        self.write_manifest()
        print(f"[{self.clock()}] finished: {self.counts['done']} done, {self.counts['failed']} failed, "
              f"{len(pending)} not started", flush=True)
        return 0 if self.counts["failed"] == 0 and not pending else 1

    def write_summary(self, pending: list[Job]) -> None:
        per_gpu = {g.index: list(g.jobs) for g in self.gpus}
        s = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_min": round((time.time() - self.t0) / 60, 1),
             "running": len(self.running), "pending": len(pending), **self.counts, "per_gpu": per_gpu}
        (self.state_dir / "status.json").write_text(json.dumps(s, indent=1))
        busy = " ".join(f"{i}:{len(v)}" for i, v in per_gpu.items())
        print(f"[{self.clock()}] running {len(self.running)}  pending {len(pending)}  "
              f"done {self.counts['done']}  failed {self.counts['failed']}  | GPU jobs {busy}", flush=True)

    def write_manifest(self) -> None:
        """List the run directories whose result.json appeared during this runner session."""
        runs = REPO / "runs"
        new = sorted(str(p.parent.relative_to(REPO)) for p in runs.glob("*/*/*/result.json")
                     if p.stat().st_mtime >= self.t0 and "vast" not in p.parts)
        (self.state_dir / "new_runs.txt").write_text("\n".join(new) + ("\n" if new else ""))
        print(f"{len(new)} new result directories listed in {self.state_dir / 'new_runs.txt'}")


# ---------------------------------------------------------------------------------------- status
def cmd_status(a) -> None:
    sd = Path(a.state_dir)
    rows = [json.loads(p.read_text()) | {"id": p.stem} for p in sorted((sd / "state").glob("*.json"))]
    if (sd / "status.json").exists():
        print(json.loads((sd / "status.json").read_text()))
    for r in sorted(rows, key=lambda r: (r.get("status", ""), r["id"])):
        print(f"{r['id']:28s} {r.get('status', '?'):8s} gpu {r.get('gpu', '-')!s:2s} "
              f"wall {r.get('wall_min', '-')!s:>6s} est {r.get('est_min', '-')!s:>5s} "
              f"rc {r.get('rc', '-')!s:>4s}{'  OOM' if r.get('oom') else ''}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "plan"):
        p = sub.add_parser(name)
        p.add_argument("jobs", help="TOML job file (see tools/vast/jobs_deferred.toml)")
        p.add_argument("--per-gpu", type=int, default=3, help="max concurrent jobs per GPU")
        p.add_argument("--vram-reserve-gb", type=float, default=1.0,
                       help="memory per GPU kept free of committed job budgets")
        p.add_argument("--fake-gpus", help="e.g. 8x24: pretend GPUs (testing/planning)")
    p = sub.choices["run"]
    p.add_argument("--threads", type=int, default=0, help="threads per job (default: CPUs / slots, 2..8)")
    p.add_argument("--gpus", help="comma-separated GPU indices to use (default: all)")
    p.add_argument("--only", help="regex: run only matching job ids")
    p.add_argument("--state-dir", default=str(REPO / "runs" / "vast"))
    p.add_argument("--python", help="interpreter for cmds starting with 'python' (default: .venv)")
    p.add_argument("--ram-reserve-gb", type=float, default=16.0)
    p.add_argument("--retries", type=int, default=1, help="OOM retries (alone on a GPU)")
    p.add_argument("--retry-failed", action="store_true", help="re-run jobs that failed earlier")
    p.add_argument("--stagger", type=float, default=5.0, help="seconds between launches")
    p.add_argument("--poll", type=float, default=5.0)
    p.add_argument("--status-every", type=float, default=300.0)
    p.add_argument("--dry-run", action="store_true")
    p = sub.choices["plan"]
    p.add_argument("--speed", type=float, default=1.0, help="per-job speed vs the local 4070 Ti")
    p.add_argument("--share-penalty", type=float, default=0.35,
                   help="slowdown per extra job sharing a GPU (0.35: 3 jobs run at 1/1.7 speed)")
    p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("status")
    p.add_argument("--state-dir", default=str(REPO / "runs" / "vast"))
    a = ap.parse_args()
    if a.cmd == "plan":
        cmd_plan(a)
        return 0
    if a.cmd == "status":
        cmd_status(a)
        return 0
    return Runner(a).run()


if __name__ == "__main__":
    sys.exit(main())

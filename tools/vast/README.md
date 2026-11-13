# Running the deferred seeds on a rented vast.ai machine

The deferred runs (TwiBot-22 seeds 1-4 for 13 jobs, plus the RGT TwiBot-20 runs) are about 62 jobs
and 24 job-hours on the local RTX 4070 Ti, each job at most ~1.2 h. They are independent, so one
multi-GPU machine runs them in parallel: `runner.py plan` predicts ~2 h on 8x RTX 4090 with 3 jobs
per GPU (~4.4 h on 4x).

Files:
- `jobs_deferred.toml`: the job list (command, seeds, measured minutes, GPU memory). **Confirm the
  commands against the frozen code before renting.**
- `runner.py`: the scheduler (stdlib only). `run`, `plan`, `status`; see its docstring.
- `push.sh`: code and data to the instance, results back (rsync over the instance's SSH port).
- `data_files.txt`: which `data/` files the jobs need (~40 GB; `push.sh size` prints the total).
- `setup_instance.sh`: run on the instance; installs the env and checks GPUs, CPU/RAM and data.

## Machine
Recommended: 8x RTX 4090 (same GPU generation as the local 4070 Ti, so our Triton kernels and
compiled code behave the same), >= 96 CPU threads, >= 256 GB RAM, >= 120 GB disk, a driver
supporting CUDA >= 13.0 (torch 2.14+cu130 needs driver >= 580). Find offers:

    vastai search offers 'gpu_name=RTX_4090 num_gpus=8 reliability>0.98 cpu_ram>=256 disk_space>=120 cuda_max_good>=13.0 verified=true rentable=true' -o dph

Prefer US/EU hosts (transfer speed). Create it from vast's "NVIDIA CUDA" template
(`vastai/base-image`, CUDA picked to match the host; any Ubuntu image with SSH works, since uv
installs its own Python and the torch wheels bring their own CUDA libraries) with ~150 GB disk:

    vastai create instance <OFFER_ID> --template_hash 72a351f3d7a2eac98fd3af979c79fb03 --disk 150
    vastai show instances          # wait for "running"; then `vastai ssh-url <ID>` gives HOST/PORT Billing runs from creation until
`vastai destroy instance <ID>`; a stopped instance still bills storage.

## Steps
1. Locally: `tools/vast/push.sh code HOST PORT`, then `tools/vast/push.sh data HOST PORT`
   (~40 GB at your upload speed; resumable: re-run to continue).
2. On the instance (`ssh -p PORT root@HOST`, `cd /workspace/bot-baselines`):
   `bash tools/vast/setup_instance.sh --tests` -> every GPU passes, data present, tests pass,
   plan printed.
3. Start inside tmux so it survives disconnects:
   `tmux new -s runs '.venv/bin/python tools/vast/runner.py run tools/vast/jobs_deferred.toml 2>&1 | tee -a runs/vast/runner.log'`
   - check: `.venv/bin/python tools/vast/runner.py status`, `nvidia-smi`, `runs/vast/logs/<job>.log`
   - Ctrl-C once: no new launches, running jobs finish. Twice: kill them.
   - restarting the same command skips finished jobs (`--retry-failed` re-runs failed ones).
4. Locally: `tools/vast/push.sh pull HOST PORT` (copies `runs/` back without overwriting local
   files; `runs/vast/new_runs.txt` lists the new result directories), then destroy the instance.

## Runner behaviour
- Longest job first; each job goes to the GPU with the least committed memory that fits it:
  <= `--per-gpu` jobs (3), sum of `vram_gb` + 0.6 GB per process <= card memory - 1 GB, and the
  GPU's actual free memory covers it. `exclusive = true` jobs get a GPU alone.
- Each job: `CUDA_VISIBLE_DEVICES=<gpu>`, thread pools capped (`--threads`, default CPUs/slots,
  2..8), pinned to its own cores, `TORCHINDUCTOR_COMPILE_THREADS=1`. No memrun/systemd needed: the
  machine has the RAM; host RAM is budgeted per job (`ram_gb`) and nothing launches while free
  RAM is under `--ram-reserve-gb` (16).
- A job that dies of OOM (exit 137 or a CUDA OOM in its log) is retried once alone on a GPU.
- `plan --fake-gpus 8x24 --speed 1.1 -v` simulates a machine with the measured minutes.

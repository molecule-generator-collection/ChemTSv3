import os

def configure_threads_from_slurm():
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if not slurm_cpus:
        return
    
    n_threads = slurm_cpus.strip()

    # don't override
    os.environ.setdefault("OMP_NUM_THREADS", n_threads)
    os.environ.setdefault("MKL_NUM_THREADS", n_threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", n_threads)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", n_threads)
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")
    print(
        f"Slurm detected. Assigned 'SLURM_CPUS_PER_TASK={n_threads}' to default thread variables "
        "(OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, NUMEXPR_NUM_THREADS, "
        "OMP_PROC_BIND, OMP_PLACES). This does not override existing values. "
        "CPU count still needs to be manually specified (e.g., `-c 16`)."
    )
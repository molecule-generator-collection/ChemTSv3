# Path setup / Imports
import argparse
import faulthandler
import os
from mpi4py import MPI
from chemtsv3.generator import worker_loop
from chemtsv3.utils import conf_from_yaml, generator_from_conf

def main():
    print("[startup] entering main()", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--yaml_path", type=str, help="Path to the config file (.yaml)")
    parser.add_argument("-l", "--load_dir", type=str, help="Path to the save directory (contains config.yaml and save.gtr)")
    
    parser.add_argument("--max_generations", type=int, help="Only used when loading the generator from the save.")
    parser.add_argument("-t", "--time_limit", type=int, help="Only used when loading the generator from the save.")
    
    args = parser.parse_args()
    
    yaml_path = args.yaml_path
    load_dir = args.load_dir
    
    if yaml_path is None:
        raise ValueError("Please specify 'yaml_path' (-c).")
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print(f"Initializing master...", flush=True)
    else:
        print(f"Initializing worker rank={rank}...", flush=True)

    conf = conf_from_yaml(yaml_path)
    generator_args = conf.setdefault("generator_args", {})
    if "dispatcher_type" in generator_args:
        if "reward_dispatcher_type" not in generator_args:
            generator_args["reward_dispatcher_type"] = generator_args.pop("dispatcher_type")
        else:
            generator_args.pop("dispatcher_type", None)

    if conf["generator_class"] == "AsyncParallelMCTS":
        generator_args["max_inflight"] = comm.Get_size() - 1
        if rank == 0:
            generator_args["reward_dispatcher_type"] = "mpi" # auto-set in this case (no reason to use chemtsv3-mpi otherwise)
        else: # workers
            generator_args["reward_dispatcher_type"] = None
    elif conf["generator_class"] == "DoubleAsyncParallelMCTS":
        if rank == 0:
            _complete_double_async_mpi_args(generator_args, comm.Get_size() - 1)
        else:
            conf["generator_class"] = "MCTS"
            for key in ("max_inflight", "max_reward_inflight", "max_transition_inflight", "max_mpi_inflight", "inflight_type", "reward_dispatcher_type", "transition_dispatcher_type", "transition_loss", "check_interval"):
                generator_args.pop(key, None)
    else:
        raise ValueError("'generator_class' must be 'AsyncParallelMCTS' or 'DoubleAsyncParallelMCTS' for MPI generation.")

    _set_mpi_output_dir(conf, rank)
    generator = generator_from_conf(conf) # called for workers as well (for Node class / reward initialization etc.)
    reward = generator.reward
    
    comm.Barrier()
    if rank ==0:
        print(f"Master initialized", flush=True)
    else:
        print(f"Worker rank={rank} initialized", flush=True)
    
    if rank == 0:
        if yaml_path is not None and load_dir is None:
            print("[master] building generator", flush=True)
            
            # while(yaml_path): comment out when supporting next_yaml_path
            print("[master] starting generation", flush=True)
            generator.generate(time_limit=conf.get("time_limit"), max_generations=conf.get("max_generations"))
            if getattr(generator, "mpi_worker_pool", None) is not None:
                generator.mpi_worker_pool.close()
            elif generator.dispatcher is not None:
                generator.dispatcher.close()

            if "next_yaml_path" in conf:
                generator.logger.warning("'next_yaml_path' is currently not supported for parallel MCTS, and was ignored. Please manually set the root node in the config file for the next generations step." )
            plot_args = conf.get("plot_args", {})
            if not "save_only" in plot_args:
                plot_args["save_only"] = True
            generator.plot(**plot_args)
            generator.analyze()
                    
        elif yaml_path is None and load_dir is not None:
            raise ValueError("Save/load feature is currently not supported for Parallel MCTS.")
        else:
            raise ValueError("Specify one of 'yaml_path' (-c) or 'load_dir' (-l), not both.")
    else: # rank >= 1
        print(f"[worker] rank={rank} entering worker_loop", flush=True)
        worker_loop(reward=reward, transition=generator.transition, comm=comm)

def _complete_double_async_mpi_args(generator_args: dict, worker_count: int):
    inflight_type = generator_args.get("inflight_type", "separate")

    if inflight_type == "mpi":
        generator_args.setdefault("max_mpi_inflight", worker_count)
        return

    if inflight_type == "separate":
        if generator_args.get("reward_dispatcher_type") == "mpi":
            generator_args.setdefault("max_reward_inflight", worker_count)
        if generator_args.get("transition_dispatcher_type") == "mpi":
            generator_args.setdefault("max_transition_inflight", worker_count)
        return

def _set_mpi_output_dir(conf: dict, rank: int):
    base_output_dir = conf.get("output_dir", "generation_results")
    role_dir = "mpi_master" if rank == 0 else f"mpi_worker_{rank}"
    conf["output_dir"] = os.path.join(base_output_dir, role_dir)

if __name__ == "__main__":
    faulthandler.enable()
    main()

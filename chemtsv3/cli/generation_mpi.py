# Path setup / Imports
import argparse
import faulthandler
import os
from mpi4py import MPI
from chemtsv3.generator import AsyncParallelMCTS, MPIRewardDispatcher, worker_loop
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
    if conf["generator_class"] != "AsyncParallelMCTS":
        raise ValueError("'generator_class' must be 'AsyncParallelMCTS' for parallel MCTS.")
    
    conf["generator_args"]["max_inflight"] = comm.Get_size() - 1
    if rank == 0:
        conf["generator_args"]["dispatcher_type"] = "mpi"
        conf["output_dir"] = os.path.join(conf["output_dir"], f"mpi_master")
    else: # workers
        # TODO: set dummy transition if it seems safe
        conf["generator_args"]["dispatcher_type"] = None
        conf["output_dir"] = os.path.join(conf["output_dir"], f"mpi_worker_{rank}")
    generator = generator_from_conf(conf) # should be AsyncParallelMCTS, called for workers as well (for Node class / reward initialization etc.)
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
        worker_loop(reward=reward, comm=comm)

if __name__ == "__main__":
    faulthandler.enable()
    main()
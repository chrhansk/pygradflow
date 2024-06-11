import datetime
import enum
import itertools
import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Process, Queue, cpu_count

import numpy as np

from pygradflow.log import logger
from pygradflow.params import Params
from pygradflow.solver import SolverStatus

run_logger = logging.getLogger(__name__)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")


def solve_instance(instance, params, log_filename, verbose):
    logger.handlers.clear()

    handler = logging.FileHandler(log_filename)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if verbose:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    logging.captureWarnings(True)
    warn_logger = logging.getLogger("py.warnings")
    warn_logger.addHandler(handler)
    warn_logger.setLevel(logging.WARN)

    try:
        np.seterr(divide="raise", over="raise", invalid="raise")
        result = instance.solve(params)
        return (instance, result)
    except Exception as exc:
        logger.error("Error solving %s", instance.name, exc_info=exc)
        return (instance, "error")


def solve_instance_queue(queue, instance, params, log_filename, verbose):
    instance_result = solve_instance(instance, params, log_filename, verbose)
    queue.put(instance_result)


def solve_instance_time_limit(instance, params, log_filename, verbose):
    if params.time_limit == np.inf:
        return solve_instance(instance, params, log_filename, verbose)

    queue = Queue()
    process = Process(
        target=solve_instance_queue,
        args=(queue, instance, params, log_filename, verbose),
    )

    process.start()
    process.join(timeout=(params.time_limit + 10.0))

    if process.is_alive():
        logger.error("Reached timeout, terminating process for %s", instance.name)
        process.terminate()
        logger.error("Terminated process for %s", instance.name)
        return (instance, "timeout")
    else:
        return queue.get()


class Runner(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_instances(self, args):
        raise NotImplementedError()

    def create_params(self, args):
        params = Params()

        for key, attr in params.annotations():
            value = getattr(args, key)
            if isinstance(attr, enum.EnumMeta):
                value = attr[value]
            setattr(params, key, value)

        return params

    def log_filename(self, args, instance):
        return self.output_filename(args, f"{instance.name}.log")

    def solve_instances_sequential(self, instances, args, params):
        verbose = args.verbose
        for instance in instances:
            log_filename = self.log_filename(args, instance)
            yield solve_instance_time_limit(instance, params, log_filename, verbose)

    def solve_instances_parallel(self, instances, args, params):
        verbose = args.verbose

        if args.parallel is True:
            num_procs = cpu_count()
        else:
            num_procs = args.parallel

        run_logger.info("Solving in parallel with up to %d processes", num_procs)

        all_params = itertools.repeat(params)
        all_log_filenames = [
            self.log_filename(args, instance) for instance in instances
        ]
        all_verbose = itertools.repeat(verbose)

        solve_args = zip(instances, all_params, all_log_filenames, all_verbose)

        with ProcessPoolExecutor(num_procs, max_tasks_per_child=1) as pool:

            futures = []

            for solve_arg in solve_args:
                future = pool.submit(solve_instance_time_limit, *solve_arg)
                future.instance = solve_arg[0]
                futures.append(future)

            run_logger.info("Started %d instances", len(futures))

            while True:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)

                for item in done:
                    yield item.result()

                if len(futures) == 0:
                    run_logger.info("Solved all instances, terminating")
                    break

                remaining_future = next(iter(futures))

                run_logger.info(
                    "Finished %d instances, waiting for %d (including %s)",
                    len(done),
                    len(futures),
                    remaining_future.instance.name,
                )

    def solve_instances(self, instances, args):
        # yields sequence of tuples of (instance, result) for each instance
        run_logger.info("Solving %d instances", len(instances))

        params = self.create_params(args)

        if args.parallel is not None:
            yield from self.solve_instances_parallel(instances, args, params)
        else:
            yield from self.solve_instances_sequential(instances, args, params)

    def filter_instances(self, args):
        instances = []

        max_size = args.max_size
        name = args.name

        for instance in self.get_instances(args):
            if max_size is not None and instance.size > max_size:
                continue

            if args.unconstrained and instance.num_cons > 0:
                continue

            if name is not None and name != instance.name:
                continue

            instances.append(instance)

        return instances

    def parser(self):
        import argparse

        parser = argparse.ArgumentParser()

        group = parser.add_argument_group(title="parameters")

        default_params = Params()

        for key, attr in default_params.annotations():
            name = f"--{key}"
            if isinstance(attr, enum.EnumMeta):
                default_value = getattr(default_params, key).name
                group.add_argument(
                    name, default=default_value, type=str, help="Default: %(default)s"
                )
            else:
                default_value = getattr(default_params, key)

                group.add_argument(
                    name,
                    default=default_value,
                    type=attr,
                    help="Default: %(default)s",
                )

        group = parser.add_argument_group(title="runner")

        parser.add_argument("--output", type=str)
        parser.add_argument("--max_size", type=int)
        parser.add_argument("--name", type=str)
        parser.add_argument("--unconstrained", action="store_true")
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--parallel", nargs="?", type=int, const=True)

        return parser

    def output_filename(self, args, filename):
        return os.path.join(args.output, filename)

    def main(self):
        run_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        run_logger.addHandler(handler)

        args = self.parser().parse_args()

        if args.output is None:
            now = datetime.datetime.now().isoformat("T", "seconds")
            args.output = f"output_{self.name}_{now}"

        os.makedirs(args.output, exist_ok=True)

        instances = self.filter_instances(args)

        self.solve(instances, args)

    def create_csv_row(self, args, instance, result):
        info = {
            "instance": instance.name,
            "num_vars": instance.num_vars,
            "num_cons": instance.num_cons,
            "size": instance.size,
        }

        default_props = {
            "iterations": 0,
            "num_accepted_steps": 0,
            "final_scaled_obj": 0.0,
            "final_stat_res": 0.0,
            "final_cons_violation": 0.0,
            "dist_factor": 0.0,
        }

        if result == "timeout":
            return {
                **info,
                "status": "timeout",
                "total_time": args.time_limit,
                **default_props,
            }

        elif result == "error":
            return {**info, "status": "error", "total_time": 0.0, **default_props}
        else:
            return {
                **info,
                "status": SolverStatus.short_name(result.status),
                "total_time": result.total_time,
                "iterations": result.iterations,
                "num_accepted_steps": result.num_accepted_steps,
                "final_scaled_obj": result.final_scaled_obj,
                "final_stat_res": result.final_stat_res,
                "final_cons_violation": result.final_cons_violation,
                "dist_factor": result.dist_factor,
            }

    def solve(self, instances, args):
        import csv

        params = self.create_params(args)

        params.write(self.output_filename(args, "params.yml"))

        filename = self.output_filename(args, "output.csv")

        run_logger.info("Writing results to '%s'", filename)

        fieldnames = [
            "instance",
            "num_vars",
            "num_cons",
            "size",
            "status",
            "total_time",
            "iterations",
            "num_accepted_steps",
            "final_scaled_obj",
            "final_stat_res",
            "final_cons_violation",
            "dist_factor",
        ]

        with open(filename, "w") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            for instance, result in self.solve_instances(instances, args):
                run_logger.info("Finished instance %s", instance.name)
                writer.writerow(self.create_csv_row(args, instance, result))
                output_file.flush()

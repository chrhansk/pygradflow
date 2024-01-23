import datetime
import enum
import itertools
import logging
import os
from abc import ABC, abstractmethod
from multiprocessing import Pool, TimeoutError, cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np

from pygradflow.log import logger
from pygradflow.params import Params
from pygradflow.solver import SolverStatus

run_logger = logging.getLogger(__name__)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")


def try_solve_instance(instance, params, log_filename):
    try:
        handler = logging.FileHandler(log_filename)
        handler.setFormatter(formatter)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logging.captureWarnings(True)
        warn_logger = logging.getLogger("py.warnings")
        warn_logger.addHandler(handler)
        warn_logger.setLevel(logging.WARN)

        def solve():
            return instance.solve(params)

        # No time limit
        if params.time_limit == np.inf:
            return solve()

        # Solve in thread pool so we can
        # await the result
        thread_pool = ThreadPool(1)

        try:
            res = thread_pool.apply_async(solve)
            return res.get(params.time_limit)
        except TimeoutError:
            logger.error("Reached timeout, aborting")
            return "timeout"
    except Exception as exc:
        logger.error("Error solving %s", instance.name)
        logger.exception(exc, exc_info=(type(exc), exc, exc.__traceback__))
        return "error"


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

    def solve(self, instances, args):
        results = []

        run_logger.info("Solving %d instances", len(instances))

        params = self.create_params(args)

        def log_filename(instance):
            return self.output_filename(args, f"{instance.name}.log")

        if args.parallel is not None:
            if args.parallel is True:
                num_procs = cpu_count()
            else:
                num_procs = args.parallel

            run_logger.info("Solving in parallel with up to %d processes", num_procs)

            all_params = itertools.repeat(params)
            all_log_filenames = [log_filename(instance) for instance in instances]

            solve_args = zip(instances, all_params, all_log_filenames)

            with Pool(num_procs, maxtasksperchild=1) as pool:
                results = pool.starmap(try_solve_instance, solve_args)

        else:
            for instance in instances:
                results.append(
                    try_solve_instance(instance, params, log_filename(instance))
                )

        self.write_results(args, params, instances, results)

    def filter_instances(self, args):
        instances = []

        max_size = args.max_size
        name = args.name

        for instance in self.get_instances(args):
            if max_size is not None and instance.size > max_size:
                continue

            if name is not None and name != instance.name:
                continue

            instances.append(instance)

        return instances

    def parser(self):
        import argparse

        parser = argparse.ArgumentParser()

        parser.add_argument("--output", type=str)
        parser.add_argument("--max_size", type=int)
        parser.add_argument("--name", type=str)
        parser.add_argument("--parallel", nargs="?", type=int, const=True)

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

    def write_results(self, args, params, instances, results):
        import csv

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
        ]

        with open(filename, "w") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for instance, result in zip(instances, results):
                info = {
                    "instance": instance.name,
                    "num_vars": instance.num_vars,
                    "num_cons": instance.num_cons,
                    "size": instance.size,
                }

                if result == "timeout":
                    writer.writerow(
                        {
                            **info,
                            "status": "timeout",
                            "total_time": args.time_limit,
                            "iterations": 0,
                            "num_accepted_steps": 0,
                        }
                    )

                elif result == "error":
                    writer.writerow(
                        {
                            **info,
                            "status": "error",
                            "total_time": 0.0,
                            "iterations": 0,
                            "num_accepted_steps": 0,
                        }
                    )
                else:
                    writer.writerow(
                        {
                            **info,
                            "status": SolverStatus.short_name(result.status),
                            "total_time": result.total_time,
                            "iterations": result.iterations,
                            "num_accepted_steps": result.num_accepted_steps,
                        }
                    )

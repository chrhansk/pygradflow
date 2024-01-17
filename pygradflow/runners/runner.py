from abc import ABC, abstractmethod

import datetime
import os
import enum
import logging

from pygradflow.params import Params
from pygradflow.log import logger


run_logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')


class Runner(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def solve_instance(self, instance, params):
        raise NotImplementedError()

    @abstractmethod
    def get_instances(self, args):
        raise NotImplementedError()

    def create_params(self, args):
        params = Params()

        for key, attr in params.annotations():
            value = getattr(args, key)
            if isinstance(attr, enum.EnumType):
                value = attr[value]
            setattr(params, key, value)

        return params

    def solve(self, instances, args):
        results = []

        run_logger.info("Solving %d instances", len(instances))

        params = self.create_params(args)

        def try_solve_instance(instance):
            try:
                filename = self.output_filename(args, f'{instance.name}.log')

                handler = logging.FileHandler(filename)
                handler.setFormatter(formatter)
                logger.handlers.clear()
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)

                return self.solve_instance(instance, params)
            except Exception as exc:
                logger.error("Error solving %s", instance.name)
                logger.exception(exc,
                                 exc_info=(type(exc), exc, exc.__traceback__))
                return None

        if args.parallel:
            from multiprocessing import Pool

            with Pool() as pool:
                results = pool.map(try_solve_instance, instances)

        else:
            for instance in instances:
                results.append(try_solve_instance(instance))

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

        parser.add_argument('--output', type=str)
        parser.add_argument('--max_size', type=int)
        parser.add_argument('--name', type=str)
        parser.add_argument('--parallel', action='store_true')

        group = parser.add_argument_group(title='parameters')

        default_params = Params()

        for key, attr in default_params.annotations():
            name = f'--{key}'
            if isinstance(attr, enum.EnumType):
                default_value = getattr(default_params, key).name
                group.add_argument(name,
                                   default=default_value,
                                   type=str,
                                   help="Default: %(default)s")
            else:
                default_value = getattr(default_params, key)
                group.add_argument(name,
                                   default=default_value,
                                   type=type(attr),
                                   help="Default: %(default)s")

        return parser

    def output_filename(self, args, filename):
        return os.path.join(args.output, filename)

    def main(self):
        run_logger.setLevel(logging.INFO)
        run_logger.addHandler(logging.StreamHandler())

        args = self.parser().parse_args()

        if args.output is None:
            now = datetime.datetime.now().isoformat("T", "seconds")
            args.output = f"output_{self.name}_{now}"

        os.makedirs(args.output, exist_ok=True)

        instances = self.filter_instances(args)

        self.solve(instances, args)

    def write_results(self,
                      args,
                      params,
                      instances,
                      results):
        import csv

        params.write(self.output_filename(args, 'params.yml'))

        filename = self.output_filename(args, 'output.csv')

        run_logger.info("Writing results to '%s'", filename)

        fieldnames = ['instance',
                      'num_vars',
                      'num_cons',
                      'size',
                      'status',
                      'total_time',
                      'iterations',
                      'num_accepted_steps']

        with open(filename, 'w') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for (instance, result) in zip(instances, results):

                info = {'instance': instance.name,
                        'num_vars': instance.num_vars,
                        'num_cons': instance.num_cons,
                        'size': instance.size}

                if result is None:
                    writer.writerow({**info,
                                     'status': 'error',
                                     'total_time': 0.,
                                     'iterations': 0,
                                     'num_accepted_steps': 0})
                else:
                    writer.writerow({**info,
                                     'status': result.status.short_name,
                                     'total_time': result.total_time,
                                     'iterations': result.iterations,
                                     'num_accepted_steps': result.num_accepted_steps})

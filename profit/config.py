from os import path
import yaml
from collections import OrderedDict
from profit import defaults
from abc import ABC
import warnings

VALID_FORMATS = ('.yaml', '.py')

"""
yaml has to be configured to represent OrderedDict 
see https://stackoverflow.com/questions/16782112/can-pyyaml-dump-dict-items-in-non-alphabetical-order
and https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
"""


def represent_ordereddict(dumper, data):
    value = []

    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)

        value.append((node_key, node_value))

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)


def try_parse(s):
    funcs = [int, float]
    for f in funcs:
        try:
            return f(s)
        except ValueError:
            pass
    return s


def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))


_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
yaml.add_representer(OrderedDict, represent_ordereddict)
yaml.add_constructor(_mapping_tag, dict_constructor)

""" now yaml is configured to handle OrderedDict input and output """


def load_config_from_py(filename):
    """ Load the configuration parameters from a python file into dict. """
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('f', filename)
    f = module_from_spec(spec)
    spec.loader.exec_module(f)
    return {name: value for name, value in f.__dict__.items() if not name.startswith('_')}


class AbstractConfig(ABC):
    """
    Abstract base class with general methods.
    """
    _sub_configs = {}

    def update(self, **entries):
        for name, value in entries.items():
            if hasattr(self, name) or name in map(str.lower, self._sub_configs):
                setattr(self, name, value)
            else:
                message = "Config parameter '{}' for {} configuration may be unused.".format(name, self.__class__.__name__)
                warnings.warn(message)
                setattr(self, name, value)

    def process_entries(self, base_config):
        pass

    def as_dict(self):
        return {name: getattr(self, name)
                if name not in map(str.lower, self._sub_configs.keys()) or getattr(self, name) is None
                else getattr(self, name).as_dict() for name in vars(self)}

    def set_defaults(self, default_dict):
        for name, value in default_dict.items():
            setattr(self, name, value)

    def create_subconfigs(self, **entries):
        entries_lower = {key.lower(): entry for key, entry in entries.items()}
        for name, sub_config in self._sub_configs.items():
            sub_config_label = name.lower()
            if name.lower() in entries_lower:
                entry = entries_lower[name.lower()]
                if isinstance(entry, str):
                    entry = {'class': entry}
                sub = sub_config(**entry)
                self.__setattr__(sub_config_label, sub)
            else:
                self.__setattr__(sub_config_label, sub_config())

    @classmethod
    def register(cls, label):
        def decorator(config):
            if label in cls._sub_configs:
                raise KeyError(f'registering duplicate label {label} for Interface')
            cls._sub_configs[label] = config
            return config
        return decorator


class BaseConfig(AbstractConfig):
    """
    This class and its modular subclasses provide all possible configuration parameters.

    Parts of the Config:
        - base_dir
        - run_dir
        - config_file
        - ntrain
        - variables
        - files
            - input
            - output
        - run
        - fit
            - surrogate
            - save / load
            - fixed_sigma_n
        - active_learning
        - ui

    Base configuration for fundamental parameters.

    Parameters:
        base_dir (str): Base directory.
        run_dir (str): Run directory.
        config_path (str): Path to configuration file.
        files (dict): Paths for input and output files.
        ntrain (int): Number of training samples.
        variables (dict): All variables.
        input (dict): Input variables.
        output (dict): Output variables.
        independent (dict): Independent variables, if the result of the simulation is a vector.
    """

    def __init__(self, base_dir=defaults.base_dir, **entries):
        self.base_dir = path.abspath(base_dir)
        self.run_dir = self.base_dir
        self.config_path = path.join(self.base_dir, defaults.config_file)
        self.ntrain = defaults.ntrain
        self.variables = defaults.variables
        self.input = {}
        self.output = {}
        self.independent = {}
        self.files = defaults.files

        self.update(**entries)  # Update the attributes with given entries.
        self.create_subconfigs(**entries)
        self.process_entries()  # Postprocess the attributes to standardize different user entries.

    def process_entries(self):
        from profit.util.variable_kinds import Variable, VariableGroup

        # Set absolute paths
        self.files['input'] = path.join(self.base_dir, self.files.get('input', defaults.files['input']))
        self.files['output'] = path.join(self.base_dir, self.files.get('output', defaults.files['output']))

        # Variable configuration as dict
        variables = VariableGroup(self.ntrain)
        vars = []
        for k, v in self.variables.items():
            if type(v) in (str, int, float):
                if isinstance(try_parse(v), (int, float)):
                    v = 'Constant({})'.format(try_parse(v))
                vars.append(Variable.create_from_str(k, (self.ntrain, 1), v))
            else:
                vars.append(Variable.create(name=k, size=(self.ntrain,1), **v))
        variables.add(vars)

        self.variables = variables.as_dict
        self.input = {k: v for k, v in self.variables.items()
                         if not any(k in v['kind'].lower() for k in ('output', 'independent'))}
        self.output = {k: v for k, v in self.variables.items()
                       if 'output' in v['kind'].lower()}
        self.independent = {k: v for k, v in self.variables.items()
                            if 'independent' in v['kind'].lower() and v['size'] != (1, 1)}

        # Process sub configurations
        for name in self._sub_configs:
            sub = getattr(self, name.lower())
            if sub:
                sub.process_entries(self)

    @classmethod
    def from_file(cls, filename=defaults.config_file):

        if filename.endswith('.yaml'):
            with open(filename) as f:
                entries = yaml.safe_load(f)
        elif filename.endswith('.py'):
            entries = load_config_from_py(filename)
        else:
            raise TypeError("Not supported file extension .{} for config file.\n"
                            "Valid file formats: {}".format(filename.split('.')[-1], VALID_FORMATS))
        self = cls(base_dir=path.split(filename)[0], **entries)
        self.config_path = path.join(self.base_dir, filename)
        return self


@BaseConfig.register("run")
class RunConfig(AbstractConfig):
    _sub_configs = {}

    def __init__(self, **entries):
        self.set_defaults(defaults.run)
        self.update(**entries)

        for key, sub_config in self._sub_configs.items():
            attr = getattr(self, key.lower())
            if isinstance(attr, str):
                attr = {'class': attr}
            try:
                setattr(self, key.lower(), sub_config._sub_configs[attr['class']](**attr))
            except KeyError:
                setattr(self, key.lower(), sub_config._sub_configs['custom'](**attr))

    def process_entries(self, base_config):
        from profit.util import load_includes

        if isinstance(self.include, str):
            setattr(self, 'include', [self.include])

        for p, include_path in enumerate(self.include):
            if not path.isabs(include_path):
                self.include[p] = path.abspath(path.join(base_config.base_dir, include_path))
        load_includes(self.include)

        if not path.isabs(self.log_path):
            setattr(self, 'log_path', path.abspath(path.join(base_config.base_dir, self.log_path)))

        for key in self._sub_configs:
            getattr(self, key.lower()).process_entries(base_config)


@RunConfig.register("runner")
class RunnerConfig(AbstractConfig):
    _sub_configs = {}

    def __init__(self, **entries):
        pass


@RunnerConfig.register("local")
class LocalRunnerConfig(AbstractConfig):

    def __init__(self, **entries):
        self.set_defaults(getattr(defaults, "run" + "_runner_" + "local"))
        self.update(**entries)


@RunnerConfig.register("slurm")
class SlurmRunnerConfig(AbstractConfig):

    def __init__(self, **entries):
        self.set_defaults(getattr(defaults, "run" + "_runner_" + "slurm"))
        self.update(**entries)

    def process_entries(self, base_config):
        # convert path to absolute path
        if not path.isabs(self.path):
            setattr(self, 'path', path.abspath(path.join(base_config.base_dir, self.path)))
        # check type of 'cpus'
        if (type(self.cpus) is not int or self.cpus < 1) and self.cpus != 'all':
            raise ValueError(f'config option "cpus" may only be a positive integer or "all" and not {self.cpus}')


@RunConfig.register("interface")
class InterfaceConfig(AbstractConfig):
    _sub_configs = {}

    def __init__(self):
        pass


@InterfaceConfig.register("memmap")
class MemmapInterfaceConfig(AbstractConfig):

    def __init__(self, **entries):
        self.set_defaults(getattr(defaults, "run" + "_interface_" + "memmap"))
        self.update(**entries)

    def process_entries(self, base_config):
        # 'path' is relative to base_dir, convert to absolute path
        if not path.isabs(self.path):
            setattr(self, 'path', path.abspath(path.join(base_config.base_dir, self.path)))


@InterfaceConfig.register("zeromq")
class ZeroMQInterfaceConfig(AbstractConfig):

    def __init__(self, **entries):
        self.set_defaults(getattr(defaults, "run" + "_interface_" + "zeromq"))
        self.update(**entries)


@RunConfig.register("pre")
class PreConfig(AbstractConfig):
    _sub_configs = {}

    def __init__(self, **entries):
        pass


@PreConfig.register("template")
class TemplatePreConfig(AbstractConfig):
    def __init__(self, **entries):
        self.set_defaults(getattr(defaults, "run" + "_pre_" + "template"))
        self.update(**entries)

    def process_entries(self, base_config):
        # 'path' is relative to base_dir, convert to absolute path
        if not path.isabs(self.path):
            setattr(self, 'path', path.abspath(path.join(base_config.base_dir, self.path)))

        if isinstance(self.param_files, str):
            setattr(self, 'param_files', [self.param_files])


@RunConfig.register("post")
class PostConfig(AbstractConfig):
    _sub_configs = {}

    def __init__(self, **entries):
        pass


@PostConfig.register("json")
class JsonPostConfig(AbstractConfig):

    def __init__(self, **entries):
        self.set_defaults(getattr(defaults, "run" + "_post_" + "json"))
        self.update(**entries)

    def process_entries(self, base_config):
        pass


@PostConfig.register("numpytxt")
class NumpytxtPostConfig(AbstractConfig):

    def __init__(self, **entries):
        self.set_defaults(getattr(defaults, "run" + "_post_" + "numpytxt"))
        self.update(**entries)

    def process_entries(self, base_config):
        if isinstance(self.names, str):
            setattr(self, 'names', list(base_config.output.keys()) if self.names == 'all' else self.names.split())


@PostConfig.register("hdf5")
class HDF5PostConfig(AbstractConfig):

    def __init__(self, **entries):
        self.set_defaults(getattr(defaults, "run" + "_post_" + "hdf5"))
        self.update(**entries)

    def process_entries(self, base_config):
        pass


@RunnerConfig.register("custom")
@InterfaceConfig.register("custom")
@PreConfig.register("custom")
@PostConfig.register("custom")
class CustomConfig(AbstractConfig):

    def __init__(self, **entries):
        self.update(**entries)

    def process_entries(self, base_config):
        pass


@BaseConfig.register("fit")
class FitConfig(AbstractConfig):

    def __init__(self, **entries):
        from profit.sur import Surrogate
        from profit.sur.gaussian_process import GaussianProcess
        self.set_defaults(defaults.fit)

        if issubclass(Surrogate._surrogates[self.surrogate], GaussianProcess):
            self.set_defaults(defaults.fit_gaussian_process)

        self.update(**entries)

    def process_entries(self, base_config):
        for mode_str in ('save', 'load'):
            mode = getattr(self, mode_str)
            if mode:
                setattr(self, mode, path.abspath(path.join(base_config.base_dir, mode)))
                if self.surrogate not in mode:
                    filepath = mode.rsplit('.', 1)
                    setattr(self, mode_str, ''.join(filepath[:-1]) + f'_{self.surrogate}.' + filepath[-1])

        if self.load:
            setattr(self, 'save', False)

        for enc in getattr(self, 'encoder'):
            cols = enc[1]
            out = enc[2]
            if isinstance(cols, str):
                variables = getattr(base_config, 'output' if out else 'input')
                if cols.lower() == 'all':
                    enc[1] = list(range(len(variables)))
                elif cols.lower() in [v['kind'] for v in variables.values()]:
                    enc[1] = [idx for idx, v in enumerate(variables.values()) if v['kind'].lower() == cols.lower()]
                else:
                    enc[1] = []


@BaseConfig.register("active_learning")
class ALConfig(AbstractConfig):

    def __init__(self, **entries):
        self.set_defaults(defaults.active_learning)
        self.update(**entries)

    def process_entries(self, base_config):
        pass


@BaseConfig.register("ui")
class UIConfig(AbstractConfig):

    def __init__(self, **entries):
        self.plot = defaults.ui['plot']
        self.update(**entries)

    def process_entries(self, base_config):
        pass

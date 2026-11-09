import importlib


def load_policy(specification, config):
    module_name, function_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    params, apply_fn = getattr(module, function_name)(config)
    if not callable(apply_fn):
        raise TypeError("Policy adapter must return a callable apply function")
    return params, apply_fn

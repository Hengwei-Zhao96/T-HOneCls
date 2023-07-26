from importlib import util


def read_config(cfg_path):
    spec = util.spec_from_file_location('config', cfg_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config
    config['meta'].update(config_path=cfg_path)
    return config

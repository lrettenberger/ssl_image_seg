import importlib

def load_class(module_path: str, class_name: str):
    """ Loads a class from a given module and object name.

    Args:
        module_path (str): Defines the module path. Expected to start after DLIP.
        class_name (str): The name of the class to be loaded.

    Raises:
        ModuleNotFoundError: If the given class_name is not found

    Returns:
        (Class): The class.
    """
    module = importlib.import_module(f"DLIP.{module_path}")
    if class_name in dir(module):
        return getattr(module, class_name)
    raise ModuleNotFoundError(f'Cant find class {class_name} in {module_path}.')
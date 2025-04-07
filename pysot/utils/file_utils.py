import yaml


def read_yaml(file_path: str):
    """
    Reads a YAML file and returns its contents as a Python object.

    :param file_path: Path to the YAML file.
    :return: Parsed YAML content as a Python object (e.g., dict or list).
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def write_yaml(data: any, file_path: str):
    """
    Writes a Python object to a YAML file.

    :param data: Python object (e.g., dict or list) to write to the YAML file.
    :param file_path: Path to the YAML file.
    """
    with open(file_path, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False)

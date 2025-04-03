class Filter:
    def __init__(self, flag):
        self.flag = flag

    def filter(self, x):
        return self.flag


class Dummy:
    def __init__(self, *arg, **kwargs):
        pass

    def __getattr__(self, arg):
        def dummy(*args, **kwargs):
            pass

        return dummy


def get_format(logger, level):
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])

        if level == logging.INFO:
            logger.addFilter(Filter(rank == 0))
    else:
        rank = 0
    format_str = "[%(asctime)s-rk{}-%(filename)s#%(lineno)3d] %(message)s".format(rank)
    formatter = logging.Formatter(format_str)
    return formatter


def get_format_custom(logger, level):
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        if level == logging.INFO:
            logger.addFilter(Filter(rank == 0))
    else:
        rank = 0
    format_str = "[%(asctime)s-rk{}-%(message)s".format(rank)
    formatter = logging.Formatter(format_str)
    return formatter


def add_file_handler(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(get_format(logger, level))
    logger.addHandler(fh)

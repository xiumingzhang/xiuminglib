import logging
import logging_colorer # noqa: F401 # pylint: disable=unused-import

logging_warn = logging.WARN


def create_logger(file_abspath, level=logging.INFO):
    logging.basicConfig(level=level)
    logger = logging.getLogger()
    folder_names = file_abspath.split('/')
    for pkg_name in ['xiuminglib', 'commandline']:
        if pkg_name in folder_names:
            thisfile = '/'.join(
                folder_names[
                    folder_names.index(pkg_name):
                ]
            )
            return logger, thisfile
    raise ValueError("Unidentified package")

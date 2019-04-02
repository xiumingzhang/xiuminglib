"""Library setup including setting logging colors, etc."""

import logging
from . import logging_colorer # noqa: F401 # pylint: disable=unused-import

logging_warn = logging.WARN
# so that won't need to import a package just for its constants


def create_logger(file_abspath, level=logging.INFO):
    """Creates a logger for functions in the library.

    Args:
        file_abspath (str): Absolute path to the file that uses the logger.
        level (int, optional): Logging level. Defaults to ``logging.INFO``.

    Returns:
        tuple:
            - **logger** (:class:`logging.Logger`) -- Logger created.
            - **thisfile** (:class:`str`) -- Partial path to the user file, starting from package name.

    Raises:
        ValueError: If the user file is not specified in the ``users`` list.
    """
    users = ['xiuminglib', 'commandline', 'job-dispatcher']
    logging.basicConfig(level=level)
    logger = logging.getLogger()
    folder_names = file_abspath.split('/')
    for user in users:
        if user in folder_names:
            thisfile = '/'.join(
                folder_names[folder_names.index(user):]
            )
            return logger, thisfile
    raise ValueError("Unexpected user")

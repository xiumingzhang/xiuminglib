import logging
import logging_colorer # noqa: F401 # pylint: disable=unused-import

logging_warn = logging.WARN
# so that won't need to import a package just for its constants

users = ['xiuminglib', 'commandline', 'job-dispatcher']


def create_logger(file_abspath, level=logging.INFO):
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

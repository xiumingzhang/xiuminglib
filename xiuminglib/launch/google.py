from os.path import join, abspath
from getpass import getuser
from time import time

from tqdm import tqdm

from ..os import call, makedirs
from .. import const
from ..config import create_logger
logger, thisfile = create_logger(abspath(__file__))


class Launcher():
    def __init__(self, label, borg_user=None, borg_submitters=24,
                 local_ram_fs_dir_size='4096M'):
        self.label = label
        myself = getuser()
        if borg_user is None:
            borg_user = myself
        self.borg_user = borg_user
        self.borg_submitters = borg_submitters
        # Deriving other values
        self.priority = 0 if borg_user == myself else 115
        if borg_user == myself:
            cell = 'qu'
        elif borg_user == 'gcam-eng':
            cell = 'ok'
        elif borg_user == 'gcam-gpu':
            cell = 'is'
        else:
            raise NotImplementedError(borg_user)
        self.cell = cell
        # Requirements
        self.local_ram_fs_dir_size = local_ram_fs_dir_size

    def blaze_run(self, param_dict=None, print_instead=False):
        logger_name = thisfile + '->Launcher:blaze_run()'
        bash_cmd = 'blaze run -c opt %s' % self.label
        if param_dict is not None:
            bash_cmd += ' --'
            for k, v in param_dict.items():
                bash_cmd += ' --%s %s' % (k, v)
        if print_instead:
            logger.name = logger_name
            logger.info("Run:\n%s", bash_cmd)
        else:
            call(bash_cmd)
            # FIXME: sometimes stdout can't catch the printouts (e.g., tqdm)

    def build_for_borg(self):
        bash_cmd = 'rabbit --verifiable build -c opt %s' % self.label
        retcode, _, _ = call(bash_cmd)
        assert retcode == 0, "Build failed"

    def submit_to_borg(self, job_ids, param_dicts, runlocal=False):
        logger_name = thisfile + '->Launcher:submit_to_borg()'
        assert isinstance(job_ids, list) and isinstance(param_dicts, list), \
            "If submitting just one job, make both arguments single-item lists"
        n_jobs = len(job_ids)
        if not self.label.endswith(':main_mpm'):
            logger.name = logger_name
            logger.warning(
                ("Submitting to Borg, but label is not `main_mpm` "
                 "(assumed in generating .borg file)"))
        assert n_jobs == len(param_dicts)
        # If just one job or no parallel workers
        if n_jobs == 1 or self.borg_submitters == 0:
            for job_id, param_dict in tqdm(zip(job_ids, param_dicts),
                                           total=n_jobs):
                self._borg_run((job_id, param_dict, runlocal))
        # Multiple jobs are submitted by a pool of workers
        else:
            from multiprocessing import Pool
            pool = Pool(self.borg_submitters)
            list(tqdm(pool.imap_unordered(
                self._borg_run,
                [(i, x, runlocal) for i, x in zip(job_ids, param_dicts)]
            ), total=len(job_ids)))
            pool.close()
            pool.join()

    def _borg_run(self, args):
        job_id, param_dict, runlocal = args
        borg_f = self.gen_borg_file(job_id, param_dict)
        # Submit
        action = 'runlocal' if runlocal else 'reload'
        # FIXME: runlocal doesn't work for temporary MPM: b/74472376
        bash_cmd = 'borgcfg %s %s --skip_confirmation --borguser %s' \
            % (borg_f, action, self.borg_user)
        call(bash_cmd)

    def gen_borg_file(self, job_id, param_dict):
        borg_file_str = self._format_borg_file_str(job_id, param_dict)
        out_dir = join(const.Dir.tmp, '%f' % time())
        makedirs(out_dir)
        borg_f = join(out_dir, '%s.borg' % job_id)
        with open(borg_f, 'w') as h:
            h.write(borg_file_str)
        return borg_f

    def _format_borg_file_str(self, job_id, param_dict):
        tab = ' ' * 4
        file_str = '''job %s = {
    // What cell should we run in?
    runtime = {
        // 'oregon' // automatically picks a Borg cell with free capacity
        cell = '%s',
    }

    // What packages are needed?
    packages {
        package bin = {
            // A blaze label pointing to a `genmpm(temporal=1)` rule. Borgcfg will
            // build a "temporal MPM" on the fly out of files in the blaze-genfiles
            // directory. See go/temporal-mpm for full documentation.
            blaze_label = '%s',
        }
    }

    // What program are we going to run?
    package_binary = 'bin/main'

    // What command line parameters should we pass to this program?
    args = {
    ''' % (job_id, self.cell, self.label)
        for i, (k, v) in enumerate(param_dict.items()):
            if isinstance(v, str):
                v = "'%s'" % v
            elif isinstance(v, int):
                v = "%d" % v
            elif isinstance(v, list):
                v = "'" + ",".join(str(x) for x in v) + "'"
            else:
                raise TypeError(type(v))
            if i == 0:
                file_str += "%s%s%s = %s,\n" % (tab, tab, k, v)
            else:
                file_str += "%s%s%s%s = %s,\n" % (tab, tab, tab, k, v)
        file_str += '''%s%s}

    // What resources does this program need to run?
    requirements = {
        autopilot = true,
        // ram = 1024M,
        // use_ram_soft_limit = true,
        local_ram_fs_dir { d1 = { size = %s } },
        cpu = 12,
    }

    // How latency-sensitive is this program?
    appclass = {
        type = 'LATENCY_TOLERANT_SECONDARY',
    }

    permissions = {
        user = '%s',
    }

    scheduling = {
        priority = %d,
    ''' % (tab, tab, self.local_ram_fs_dir_size, self.borg_user, self.priority)
        if self.priority == 0:
            file_str += '''%s}
    }''' % tab
        else: # e.g., P115 needs a batch strategy
            file_str += '''
        batch_quota {
            strategy = 'RUN_SOON'
        }
    }
}'''
        return file_str

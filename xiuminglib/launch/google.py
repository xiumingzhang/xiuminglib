from os.path import join, abspath
from getpass import getuser
from time import time
from random import choices
from string import ascii_uppercase, digits

from tqdm import tqdm

from ..os import call, makedirs
from .. import const
from ..config import create_logger
logger, thisfile = create_logger(abspath(__file__))


class Launcher():
    def __init__(self, label, print_instead=False,
                 borg_user=None, borg_submitters=24, borg_cell=None,
                 local_ram_fs_dir_size='4096M'):
        self.label = label
        self.pkg_bin = self._derive_bin()
        self.print_instead = print_instead
        self.myself = getuser()
        if borg_user is None:
            borg_user = self.myself
        self.borg_user = borg_user
        self.borg_submitters = borg_submitters
        # Deriving other values
        self.priority = self._select_priority()
        if borg_cell is None:
            self.cell = self._select_cell()
        else:
            self.cell = borg_cell
        # Requirements
        self.local_ram_fs_dir_size = local_ram_fs_dir_size

    def _derive_bin(self):
        logger_name = thisfile + '->Launcher:_derive_bin()'
        assert ':' in self.label, "Must specify target explicitly"
        pkg_bin = self.label.split(':')[-1]
        if pkg_bin.endswith('_mpm'):
            pkg_bin = pkg_bin[:-4]
        logger.name = logger_name
        logger.warning(("Package binary derived to be `%s`, so make sure "
                        "BUILD is consistent with this"), pkg_bin)
        return pkg_bin

    def _select_priority(self):
        if self.borg_user == self.myself:
            return 0
        return 115

    def _select_cell(self):
        if self.borg_user == self.myself:
            cell = 'qu'
        elif self.borg_user == 'gcam-eng':
            cell = 'ok'
        elif self.borg_user == 'gcam-gpu':
            cell = 'is'
        else:
            raise NotImplementedError(self.borg_user)
        return cell

    def blaze_run(self, blaze_dict=None, param_dict=None):
        logger_name = thisfile + '->Launcher:blaze_run()'
        bash_cmd = 'blaze run -c opt %s' % self.label
        # Blaze parameters
        if blaze_dict is not None:
            for k, v in blaze_dict.items():
                bash_cmd += ' --%s=%s' % (k, v)
        # Job parameters
        if param_dict is not None:
            bash_cmd += ' --'
            for k, v in param_dict.items():
                bash_cmd += ' --%s=%s' % (k, v)
        # To avoid IO permission issues
        bash_cmd += ' --gfs_user=%s' % self.borg_user
        if self.print_instead:
            logger.name = logger_name
            logger.info("To blaze-run the job, run:\n\t%s\n", bash_cmd)
        else:
            call(bash_cmd)
            # FIXME: sometimes stdout can't catch the printouts (e.g., tqdm)

    def build_for_borg(self):
        logger_name = thisfile + '->Launcher:build_for_borg()'
        assert self.label.endswith('_mpm'), \
            "Label must be MPM, because .borg generation assumes temporary MPM"
        bash_cmd = 'rabbit build -c opt %s' % self.label
        # FIXME: --verifiable leads to "MPM failed to find the .pkgdef file"
        if self.print_instead:
            logger.name = logger_name
            logger.info("To build for Borg, run:\n\t%s\n", bash_cmd)
        else:
            retcode, _, _ = call(bash_cmd)
            assert retcode == 0, "Build failed"

    def submit_to_borg(self, job_ids, param_dicts):
        assert isinstance(job_ids, list) and isinstance(param_dicts, list), \
            "If submitting just one job, make both arguments single-item lists"
        n_jobs = len(job_ids)
        assert n_jobs == len(param_dicts)
        # If just one job or no parallel workers or printing only
        if n_jobs == 1 or self.borg_submitters == 0 or self.print_instead:
            for job_id, param_dict in tqdm(
                    zip(job_ids, param_dicts), total=n_jobs,
                    desc="Submitting jobs to Borg"):
                self._borg_run((job_id, param_dict))
        # Multiple jobs are submitted by a pool of workers
        else:
            from multiprocessing import Pool
            pool = Pool(self.borg_submitters)
            list(tqdm(pool.imap_unordered(
                self._borg_run,
                [(i, x) for i, x in zip(job_ids, param_dicts)]
            ), total=len(job_ids), desc="Submitting jobs to Borg"))
            pool.close()
            pool.join()

    def _borg_run(self, args):
        """.borg generation assumed temporary MPM.
        """
        logger_name = thisfile + '->Launcher:_borg_run()'
        job_id, param_dict = args
        borg_f = self.__gen_borg_file(job_id, param_dict)
        # Submit
        action = 'reload'
        # NOTE: runlocal doesn't work for temporary MPM: b/74472376
        bash_cmd = 'borgcfg %s %s --skip_confirmation --borguser %s' \
            % (borg_f, action, self.borg_user)
        if self.print_instead:
            logger.name = logger_name
            logger.info("To launch the job on Borg, run:\n\t%s\n", bash_cmd)
        else:
            call(bash_cmd)

    def __gen_borg_file(self, job_id, param_dict):
        borg_file_str = self.___format_borg_file_str(job_id, param_dict)
        out_dir = join(const.Dir.tmp, '{t}_{s}'.format(
            s=''.join(choices(ascii_uppercase + digits, k=16)),
            t=time()))
        makedirs(out_dir)
        borg_f = join(out_dir, '%s.borg' % job_id)
        with open(borg_f, 'w') as h:
            h.write(borg_file_str)
        return borg_f

    def ___format_borg_file_str(self, job_id, param_dict):
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
    package_binary = 'bin/%s'

    // What command line parameters should we pass to this program?
    args = {
    ''' % (job_id, self.cell, self.label, self.pkg_bin)
        for i, (k, v) in enumerate(param_dict.items()):
            if isinstance(v, str):
                v = "'%s'" % v
            elif isinstance(v, int):
                v = "%d" % v
            elif isinstance(v, float):
                v = "%f" % v
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
        autopilot_params {
            // Let autopilot increase limits past the Borg pickiness limit
            scheduling_strategy = "NO_SCHEDULING_SLO",
        }
        // ram = 1024M,
        // use_ram_soft_limit = true,
        local_ram_fs_dir { d1 = { size = %s } },
        // cpu = 12,
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

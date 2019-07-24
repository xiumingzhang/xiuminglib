"""Specific to Google's Borg."""

from os.path import join
from getpass import getuser
from time import time

from .os import call
from . import const


class JobSubmitter():
    def __init__(self, citc, label, user=None, workers=24,
                 local_ram_fs_dir_size='4096M'):
        self.citc = citc
        self.label = label
        myself = getuser()
        if user is None:
            user = myself
        self.user = user
        self.workers = workers
        # Deriving other values
        self.priority = 0 if user == myself else 115
        if user == myself:
            cell = 'qu'
        elif user == 'gcam-eng':
            cell = 'ok'
        elif user == 'gcam-gpu':
            cell = 'is'
        else:
            raise NotImplementedError(user)
        # Requirements
        self.local_ram_fs_dir_size = local_ram_fs_dir_size
        self.cell = cell

    def build(self):
        bash_cmd = 'cd %s && ' % self.citc
        bash_cmd += 'rabbit --verifiable build -c opt %s ' % self.label
        bash_cmd += '--config=libc++-preview' # bpy needs it
        retcode, _, _ = call(bash_cmd)
        assert retcode == 0, "Build failed"

    def gen_borg_file(self, job_id, param):
        borg_file_str = self._format_borg_file_str(job_id, param)
        borg_f = join(const.Dir.tmp, '%s_%f.borg' % (job_id, time()))
        with open(borg_f, 'w') as h:
            h.write(borg_file_str)
        return borg_f

    def submit(self, job_ids, param_dicts, test=False):
        if test:
            # Submit just one and see how it goes
            self._submit((job_ids[0], param_dicts[0], test))
        else:
            # Submit all using a pool of workers
            from multiprocessing import Pool
            from tqdm import tqdm
            pool = Pool(self.workers)
            list(tqdm(pool.imap_unordered(
                self._submit,
                [(i, x, test) for i, x in zip(job_ids, param_dicts)]
            ), total=len(job_ids)))
            pool.close()
            pool.join()

    def _submit(self, args):
        job_id, param, test = args
        borg_f = self.gen_borg_file(job_id, param)
        # Submit
        action = 'runlocal' if test else 'reload'
        action = 'reload' # FIXME: runlocal doesn't work: b/74472376
        bash_cmd = 'cd %s && ' % self.citc
        bash_cmd += 'borgcfg %s %s --skip_confirmation --borguser %s' \
            % (borg_f, action, self.user)
        call(bash_cmd)

    def _format_borg_file_str(self, job_id, param):
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
                blaze_label = '//%s',
            }
        }

        // What program are we going to run?
        package_binary = 'bin/main'

        // What command line parameters should we pass to this program?
        args = {
    ''' % (job_id, self.cell, self.label)
        for k, v in param.items():
            if isinstance(v, str):
                v = "'%s'" % v
            elif isinstance(v, int):
                v = "%d" % v
            elif isinstance(v, list):
                v = "'" + ",".join(str(x) for x in v) + "'"
            else:
                raise TypeError(type(v))
            file_str += "        %s = %s,\n" % (k, v)
        file_str += '''    }

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
    ''' % (self.local_ram_fs_dir_size, self.user, self.priority)
        if self.priority == 0:
            file_str += '''    }
    }'''
        else: # e.g., P115 needs a batch strategy
            file_str += '''
            batch_quota {
                strategy = 'RUN_SOON'
            }
        }
    }'''
        return file_str

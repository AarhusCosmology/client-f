"""Bounded MPI checks; requires mpiexec and mpi4py, no CLASS or data files."""
import os
from pathlib import Path
import shutil
import signal
import shlex
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
launcher=shutil.which('mpiexec')
if not launcher:raise SystemExit('mpiexec is required')
launcher_args=shlex.split(os.environ.get('CLIENT_TEST_MPI_ARGS',''))
def check(name, arguments, expect_success=True):
    proc=subprocess.Popen([launcher,*launcher_args,'-n','2',sys.executable,*arguments],
                          cwd=root,start_new_session=True)
    try:code=proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        raise SystemExit(f'{name}: timed out')
    if (code == 0) != expect_success:raise SystemExit(f'{name}: unexpected exit {code}')
    print(f'{name}: passed',flush=True)

for name in ('mpi_audit_failure_smoke.py','mpi_boost_fallback_smoke.py','mpi_evaluation_failure_smoke.py'):
    check(name,[str(root/'tests'/name)])
import uuid
check('MPI CLI setup failure',[str(root/'client.py'),'/tmp/no-client-run-'+uuid.uuid4().hex],False)
# Exits that every rank makes alike are ordinary exits, not a job to abort with a traceback.
check('MPI CLI --help',[str(root/'client.py'),'--help'])
import tempfile
with tempfile.TemporaryDirectory() as combined:
    (Path(combined)/'combine.json').write_text('{}')
    said=subprocess.run([launcher,*launcher_args,'-n','2',sys.executable,str(root/'client.py'),combined],cwd=root,
                        capture_output=True,text=True,timeout=90)
if said.returncode==0 or 'combined emulator' not in said.stderr or 'Traceback' in said.stderr:
    raise SystemExit(f'MPI CLI combined emulator: exit {said.returncode}\n{said.stderr[-2000:]}')
print('MPI CLI combined emulator: passed',flush=True)
# The loop itself on two ranks, killed inside an acquisition and continued twice. It starts
# mpiexec for each of its three runs, so it is started here without one.
loop=subprocess.run([sys.executable,str(root/'tests'/'mpi_loop_recovery_smoke.py')],cwd=root,timeout=900)
if loop.returncode:raise SystemExit(f'mpi_loop_recovery_smoke.py: unexpected exit {loop.returncode}')
print('mpi_loop_recovery_smoke.py: passed',flush=True)
# The audit's chain search in a boost iteration (acquisition.audit.chain): the master runs a chain
# inside the audit's collective, and its summits are evaluated by every rank. Starts mpiexec itself.
chain=subprocess.run([sys.executable,str(root/'tests'/'mpi_audit_chain_smoke.py')],cwd=root,timeout=900)
if chain.returncode:raise SystemExit(f'mpi_audit_chain_smoke.py: unexpected exit {chain.returncode}')
print('mpi_audit_chain_smoke.py: passed',flush=True)

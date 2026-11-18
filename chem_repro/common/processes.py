"""Bounded scientific worker processes; preserve logs and stop the whole process group."""

import os
import signal
import subprocess


def run_worker(command, *, log_path, timeout, env=None, pass_fds=()):
    """Keep child logs, including failures, and avoid orphaned engine processes on timeout."""
    with log_path.open("a") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                 pass_fds=pass_fds, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            raise
    if code:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        raise RuntimeError(f"Scientific worker exited with code {code}; see {log_path.name}")

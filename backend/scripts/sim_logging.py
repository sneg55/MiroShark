"""
Logging configuration for the simulation process.

Suppresses verbose Wonderwall library output and cleans up stale log directories.
"""

import logging
import os


class MaxTokensWarningFilter(logging.Filter):
    """Suppress camel-ai warnings about max_tokens (we intentionally omit it)."""

    def filter(self, record):
        msg = record.getMessage()
        if "max_tokens" in msg and "Invalid or missing" in msg:
            return False
        return True


# Install filter at module load time, before any camel code runs
logging.getLogger().addFilter(MaxTokensWarningFilter())


def disable_oasis_logging() -> None:
    """Suppress verbose log output from the Wonderwall library."""
    for logger_name in ("social.agent", "social.twitter", "social.rec", "wonderwall.env", "table"):
        lg = logging.getLogger(logger_name)
        lg.setLevel(logging.CRITICAL)
        lg.handlers.clear()
        lg.propagate = False


def init_logging_for_simulation(simulation_dir: str) -> None:
    """
    Prepare logging for a simulation run.

    - Disables Wonderwall verbose logging.
    - Removes stale log/ subdirectory if present.
    """
    disable_oasis_logging()
    old_log_dir = os.path.join(simulation_dir, "log")
    if os.path.exists(old_log_dir):
        import shutil
        shutil.rmtree(old_log_dir, ignore_errors=True)

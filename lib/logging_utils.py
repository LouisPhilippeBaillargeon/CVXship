from __future__ import annotations

import contextlib
import logging
import sys
import warnings
from pathlib import Path
from typing import Iterator, TextIO


VERBOSE = 15
PROGRESS = 25

logging.addLevelName(VERBOSE, "VERBOSE")
logging.addLevelName(PROGRESS, "PROGRESS")

LOGGER_NAME = "cvxship"
BATTERY_SIMULTANEOUS_WARNING_THRESHOLD_MW = 1e-3
BATTERY_COMMAND_WARNING_THRESHOLD_MW = 1e-3
_SMALL_WARNING_THRESHOLDS_MW = {
    "battery_simultaneous_command_netted": BATTERY_SIMULTANEOUS_WARNING_THRESHOLD_MW,
    "battery_charge_command_reduced": BATTERY_COMMAND_WARNING_THRESHOLD_MW,
    "battery_charge_command_increased": BATTERY_COMMAND_WARNING_THRESHOLD_MW,
    "battery_discharge_command_reduced": BATTERY_COMMAND_WARNING_THRESHOLD_MW,
    "battery_discharge_command_increased": BATTERY_COMMAND_WARNING_THRESHOLD_MW,
    "battery_charge_command_reduced_for_power_balance": BATTERY_COMMAND_WARNING_THRESHOLD_MW,
    "battery_charge_command_increased_for_power_balance": BATTERY_COMMAND_WARNING_THRESHOLD_MW,
    "battery_discharge_command_reduced_for_power_balance": BATTERY_COMMAND_WARNING_THRESHOLD_MW,
    "battery_discharge_command_increased_for_power_balance": BATTERY_COMMAND_WARNING_THRESHOLD_MW,
}

_LOGGER = logging.getLogger(LOGGER_NAME)
_LOGGER.addHandler(logging.NullHandler())
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_PRINT_CAPTURE_INSTALLED = False
_CAPTURED_STDOUT = None
_CAPTURED_STDERR = None
_LOGGING_EXCEPTHOOK_INSTALLED = False
_PREVIOUS_EXCEPTHOOK = sys.excepthook


class _ConsoleFilter(logging.Filter):
    def __init__(self, *, verbose_enabled: bool):
        super().__init__()
        self.verbose_enabled = bool(verbose_enabled)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        if record.levelno == PROGRESS:
            return True
        if record.levelno == VERBOSE:
            return self.verbose_enabled
        return False


class _PrintCapture(TextIO):
    def __init__(self, logger: logging.Logger, default_level: int):
        self._logger = logger
        self._default_level = default_level
        self._buffer = ""

    @property
    def encoding(self) -> str:
        return "utf-8"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._buffer += str(data)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        return len(data)

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""

    def _emit(self, line: str) -> None:
        message = line.rstrip()
        if not message:
            return
        level = _level_from_print_text(message, self._default_level)
        self._logger.log(level, message)


class _ForwardingLogHandler(logging.Handler):
    def __init__(self, target_logger: logging.Logger, *, verbose_level: int):
        super().__init__(level=logging.DEBUG)
        self._target_logger = target_logger
        self._verbose_level = verbose_level

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno >= logging.WARNING:
                level = record.levelno
            else:
                level = self._verbose_level
            self._target_logger.log(level, record.getMessage())
        except Exception:
            self.handleError(record)


def _level_from_print_text(message: str, default_level: int) -> int:
    normalized = message.lstrip().upper()
    if normalized.startswith(("[ERROR]", "ERROR", "[EMS ERROR]", "[FR_O ERROR]")):
        return logging.ERROR
    if normalized.startswith(("[WARN]", "WARNING", "[EMS WARNING]", "[FR_O WARNING]")):
        return logging.WARNING
    return default_level


def get_logger() -> logging.Logger:
    return _LOGGER


def configure_run_logging(
    *,
    debug_log_path: str | Path,
    warnings_errors_log_path: str | Path,
    console_verbose: bool,
    console_log_path: str | Path | None = None,
    capture_prints: bool = True,
) -> None:
    global _PRINT_CAPTURE_INSTALLED, _CAPTURED_STDOUT, _CAPTURED_STDERR

    if _PRINT_CAPTURE_INSTALLED:
        shutdown_run_logging()

    console_stream = sys.stdout
    logger = get_logger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console_formatter = logging.Formatter("%(message)s")

    debug_path = Path(debug_log_path)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_handler = logging.FileHandler(debug_path, mode="w", encoding="utf-8")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)

    warn_path = Path(warnings_errors_log_path)
    warn_path.parent.mkdir(parents=True, exist_ok=True)
    warn_handler = logging.FileHandler(warn_path, mode="w", encoding="utf-8")
    warn_handler.setLevel(logging.WARNING)
    warn_handler.setFormatter(formatter)
    logger.addHandler(warn_handler)

    console_handler = logging.StreamHandler(console_stream)
    console_handler.setLevel(logging.DEBUG)
    console_handler.addFilter(_ConsoleFilter(verbose_enabled=console_verbose))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if console_log_path is not None:
        console_path = Path(console_log_path)
        console_path.parent.mkdir(parents=True, exist_ok=True)
        console_file_handler = logging.FileHandler(console_path, mode="w", encoding="utf-8")
        console_file_handler.setLevel(logging.DEBUG)
        console_file_handler.addFilter(_ConsoleFilter(verbose_enabled=console_verbose))
        console_file_handler.setFormatter(formatter)
        logger.addHandler(console_file_handler)

    logging.captureWarnings(True)
    py_warnings_logger = logging.getLogger("py.warnings")
    py_warnings_logger.handlers.clear()
    py_warnings_logger.propagate = True
    py_warnings_logger.parent = logger

    if capture_prints and not _PRINT_CAPTURE_INSTALLED:
        _CAPTURED_STDOUT = sys.stdout
        _CAPTURED_STDERR = sys.stderr
        sys.stdout = _PrintCapture(logger, logging.DEBUG)
        sys.stderr = _PrintCapture(logger, logging.ERROR)
        _PRINT_CAPTURE_INSTALLED = True

    _install_excepthook()


def shutdown_run_logging() -> None:
    global _PRINT_CAPTURE_INSTALLED, _CAPTURED_STDOUT, _CAPTURED_STDERR

    if _PRINT_CAPTURE_INSTALLED:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = _CAPTURED_STDOUT or _ORIGINAL_STDOUT
            sys.stderr = _CAPTURED_STDERR or _ORIGINAL_STDERR
            _PRINT_CAPTURE_INSTALLED = False
            _CAPTURED_STDOUT = None
            _CAPTURED_STDERR = None

    logging.captureWarnings(False)

    logger = get_logger()
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
    logger.addHandler(logging.NullHandler())


def _install_excepthook() -> None:
    global _LOGGING_EXCEPTHOOK_INSTALLED
    if _LOGGING_EXCEPTHOOK_INSTALLED:
        return

    def _log_exception(exc_type, exc_value, traceback):
        get_logger().error(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, traceback),
        )
        _PREVIOUS_EXCEPTHOOK(exc_type, exc_value, traceback)

    sys.excepthook = _log_exception
    _LOGGING_EXCEPTHOOK_INSTALLED = True


def debug(message: str, *args, **kwargs) -> None:
    get_logger().debug(message, *args, **kwargs)


def verbose(message: str, *args, **kwargs) -> None:
    get_logger().log(VERBOSE, message, *args, **kwargs)


def progress(message: str, *args, **kwargs) -> None:
    get_logger().log(PROGRESS, message, *args, **kwargs)


def warning(message: str, *args, **kwargs) -> None:
    get_logger().warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs) -> None:
    get_logger().error(message, *args, **kwargs)


@contextlib.contextmanager
def capture_solver_output(*, echo_verbose: bool) -> Iterator[None]:
    """
    Capture solver stdout/stderr into debug.log.

    If echo_verbose is true, the captured lines are tagged as VERBOSE so the
    console filter mirrors them; otherwise they remain DEBUG-only.
    """
    logger = get_logger()
    level = VERBOSE if echo_verbose else logging.DEBUG
    stdout_capture = _PrintCapture(logger, level)
    stderr_capture = _PrintCapture(logger, logging.ERROR)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        yield
    finally:
        stdout_capture.flush()
        stderr_capture.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@contextlib.contextmanager
def capture_cvxpy_logs(*, echo_verbose: bool) -> Iterator[None]:
    """
    Route CVXPY's own verbose logger through the run logger.

    CVXPY installs a stderr handler at import time, so redirecting sys.stderr
    during a solve is not enough to prevent verbose messages from reaching the
    console. Temporarily replacing that handler keeps those lines in debug.log
    and lets the normal console filter decide whether to mirror them.
    """
    cvxpy_logger = logging.getLogger("__cvxpy__")
    old_handlers = list(cvxpy_logger.handlers)
    old_propagate = cvxpy_logger.propagate
    old_level = cvxpy_logger.level
    level = VERBOSE if echo_verbose else logging.DEBUG
    cvxpy_logger.handlers = [
        _ForwardingLogHandler(get_logger(), verbose_level=level)
    ]
    cvxpy_logger.propagate = False
    cvxpy_logger.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        for handler in cvxpy_logger.handlers:
            handler.flush()
            handler.close()
        cvxpy_logger.handlers = old_handlers
        cvxpy_logger.propagate = old_propagate
        cvxpy_logger.setLevel(old_level)


def log_warning_record(prefix: str, label: str, rec: dict, *, amount_label: str = "max_delta") -> None:
    warning(
        "%s %s: %s count=%s, %s=%.6g",
        prefix,
        label,
        rec.get("message", ""),
        rec.get("count", 0),
        amount_label,
        float(rec.get("max_amount", 0.0)),
    )


def validation_warning_is_reportable(key: str, rec: dict) -> bool:
    return validation_warning_amount_is_reportable(key, rec.get("max_amount", 0.0))


def validation_warning_amount_is_reportable(key: str, amount: float) -> bool:
    threshold = _SMALL_WARNING_THRESHOLDS_MW.get(str(key))
    if threshold is None:
        return True
    return float(abs(amount)) > threshold


def solve_with_logging(problem, *, echo_verbose: bool, **solve_kwargs):
    solve_kwargs = dict(solve_kwargs)
    solve_kwargs["verbose"] = True
    with (
        capture_cvxpy_logs(echo_verbose=echo_verbose),
        capture_solver_output(echo_verbose=echo_verbose),
    ):
        return problem.solve(**solve_kwargs)

import sys
import threading

# Global print lock
print_lock = threading.Lock()


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue

        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            pass


def safe_print(message: str) -> None:
    print(message)


def thread_safe_print(message: str) -> None:
    with print_lock:
        print(message)


configure_utf8_output()

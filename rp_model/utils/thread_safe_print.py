import threading

# Global print lock
print_lock = threading.Lock()


def thread_safe_print(message: str) -> None:
    with print_lock:
        print(message)

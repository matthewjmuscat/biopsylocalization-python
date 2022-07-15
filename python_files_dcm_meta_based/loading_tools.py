from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
from decimal import Decimal # for use in the loading percentage calculation

class Loader:
    def __init__(self, max_iterator, desc="Loading...", end="Done!", timeout=0.1):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.max_iterator = max_iterator
        self.timeout = timeout
        self.iterator = 0

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            percent_complete = (self.iterator/self.max_iterator)*100
            rounded_percent_complete = round(Decimal(percent_complete),1)
            print(f"\r{self.desc} {rounded_percent_complete} % complete, {c} ", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        self.start()
        return self 

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()


if __name__ == "__main__":
    with Loader("Loading with context manager..."):
        for i in range(10):
            sleep(0.25)

    loader = Loader("Loading with object...", "That was fast!", 0.05).start()
    for i in range(10):
        sleep(0.25)
    loader.stop()
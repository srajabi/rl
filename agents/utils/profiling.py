import time
import cProfile
import pstats


class timewith:
    def __init__(self, name=''):
        self.name = name
        self.start = time.time()

    @property
    def elapsed(self):
        return time.time() - self.start

    def checkpoint(self, name=''):
        print('{timer} {checkpoint} took {elapsed} seconds'.format(
            timer=self.name,
            checkpoint=name,
            elapsed=self.elapsed,
        ).strip())

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.checkpoint('finished')
        pass


def profile_sort_and_print(code):
    cProfile.run(code, './tmp/profile.stats')
    stats = pstats.Stats('./tmp/profile.stats')
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()

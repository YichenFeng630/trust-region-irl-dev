import os

# bit-reproducible GPU runs for the tests that compare two trainings; set before JAX initialises its backend
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: trains for a few minutes (a GPU is recommended); skip with -m 'not slow'")

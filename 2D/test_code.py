import resource
resource.setrlimit(resource.RLIMIT_NPROC, (100000, 100000))
import pickle
import inversion_loop


with open(f"/DATA/eyal/specfem2d/mtinv/DATA/SOURCES_it0.pk", "rb") as f:
    solutions = pickle.load(f)
inversion_loop.LBFGS_inversion_multi(solutions, 42, 0.9, 1.5, False, 10)
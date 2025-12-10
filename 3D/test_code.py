import pickle
import inversion_loop


solutions = []
with open("/DATA/eyal/specfem3d/PROJECTS/mtinv/test5/DATA/CMTSOLUTION_0_1.pk", "rb") as f:
    solutions.append(pickle.load(f))
with open("/DATA/eyal/specfem3d/PROJECTS/mtinv/test5/DATA/CMTSOLUTION_1_1.pk", "rb") as f:
    solutions.append(pickle.load(f))

inversion_loop.LBFGS_inversion_multi(solutions, 22, 0.01, 2.25, True, 10)
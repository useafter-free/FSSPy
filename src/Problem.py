
class Problem:
    def __init__(self, discrt, dim, minim, objtve, M, constrnt, bounds, optim=0, solved=False):
        self.discrete = discrt
        self.dim = dim
        self.minimize = minim
        self.objective = objtve
        self.n_constraint = M
        self.constraints = constrnt
        self.bounds = bounds
        self.optima = optim
        self.solved = solved
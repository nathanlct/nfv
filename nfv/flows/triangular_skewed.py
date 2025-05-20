from nfv.flows import Triangular


class TriangularSkewed(Triangular):
    def __init__(self, kmax=1.0):
        super().__init__(vmax=2.0, w=-1.0, kmax=kmax)

    def __repr__(self):
        return "triangular_skewed"

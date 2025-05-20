# Flux average + a diffusion term
def LaxFriedrichs(dx, dt):
    def _LaxFriedrichs(k_BX, flow_fn):
        flow_BX = flow_fn(k_BX)
        return (flow_BX[:, :-1] + flow_BX[:, 1:] - dx / dt * (k_BX[:, 1:] - k_BX[:, :-1])) / 2.0

    return _LaxFriedrichs

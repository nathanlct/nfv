import matplotlib.pyplot as plt
import numpy as np

from nfv.flows import Greenberg, Greenshield, Trapezoidal, Triangular, TriangularSkewed, Underwood

for flow in [Greenshield(), Triangular(), TriangularSkewed(), Trapezoidal(), Greenberg(), Underwood()]:
    print(flow)

    fig, axes = plt.subplots(ncols=4, figsize=(12, 4), dpi=300)

    ks = np.linspace(0, flow.kmax, 1000)
    qs = flow.q(ks).numpy()

    # Q
    axes[0].plot(ks, qs)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("q(k)")
    axes[0].axvline(flow.kmax, color="magenta", linewidth=1.0)
    if hasattr(flow, "k_crits"):
        for k_crit in flow.k_crits:
            axes[0].axvline(k_crit, color="orange", linestyle="--", linewidth=0.5)
    axes[0].axvline(flow.k_crit, color="red", linewidth=1.0)
    axes[0].axhline(flow.qmax, color="green", linewidth=1.0)

    # Q'
    qps = np.diff(qs) / np.diff(ks)
    axes[1].plot(ks[:-1], flow.qp(ks[:-1]))
    axes[1].plot(ks[:-1], qps, color="orange", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("q'(k)")

    # R
    us = np.linspace(-1, 1, 1000)
    Rs = flow.R(us)
    Rps = flow.Rp(us)

    Rs_num = np.max(qs - us.reshape(-1, 1) * ks, axis=1).reshape(-1)
    Rps_num = np.diff(Rs_num) / np.diff(us)

    axes[2].plot(us, Rs)
    axes[2].plot(us, Rs_num, color="orange", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("u")
    axes[2].set_ylabel("R(u)")

    axes[3].plot(us[:-1], Rps[:-1])
    axes[3].plot(us[:-1], Rps_num, color="orange", linestyle="--", linewidth=1.0)
    axes[3].set_xlabel("u")
    axes[3].set_ylabel("R'(u)")

    plt.tight_layout()
    plt.savefig(f"flow_{str(flow)}.png")
    plt.close(fig)

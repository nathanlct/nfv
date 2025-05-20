from torch import nn

from nfv.flows import Greenberg, Greenshield, Trapezoidal, Triangular, TriangularSkewed, Underwood
from nfv.models import CNNStencilModel

model_list = {
    "greenshield": {
        "supervised": CNNStencilModel(act=nn.ELU, clip=Greenshield().qmax),  # q1lcz88q
        "unsupervised": CNNStencilModel(act=nn.ReLU, clip=Greenshield().qmax),
    },
    "triangular": {
        "supervised": CNNStencilModel(act=nn.ELU, clip=Triangular().qmax),  # 4vo3xy6v
        "unsupervised": CNNStencilModel(act=nn.ReLU, clip=Triangular().qmax),
    },
    "triangular_skewed": {
        "supervised": CNNStencilModel(act=nn.ReLU, clip=TriangularSkewed().qmax),  # v80tklua
        "unsupervised": CNNStencilModel(act=nn.ReLU, clip=TriangularSkewed().qmax),
    },
    "trapezoidal": {
        "supervised": CNNStencilModel(act=nn.ReLU, clip=Trapezoidal().qmax),  # 140h69kz
        "unsupervised": CNNStencilModel(act=nn.ReLU, clip=Trapezoidal().qmax),
    },
    "greenberg": {
        "supervised": CNNStencilModel(act=nn.ELU, clip=Greenberg().qmax),  # 9pfoo160 (25k)
        "unsupervised": CNNStencilModel(act=nn.Tanh, clip=1.0),
    },
    "underwood": {
        "supervised": CNNStencilModel(act=nn.ELU, clip=Underwood().qmax),  # 3zbnn523
        "unsupervised": CNNStencilModel(act=nn.ReLU, clip=1.0),
    },
}


def load_model(flow, name):
    model = model_list[flow][name]
    model.load_checkpoint(f"models/{flow}/{name}.pth")
    return model

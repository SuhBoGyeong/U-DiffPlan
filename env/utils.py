from dataclasses import dataclass, asdict, astuple



def RK4(f, dt, x):

    k1 = f(x)
    k2 = f(x + dt * k1 / 2.0)
    k3 = f(x + dt * k2 / 2.0)
    k4 = f(x + dt * k3)

    return x + dt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0



@dataclass
class Parameters:

    def get_dict(self):
        return asdict(self)

    def get_tuple(self):
        return astuple(self)

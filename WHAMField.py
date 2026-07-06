import numpy as np
from scipy.interpolate import LinearNDInterpolator


class WHAMField:
    def __init__(
        self,
        br_file="./data/Field_WHAM_R_Br_6kA_17500A.txt",
        bz_file="./data/Field_WHAM_R_Bz_6kA_17500A.txt",
        m=1.0,
        q=1.0,
        B0=1.0,
        T=1.0,
        scale=1.0
    ):
        br_data = np.loadtxt(br_file, comments="%")
        bz_data = np.loadtxt(bz_file, comments="%")

        #Scaling to make data dimensionless
        br_data[:, :2] *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
        br_data[:, 2] /= B0
        bz_data[:, :2] *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
        bz_data[:, 2] /= B0
        
        self.B0 = B0 # center field in tesla

        self.Br_interp = LinearNDInterpolator(br_data[:, 0:2], br_data[:, 2])
        self.Bz_interp = LinearNDInterpolator(bz_data[:, 0:2], bz_data[:, 2])

    def field(self, pos, t=0):
        x, y, z = pos
        r = np.sqrt(x**2 + y**2)

        Br = self.Br_interp(r, z)
        Bz = self.Bz_interp(r, z)

        Br = float(Br)
        Bz = float(Bz)

        if np.isnan(Br) or np.isnan(Bz):
            return np.array([0.0, 0.0, 0.0])

        if r < 1e-10:
            Br = 0.0
            Bx = 0.0
            By = 0.0
        else:
            Bx = Br * x / r
            By = Br * y / r

        return np.array([Bx, By, Bz])
    
print("WHAMField module loaded")
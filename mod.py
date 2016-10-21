from daetools.pyDAE import *
import numpy as np

from dae.pyUnits import m, s, K, mol, J, A
V = J/(A*s)

F = 96485.34  # A s / mol
R = 8.31447  # J / (mol K)
T = 298  # K

# Define some variable types
conc_t = daeVariableType(
    name="conc_t", units=mol/(m**3), lowerBound=0,
    upperBound=1e20, initialGuess=1.00, absTolerance=1e-6)
elec_pot_t = daeVariableType(
    name="elec_pot_t", units=V, lowerBound=-1e20,
    upperBound=1e20, initialGuess=0, absTolerance=1e-5)
rxn_t = daeVariableType(
    name="rxn_t", units=mol/(m**2 * s), lowerBound=-1e20,
    upperBound=1e20, initialGuess=0, absTolerance=1e-5)


def kappa(c):
    out = 1  # S/m
    return out


def D(c):
    out = 1e-10  # m^2/s
    return out


def thermodynamic_factor(c):
    out = 1
    return out


def tp(c):
    out = 0.3
    return out


def Ds_n(c):
    out = 1e-12  # m^2/s
    return out


def Ds_p(c):
    out = 1e-12  # m^2/s
    return out


def U_n(c):
    out = 0.1  # V
    return out


def U_p(c):
    out = 2.0  # V
    return out


class portFromElyte(daePort):
    def __int__(self, Name, PortType, Model, Description):
        self.c_2 = daeVariable("c_2", conc_t, self, "Concentration in electrolyte")
        self.phi_2 = daeVariable("phi_2", conc_t, self, "Electric potential in electrolyte")


class portFromEtrode(daePort):
    def __int__(self, Name, PortType, Model, Description):
        self.phi_1 = daeVariable("phi_1", conc_t, self, "Electric potential in electrode")


class ModParticle(daeModel):
    def __init__(self, Name, Parent=None, Description="", Ds=Ds, U=U):
        daeModel.__init__(self, Name, Parent, Description)

        # Domain where variables are distributed
        self.r = daeDomain("r", self, m, "radial domain in particle")

        # Variables
        self.c = daeVariable("c", conc_t, self, "Concentration in the solid")
        self.c.DistributeOnDomain(self.r)
        self.j_p = daeVariable("j_p", rxn_t, self, "Rate of reaction into the solid")

        # Parameter
        self.w = daeParameter("w", m**2, self, "Weight factor for operators")
        self.w.DistributeOnDomain(self.r)
        self.i_0 = daeParameter("i_0", A/m**2, self, "Exchange current density")
        self.alpha = daeParameter("alpha", unit(), self, "Reaction symmetry factor")

        # Ports
        self.portIn_2 = portFromElyte("portInLyte", eInletPort, self, "inlet port from elyte")
        self.portIn_1 = portFromEtrode("portInEtrode", eInletPort, self, "inet port from e- conducting phase")
        self.phi_2 = self.portInLyte.phi_2
        self.c_2 = self.portInLyte.c_2
        self.phi_1 = self.portInBulk.phi_1
        self.mu_2 = self.phi_2

    def DeclareEquations(self):
        dae.daeModel.DeclareEquations(self)
        eq = self.CreateEquation("MassCons", "Mass conservation eq.")
        r = eq.DistributeOnDomain(self.r, eOpenOpen)
        c = self.c(r)
        eq.Residual = dt(c) - 1/self.w()*d(self.w() * D_s(c)*d(c, self.r, eCFDM), self.r, eCFDM)

        eq = self.CreateEquation("CenterSymmetry", "dc/dr = 0 at r=0")
        r = eq.DistributeOnDomain(self.r, eClosedOpen)
        c = self.c(r)
        eq.Residual = d(c, self.r, eCFDM)

        eq = self.CreateEquation("SurfaceGradient", "D_s*dc/dr = j_+ at r=R_p")
        r = eq.DistributeOnDomain(self.r, eOpenClosed)
        c = self.c(r)
        eq.Residual = D_s(c) * d(c, self.r, eCFDM) - self.j_p()

        eq = self.CreateEquation("SurfaceRxn", "Reaction rate")
        eta = self.phi_1() - self.phi_2() - U(c)
        eq.Residual = self.j_p() - self.i_0() * (Exp(-alpha*eta) - Exp((1 - alpha)*eta))


class ModCell(daeModel):
    def __init__(self, Name, Parent=None, Description="", ndD=None):
        daeModel.__init__(self, Name, Parent, Description)
        self.ndD = ndD

        # Domains where variables are distributed
        self.x_n = daeDomain("x_n", self, m, "X domain in negative electrode")
        self.x_s = daeDomain("x_s", self, m, "X domain in separator")
        self.x_p = daeDomain("x_p", self, m, "X domain in positive electrode")

        # Variables
        # Concentration/potential in electrode regions of elyte
        self.c_lyte = dae.daeVariable("c_lyte", conc_t, self, "Concentration in the elyte")
        self.phi_applied = dae.daeVariable("phi_applied", elec_pot_t, self, "Elec. pot. in elyte")
        self.c = dae.daeVariable("c", mole_frac_t, self, "Conc in solid")
        self.c.DistributeOnDomain(self.x)
        self.c.DistributeOnDomain(self.y)
        self.mu = dae.daeVariable("mu", elec_pot_t, self, "chem pot. in solid")
        self.mu.DistributeOnDomain(self.x)
        self.mu.DistributeOnDomain(self.y)
        self.current = dae.daeVariable("current", dae.no_t, self, "Total current of the cell")
        self.dummyVar = dae.daeVariable("dummyVar", dae.no_t, self, "dummyVar")

        # Parameters
        self.F = daeParameter("F", A*s/mol, self, "Faraday's constant")
        self.R = daeParameter("R", J/(mol*K), self, "Gas constant")
        self.T = daeParameter("T", K, self, "Temperature")
        self.ap = daeParameter("a", 1/m, "Reacting area per electrode volume")
        self.n = daeParameter("n", unit(), "Number of electrons per reaction")
        self.s_p = daeParameter("s_p", unit(), "Stoichiometric number of cation in reaction")
        self.s_m = daeParameter("s_m", unit(), "Stoichiometric number of anion in reaction")
        self.nu_p = daeParameter("nu_p", unit(), "Stoichiometric number of cation in salt")
        self.nu_m = daeParameter("nu_m", unit(), "Stoichiometric number of anion in salt")
        self.z_p = daeParameter("z_p", unit(), "Valence number of cation")
        self.z_m = daeParameter("z_m", unit(), "Valence number of anion")
        self.BruggExp_n = daeParameter("BruggExp_n", unit(), "Bruggeman exponent in x_n")
        self.BruggExp_p = daeParameter("BruggExp_p", unit(), "Bruggeman exponent in x_p")
        self.poros_n = daeParameter("poros_n", unit(), "porosity in x_n")
        self.poros_p = daeParameter("poros_p", unit(), "porosity in x_p")
        self.nu = self.nu_p() + self.nu_m()

    def DeclareEquations(self):
        dae.daeModel.DeclareEquations(self)

        # Define the total current.
        eq = self.CreateEquation("Total_Current")
        eq.Residual = self.current()
        Nx, Ny = self.x.NumberOfPoints, self.y.NumberOfPoints
        c_rb = self.c.array(Nx-1, '*')
        dmu_rb = self.mu.d_array(self.x, Nx-1, '*')
        c_tb = self.c.array('*', Ny-1)
        dmu_tb = self.mu.d_array(self.y, '*', Ny-1)
        eq.Residual -= dae.Integral(Dfunc(c_rb)*dmu_rb)
        eq.Residual -= dae.Integral(Dfunc(c_tb)*dmu_tb)

        # For this simplified simulation, keep the electrolyte constant
        eq = self.CreateEquation("elyte_c")
        eq.Residual = self.c_lyte.dt()

        if ndD["profileType"] == "CC":
            # Total Current Constraint Equation
            eq = self.CreateEquation("Total_Current_Constraint")
            eq.Residual = self.current() - (
                ndD["currPrev"] + (ndD["currset"] - ndD["currPrev"])
                * (1 - np.exp(-dae.Time()/(ndD["tend"]*ndD["tramp"]))))
        elif ndD["profileType"] == "CV":
            # Keep applied potential constant
            eq = self.CreateEquation("applied_potential")
            eq.Residual = self.phi_applied() - (
                ndD["phiPrev"] + (ndD["Vset"] - ndD["phiPrev"])
                * (1 - np.exp(-dae.Time()/(ndD["tend"]*ndD["tramp"])))
                )

        for eq in self.Equations:
            eq.CheckUnitsConsistency = False

        if ndD["profileType"] == "CC":
            # Set the condition to terminate the simulation upon reaching
            # a cutoff voltage.
            self.stopCondition = (
                ((self.phi_applied() <= ndD["phimin"])
                    | (self.phi_applied() >= ndD["phimax"]))
                & (self.dummyVar() < 1))
            self.ON_CONDITION(self.stopCondition,
                              setVariableValues=[(self.dummyVar, 2)])

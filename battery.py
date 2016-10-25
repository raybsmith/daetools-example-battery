from daetools.pyDAE import *
import numpy as np

from pyUnits import m, s, K, mol, J, A
V = J/(A*s)
S = A/V

# Define some variable types
conc_t = daeVariableType(
    name="conc_t", units=mol/(m**3), lowerBound=0,
    upperBound=1e20, initialGuess=1.00, absTolerance=1e-6)
elec_pot_t = daeVariableType(
    name="elec_pot_t", units=V, lowerBound=-1e20,
    upperBound=1e20, initialGuess=0, absTolerance=1e-5)
current_dens_t = daeVariableType(
    name="current_dens_t", units=A/m**2, lowerBound=-1e20,
    upperBound=1e20, initialGuess=0, absTolerance=1e-5)
rxn_t = daeVariableType(
    name="rxn_t", units=mol/(m**2 * s), lowerBound=-1e20,
    upperBound=1e20, initialGuess=0, absTolerance=1e-5)
process_info = {"profileType": "CC",
                "tend": 1 * s,
                }

def kappa(c):
    """Return the conductivity of the electrolyte in S/m as a function of concentration in M."""
    out = 1  # S/m
    return out

def D(c):
    """Return electrolyte diffusivity (in m^2/s) as a function of concentration in M."""
    out = 1e-10  # m**2/s
    return out

def thermodynamic_factor(c):
    """Return the electrolyte thermodynamic factor as a function of concentration in M."""
    out = 1
    return out

def t_p(c):
    """Return the electrolyte cation transference number as a function of concentration in M."""
    out = 0.3 * c/c
    return out

def Ds_n(y):
    """Return diffusivity (in m^2/s) as a function of solid filling fraction, y."""
    out = 1e-12  # m**2/s
    return out

def Ds_p(y):
    """Return diffusivity (in m^2/s) as a function of solid filling fraction, y."""
    out = 1e-12  # m**2/s
    return out

def U_n(y):
    """Return the equilibrium potential (V vs Li) of the negative electrode active material
    as a function of solid filling fraction, y.
    """
    out = 0.1  # V
    return out

def U_p(y):
    """Return the equilibrium potential (V vs Li) of the positive electrode active material
    as a function of solid filling fraction, y.
    """
    out = 2.0  # V
    return out

class portFromMacro(daePort):
    def __init__(self, Name, PortType, Model, Description=""):
        daePort.__init__(self, Name, PortType, Model, Description)
        self.c_2 = daeVariable("c_2", conc_t, self, "Concentration in electrolyte")
        self.phi_2 = daeVariable("phi_2", elec_pot_t, self, "Electric potential in electrolyte")
        self.phi_1 = daeVariable("phi_1", elec_pot_t, self, "Electric potential in bulk electrode")

class portFromParticle(daePort):
    def __init__(self, Name, PortType, Model, Description=""):
        daePort.__init__(self, Name, PortType, Model, Description)
        self.j_p = daeVariable("j_p", rxn_t, self, "Reaction rate at particle surface")

class ModParticle(daeModel):
    def __init__(self, Name, Parent=None, Description="", Ds=None, U=None):
        daeModel.__init__(self, Name, Parent, Description)
        self.Ds = Ds
        self.U = U

        # Domain where variables are distributed
        self.r = daeDomain("r", self, m, "radial domain in particle")

        # Variables
        self.c = daeVariable("c", conc_t, self, "Concentration in the solid")
        self.c.DistributeOnDomain(self.r)
        self.j_p = daeVariable("j_p", rxn_t, self, "Rate of reaction into the solid")

        # Parameter
        self.w = daeParameter("w", m**2, self, "Weight factor for operators")
        self.w.DistributeOnDomain(self.r)
        self.j_0 = daeParameter("j_0", mol/(m**2 * s), self, "Exchange current density / F")
        self.alpha = daeParameter("alpha", unit(), self, "Reaction symmetry factor")
        self.c_ref = daeParameter("c_ref", mol/m**3, self, "Max conc of species in the solid")
        self.D_ref = daeParameter("D_ref", m**2/s, self, "Reference units for diffusivity in the solid")
        self.U_ref = daeParameter("U_ref", V, self, "Reference units for equilibrium voltage of the solid")
        self.V_thermal = daeParameter("V_thermal", V, self, "Thermal voltage")

        # Ports
        self.portIn = portFromMacro("portInMacro", eInletPort, self, "inlet port from macroscopic phases")
        self.portOut = portFromParticle("portOut", eOutletPort, self, "outlet to elyte")
        self.phi_2 = self.portIn.phi_2
        self.c_2 = self.portIn.c_2
        self.phi_1 = self.portIn.phi_1

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        eq = self.CreateEquation("MassCons", "Mass conservation eq.")
        r = eq.DistributeOnDomain(self.r, eOpenOpen)
        c = self.c(r)
        w = self.w(r)
        eq.Residual = dt(c) - 1/w*d(w * self.D_ref()*self.Ds(c/self.c_ref())*d(c, self.r, eCFDM), self.r, eCFDM)

        eq = self.CreateEquation("CenterSymmetry", "dc/dr = 0 at r=0")
        r = eq.DistributeOnDomain(self.r, eLowerBound)
        c = self.c(r)
        eq.Residual = d(c, self.r, eCFDM)

        eq = self.CreateEquation("SurfaceGradient", "D_s*dc/dr = j_+ at r=R_p")
        r = eq.DistributeOnDomain(self.r, eUpperBound)
        c = self.c(r)
        eq.Residual = self.D_ref()*self.Ds(c/self.c_ref()) * d(c, self.r, eCFDM) - self.j_p()

        eq = self.CreateEquation("SurfaceRxn", "Reaction rate")
        c_surf = self.c(self.r.NumberOfPoints - 1)
        eta = self.phi_1() - self.phi_2() - self.U_ref()*self.U(c_surf/self.c_ref())
#        eta_ndim = eta * self.F() / (self.R() * self.T())
        eta_ndim = eta / self.V_thermal()
        eq.Residual = self.j_p() - self.j_0() * (Exp(-self.alpha()*eta_ndim) - Exp((1 - self.alpha())*eta_ndim))

        # Set output port info
        eq = self.CreateEquation("portOut")
        eq.Residual = self.portOut.j_p() - self.j_p()

class ModCell(daeModel):
    def __init__(self, Name, Parent=None, Description="", process_info=process_info):
        daeModel.__init__(self, Name, Parent, Description)
        self.process_info = process_info

        # Domains where variables are distributed
        self.x_n = daeDomain("x_n", self, m, "X domain in negative electrode")
        self.x_s = daeDomain("x_s", self, m, "X domain in separator")
        self.x_p = daeDomain("x_p", self, m, "X domain in positive electrode")

        # Sub-models
        N_n = self.process_info["N_n"]
        N_p = self.process_info["N_p"]
        self.particles_n = np.empty(N_n, dtype=object)
        self.particles_p = np.empty(N_p, dtype=object)
        for indx_n in range(N_n):
            self.particles_n[indx_n] = ModParticle("particle_n_{}".format(indx_n), self, Ds=Ds_n, U=U_n)
        for indx_p in range(N_p):
            self.particles_p[indx_p] = ModParticle("particle_p_{}".format(indx_p), self, Ds=Ds_p, U=U_p)

        # Ports
        self.portsOut_n = np.empty(N_n, dtype=object)
        self.portsIn_n = np.empty(N_n, dtype=object)
        self.portsOut_p = np.empty(N_p, dtype=object)
        self.portsIn_p = np.empty(N_p, dtype=object)
        # negative electrode
        for indx_n in range(N_n):
            self.portsOut_n[indx_n] = portFromMacro("portOut_n_{}".format(indx_n), eOutletPort, self, "To particle")
            self.portsIn_n[indx_n] = portFromParticle("portIn_n_{}".format(indx_n), eOutletPort, self, "From particle")
        # positive electrode
        for indx_p in range(N_p):
            self.portsOut_p[indx_p] = portFromMacro("portOut_p_{}".format(indx_p), eOutletPort, self, "To particle")
            self.portsIn_p[indx_p] = portFromParticle("portIn_p_{}".format(indx_p), eOutletPort, self, "From particle")

        # Connect ports
        for indx_n in range(N_n):
            self.ConnectPorts(self.portsOut_n[indx_n], self.particles_n[indx_n].portIn)
            self.ConnectPorts(self.portsIn_n[indx_n], self.particles_n[indx_n].portOut)
        for indx_p in range(N_p):
            self.ConnectPorts(self.portsOut_p[indx_p], self.particles_p[indx_p].portIn)
            self.ConnectPorts(self.portsIn_p[indx_p], self.particles_p[indx_p].portOut)

        # Variables
        # Concentration/potential in different regions of electrolyte and electrode
        self.c_n = daeVariable("c_n", conc_t, self, "Concentration in the elyte in negative")
        self.phi1_n = daeVariable("phi1_n", elec_pot_t, self, "Electric potential in bulk sld in negative")
        self.phi2_n = daeVariable("phi2_n", elec_pot_t, self, "Electric potential in the elyte in negative")
        self.c_n.DistributeOnDomain(self.x_n)
        self.phi1_n.DistributeOnDomain(self.x_n)
        self.phi2_n.DistributeOnDomain(self.x_n)
        self.c_s = daeVariable("c_s", conc_t, self, "Concentration in the elyte in separator")
        self.phi2_s = daeVariable("phi2_s", elec_pot_t, self, "Electric potential in the elyte in separator")
        self.c_s.DistributeOnDomain(self.x_s)
        self.phi2_s.DistributeOnDomain(self.x_s)
        self.c_p = daeVariable("c_p", conc_t, self, "Concentration in the elyte in positive")
        self.phi1_p = daeVariable("phi1_p", elec_pot_t, self, "Electric potential in bulk sld in positive")
        self.phi2_p = daeVariable("phi2_p", elec_pot_t, self, "Electric potential in the elyte in positive")
        self.c_p.DistributeOnDomain(self.x_p)
        self.phi1_p.DistributeOnDomain(self.x_p)
        self.phi2_p.DistributeOnDomain(self.x_p)
        self.phiCC_n = daeVariable("phiCC_n", elec_pot_t, self, "phi at negative current collector")
        self.phiCC_p = daeVariable("phiCC_p", elec_pot_t, self, "phi at positive current collector")
        self.V = daeVariable("V", elec_pot_t, self, "Applied voltage")
        self.current = daeVariable("current", current_dens_t, self, "Total current of the cell")

        # Parameters
        self.F = daeParameter("F", A*s/mol, self, "Faraday's constant")
        self.R = daeParameter("R", J/(mol*K), self, "Gas constant")
        self.T = daeParameter("T", K, self, "Temperature")
        self.a_n = daeParameter("a_n", m**(-1), self, "Reacting area per electrode volume, negative electrode")
        self.a_p = daeParameter("a_p", m**(-1), self, "Reacting area per electrode volume, positive electrode")
        self.L_n = daeParameter("L_n", m, self, "Length of negative electrode")
        self.L_s = daeParameter("L_s", m, self, "Length of separator")
        self.L_p = daeParameter("L_p", m, self, "Length of positive electrode")
        self.BruggExp_n = daeParameter("BruggExp_n", unit(), self, "Bruggeman exponent in x_n")
        self.BruggExp_s = daeParameter("BruggExp_s", unit(), self, "Bruggeman exponent in x_s")
        self.BruggExp_p = daeParameter("BruggExp_p", unit(), self, "Bruggeman exponent in x_p")
        self.poros_n = daeParameter("poros_n", unit(), self, "porosity in x_n")
        self.poros_s = daeParameter("poros_s", unit(), self, "porosity in x_s")
        self.poros_p = daeParameter("poros_p", unit(), self, "porosity in x_p")
        self.D_ref = daeParameter("D_ref", m**2/s, self, "Reference units for diffusivity")
        self.cond_ref = daeParameter("cond_ref", S/m, self, "Reference units for conductivity")
        self.c_ref = daeParameter("c_ref", mol/m**3, self, "Reference electrolyte concentration")
        self.j_ref = daeParameter("j_ref", mol/(m**2 * s), self, "Reference units for reaction")
        self.a_ref = daeParameter("a_ref", m**(-1), self, "Reference units for area/volume")
        self.currset = daeParameter("currset", A/m**2, self, "current per electrode area")
        self.Vset = daeParameter("Vset", V, self, "applied voltage set point")
        self.tau_ramp = daeParameter("tau_ramp", s, self, "Time scale for ramping voltage or current")

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        pinfo = self.process_info
        V_thm = self.R() * self.T() / self.F()
        N_n, N_s, N_p = self.x_n.NumberOfPoints, self.x_s.NumberOfPoints, self.x_p.NumberOfPoints
        # Generally, we cannot easily use the built-in FDM because we need to couple the equations
        # in the cell domain with those in the particles. Thus, we need grid spacing info and
        # finite difference approximation formulas.
        h_n = self.L_n() / (N_n - 1)
        h_s = self.L_s() / (N_s - 1)
        h_p = self.L_p() / (N_p - 1)
        # We'll combine the separate electrolyte domains into single numpy arrays for convenience
        c = np.array([self.c_n(indx) for indx in range(N_n)]
                     + [self.c_s(indx) for indx in range(1, N_s-1)]
                     + [self.c_p(indx) for indx in range(N_p)])
        dcdt = np.array([self.c_n.dt(indx) for indx in range(N_n)]
                        + [self.c_s.dt(indx) for indx in range(1, N_s-1)]
                        + [self.c_p.dt(indx) for indx in range(N_p)])
        phi2 = np.array([self.phi2_n(indx) for indx in range(N_n)]
                        + [self.phi2_s(indx) for indx in range(1, N_s-1)]
                        + [self.phi2_p(indx) for indx in range(N_p)])
        h = np.hstack((h_n*np.ones(N_n-1), h_s*np.ones(N_s-1), h_p*np.ones(N_p-1)))
        a = np.hstack((self.a_n()*np.ones(N_n), self.a_ref()*np.ones(N_s-2), self.a_p()*np.ones(N_p)))
        j_p = np.array([self.portsIn_n[indx].j_p() for indx in range(N_n)]
                       + (N_s-2)*[0 * self.j_ref()]
                       + [self.portsIn_p[indx].j_p() for indx in range(N_p)])
        eff_factor = np.hstack((self.poros_n() / (self.poros_n()**self.BruggExp_n()) * np.ones(N_n),
                                self.poros_s() / (self.poros_s()**self.BruggExp_s()) * np.ones(N_s-2),
                                self.poros_p() / (self.poros_p()**self.BruggExp_p()) * np.ones(N_p)))

        # Non-uniform finite difference approximations
        def dfdx_center(f_left, f_cent, f_right, h_left, h_right):
            dfac = 1 + h_right/h_left
            a = -h_right/(h_left**2*dfac)
            b = (h_right/h_left**2 - 1/h_right)/dfac
            c = 1/(h_right*dfac)
            df = a*f_left + b*f_cent + c*f_right
            return df

        def dfdx_direction(f_cent, f_side1, f_side2, h1, h2, direction):
            b = (h1+h2)/(h1*h2)
            c = -h1/(h2*(h1+h2))
            a = -(b+c)
            if direction == "forward":
                pass
            elif direction == "backward":
                a *= -1
                b *= -1
                c *= -1
            df = a*f_cent + b*f_side1 + c*f_side2
            return df

        def dfdx_vec(fvec, hvec):
            df = np.hstack((dfdx_direction(fvec[0], fvec[1], fvec[2], h[0], h[1], "forward"),
                            dfdx_center(fvec[:-2], fvec[1:-1], fvec[2:], hvec[:-1], hvec[1:]),
                            dfdx_direction(fvec[-1], fvec[-2], fvec[-3], h[-1], h[-2], "backward")))
            return df

        # Set output port info (connecting c, phi1, phi2 in this model to each particle)
        # negative electrode
        for indx_n in range(N_n):
            eq = self.CreateEquation("portOut_n_c_{}".format(indx_n))
            eq.Residual = self.portsOut_n[indx_n].c_2() - self.c_n(indx_n)
            eq = self.CreateEquation("portOut_n_phi2_{}".format(indx_n))
            eq.Residual = self.portsOut_n[indx_n].phi_2() - self.phi2_n(indx_n)
            eq = self.CreateEquation("portOut_n_phi1_{}".format(indx_n))
            eq.Residual = self.portsOut_n[indx_n].phi_1() - self.phi1_n(indx_n)
        # positive electrode
        for indx_p in range(N_p):
            eq = self.CreateEquation("portOut_p_c_{}".format(indx_p))
            eq.Residual = self.portsOut_p[indx_p].c_2() - self.c_p(indx_p)
            eq = self.CreateEquation("portOut_p_phi2_{}".format(indx_p))
            eq.Residual = self.portsOut_p[indx_p].phi_2() - self.phi2_p(indx_p)
            eq = self.CreateEquation("portOut_p_phi1_{}".format(indx_p))
            eq.Residual = self.portsOut_p[indx_p].phi_1() - self.phi1_p(indx_p)

        # Electrolyte: mass and charge conservation
        dc = dfdx_vec(c, h)
        dphi2 = dfdx_vec(phi2, h)
        trans_m = 1 - t_p(c)
        dtrans_m = dfdx_vec(1 - t_p(c), h)
        kappa_eff = eff_factor * self.cond_ref() * kappa(c / self.c_ref())
        D_eff = eff_factor * self.D_ref() * D(c / self.c_ref())
        i = -kappa_eff * (dphi2 + 2*V_thm*(1 - t_p(c))*thermodynamic_factor(c)*(1/c)*dc)
        di = dfdx_vec(i, h)
        mass_term_d = dfdx_vec(D_eff*dc, h)
        mass_term_i = (trans_m*di + i*dtrans_m)/self.F()
        for indx in range(1, len(c)-1):
            # mass
            eq = self.CreateEquation("massCons_{}".format(indx))
            eq.Residual = dcdt[indx] - (mass_term_d[indx] + mass_term_i[indx])
            # charge
            eq = self.CreateEquation("chargeCons_{}".format(indx))
            eq.Residual = di[indx] - self.F()*a[indx]*j_p[indx]
#            eq.Residual = di[indx] - a[indx]*j_p[indx]
#            eq.Residual = di[indx] - self.F()

        # Electrolyte: current collector BC's on concentration and phi:
        # concentration -- no slope at either current collector
        eq = self.CreateEquation("BC_c_CC_n", "BC for c at the negative current collector")
        eq.Residual = dc[0]
        eq = self.CreateEquation("BC_c_CC_p", "BC for c at the positive current collector")
        eq.Residual = dc[-1]
        # phi -- no slope at one; arbitrary datum at the other
        eq = self.CreateEquation("BC_phi_CC_n", "BC for phi at the negative current collector")
        eq.Residual = phi2[0]
        eq = self.CreateEquation("BC_phi_CC_p", "BC for phi at the positive current collector")
        eq.Residual = dphi2[-1]

        # Tie regions together: Continuity of field variables at the electrode-separator interfaces
        # negative-separator
        eq = self.CreateEquation("ns_c_cont", "continuity of c at the negative-separator interface")
        eq.Residual = self.c_s(0) - self.c_n(N_n - 1)
        eq = self.CreateEquation("ns_phi_cont", "continuity of phi at the negative-separator interface")
        eq.Residual = self.phi2_s(0) - self.phi2_n(N_n - 1)
        # separator-positive
        eq = self.CreateEquation("sp_c_cont", "continuity of c at the separator-positive interface")
        eq.Residual = self.c_s(N_s - 1) - self.c_p(0)
        eq = self.CreateEquation("sp_phi_cont", "continuity of phi at the separator-positive interface")
        eq.Residual = self.phi2_s(N_s - 1) - self.phi2_p(0)

        # Electrode: charge conservation
        phi1_n = np.array([self.phi1_n(indx) for indx in range(N_n)])
        phi1_p = np.array([self.phi1_p(indx) for indx in range(N_p)])
        dphi1_n = dfdx_vec(phi1_n, h_n*np.ones(N_n-1))
        d2phi1_n = dfdx_vec(dphi1_n, h_n*np.ones(N_n-1))
        dphi1_p = dfdx_vec(phi1_p, h_p*np.ones(N_p-1))
        d2phi1_p = dfdx_vec(dphi1_p, h_p*np.ones(N_p-1))
        # We assume infinite conductivity in the electron conducting phase for simplicity
        # negative
        for indx in range(N_n):
            eq = self.CreateEquation("phi1_n_{}".format(indx))
            eq.Residual = d2phi1_n[indx]
        # positive
        for indx in range(N_p):
            eq = self.CreateEquation("phi1_p_{}".format(indx))
            eq.Residual = d2phi1_p[indx]

        # Electrode boundary conditions
        # at the electrode-separator interfaces, dphi/dx = 0
        eq = self.CreateEquation("phi1_n_right")
        eq.Residual = dphi1_n[-1]
        eq = self.CreateEquation("phi1_n_left")
        eq.Residual = dphi1_p[0]
        # at the current collectors, phi = that of the current collector
        eq = self.CreateEquation("phi1_n_left")
        eq.Residual = phi1_n[0] - self.phiCC_n()
        eq = self.CreateEquation("phi1_p_right")
        eq.Residual = phi1_p[-1] - self.phiCC_p()

        # Define the total current.
        eq = self.CreateEquation("Total_Current")
        eq.Residual = self.current() + np.sum(self.F()*a[:N_n]*j_p[:N_n]*h[:N_n])

        # Define the measured voltage
        eq = self.CreateEquation("Voltage")
        eq.Residual = self.phiCC_p() - self.phiCC_n()

        if pinfo["profileType"] == "CC":
            # Total Current Constraint Equation
            eq = self.CreateEquation("Total_Current_Constraint")
            eq.Residual = self.current() - self.currset()*(1 - np.exp(-Time()/self.tau_ramp()))
        elif pinfo["profileType"] == "CV":
            # Keep applied potential constant
            eq = self.CreateEquation("applied_potential")
            eq.Residual = self.V() - self.Vset()*(1 - np.exp(-Time()/self.tau_ramp()))


class SimBattery(daeSimulation):
    def __init__(self):
        daeSimulation.__init__(self)
        # Define the model we're going to simulate
        self.L_n = 100e-6 * m
        self.L_s = 80e-6 * m
        self.L_p = 100e-6 * m
        self.N_n = 16
        self.N_s = 17
        self.N_p = 18
        self.NR_n = 21
        self.NR_p = 22
        self.Rp_n = 10e-6 * m
        self.Rp_p = 10e-6 * m
        self.csmax_n = 13e3 * mol/m**3
        self.csmax_p = 5e3 * mol/m**3
        self.ff0_n = 0.01
        self.ff0_p = 0.99
        self.process_info = process_info
        self.process_info["N_n"] = self.N_n
        self.process_info["N_s"] = self.N_s
        self.process_info["N_p"] = self.N_p
        self.m = ModCell("ModCell", process_info=self.process_info)

    def SetUpParametersAndDomains(self):
        # Domains in ModCell
        self.m.x_n.CreateStructuredGrid(self.N_n - 1, 0, self.L_n.value)
        self.m.x_s.CreateStructuredGrid(self.N_s - 1, 0, self.L_s.value)
        self.m.x_p.CreateStructuredGrid(self.N_p - 1, 0, self.L_p.value)
        # Domains in each particle
        for indx_n in range(self.m.x_n.NumberOfPoints):
            self.m.particles_n[indx_n].r.CreateStructuredGrid(self.NR_n - 1, 0, self.Rp_n.value)
        for indx_p in range(self.m.x_p.NumberOfPoints):
            self.m.particles_p[indx_p].r.CreateStructuredGrid(self.NR_p - 1, 0, self.Rp_p.value)
        # Parameters in ModCell
        self.m.F.SetValue(96485.34 * A*s/mol)
        self.m.R.SetValue(8.31447 * J/(mol*K))
        self.m.T.SetValue(298 * K)
        self.m.L_n.SetValue(self.L_n)
        self.m.L_s.SetValue(self.L_s)
        self.m.L_p.SetValue(self.L_p)
        self.m.BruggExp_n.SetValue(-0.5)
        self.m.BruggExp_s.SetValue(-0.5)
        self.m.BruggExp_p.SetValue(-0.5)
        self.m.poros_n.SetValue(0.3)
        self.m.poros_s.SetValue(0.4)
        self.m.poros_p.SetValue(0.3)
        self.m.a_n.SetValue((1-self.m.poros_n.GetValue())*3/self.Rp_n)
        self.m.a_p.SetValue((1-self.m.poros_p.GetValue())*3/self.Rp_p)
        self.m.D_ref.SetValue(1 * m**2/s)
        self.m.cond_ref.SetValue(1 * S/m)
        self.m.c_ref.SetValue(1000 * mol/m**3)
        self.m.j_ref.SetValue(1 * mol/(m**2 * s))
        self.m.a_ref.SetValue(1 * m**(-1))
        self.m.currset.SetValue(1e-4 * A/m**2)
        self.m.Vset.SetValue(1.9 * V)
        self.m.tau_ramp.SetValue(1e-3 * process_info["tend"])
        # Parameters in each particle
        for indx_n in range(self.m.x_n.NumberOfPoints):
            p = self.m.particles_n[indx_n]
            N = p.r.NumberOfPoints
            rvec = np.empty(N, dtype=object)
            rvec[:] = np.linspace(0, self.Rp_n.value, N) * m
            p.w.SetValues(rvec**2)
            p.j_0.SetValue(1e-4 * mol/(m**2 * s))
            p.alpha.SetValue(0.5)
            p.c_ref.SetValue(self.csmax_n)
            p.D_ref.SetValue(1 * m**2/s)
            p.U_ref.SetValue(1 * V)
            p.V_thermal.SetValue(self.m.R.GetValue()*self.m.T.GetValue()/self.m.F.GetValue())
        for indx_p in range(self.m.x_p.NumberOfPoints):
            p = self.m.particles_p[indx_p]
            N = p.r.NumberOfPoints
            rvec = np.empty(N, dtype=object)
            rvec[:] = np.linspace(0, self.Rp_p.value, N) * m
            p.w.SetValues(rvec**2)
            p.j_0.SetValue(1e-4 * mol/(m**2 * s))
            p.alpha.SetValue(0.5)
            p.c_ref.SetValue(self.csmax_p)
            p.D_ref.SetValue(1 * m**2/s)
            p.U_ref.SetValue(1 * V)
            p.V_thermal.SetValue(self.m.R.GetValue()*self.m.T.GetValue()/self.m.F.GetValue())

    def SetUpVariables(self):
        cs0_n = self.ff0_n * self.csmax_n
        cs0_p = self.ff0_p * self.csmax_p
        # ModCell
        for indx_x_n in range(1, self.m.x_n.NumberOfPoints-1):
            self.m.c_n.SetInitialCondition(indx_x_n, 1e3 * mol/m**3)
            self.m.phi1_n.SetInitialGuess(indx_x_n, U_n(cs0_n))
        for indx_x_s in range(1, self.m.x_s.NumberOfPoints-1):
            self.m.c_s.SetInitialCondition(indx_x_s, 1e3 * mol/m**3)
        for indx_x_p in range(1, self.m.x_p.NumberOfPoints-1):
            self.m.c_p.SetInitialCondition(indx_x_p, 1e3 * mol/m**3)
            self.m.phi1_p.SetInitialGuess(indx_x_p, U_p(cs0_p))
        self.m.phiCC_n.SetInitialGuess(U_n(cs0_n))
        self.m.phiCC_p.SetInitialGuess(U_p(cs0_p))
        # particles
        for indx_n in range(self.m.x_n.NumberOfPoints):
            p = self.m.particles_n[indx_n]
            for indx_r in range(1, p.r.NumberOfPoints-1):
                p.c.SetInitialCondition(indx_r, cs0_n)
        for indx_p in range(self.m.x_p.NumberOfPoints):
            p = self.m.particles_p[indx_p]
            for indx_r in range(1, p.r.NumberOfPoints-1):
                p.c.SetInitialCondition(indx_r, cs0_p)

# Use daeSimulator class
def guiRun(app):
    sim = SimBattery()
    sim.m.SetReportingOn(True)
    sim.ReportingInterval = 10
    sim.TimeHorizon       = 1000
    simulator  = daeSimulator(app, simulation=sim)
    simulator.exec_()

# Setup everything manually and run in a console
def consoleRun():
    # Create Log, Solver, DataReporter and Simulation object
    log          = daePythonStdOutLog()
    daesolver    = daeIDAS()
    datareporter = daeTCPIPDataReporter()
    simulation   = SimBattery()

    # Enable reporting of all variables
    simulation.m.SetReportingOn(True)

    # Set the time horizon and the reporting interval
    simulation.ReportingInterval = process_info["tend"].value / 100
    simulation.TimeHorizon = process_info["tend"].value

    # Connect data reporter
    simName = simulation.m.Name + strftime(" [%d.%m.%Y %H:%M:%S]", localtime())
    if(datareporter.Connect("", simName) is False):
        sys.exit()

    # Initialize the simulation
    simulation.Initialize(daesolver, datareporter, log)

    # Save the model report and the runtime model report
    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

    # Solve at time=0 (initialization)
    simulation.SolveInitial()

    # Run
    simulation.Run()

    simulation.Finalize()

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == 'console'):
        consoleRun()
    else:
        from PyQt4 import QtGui
        app = QtGui.QApplication(sys.argv)
        guiRun(app)

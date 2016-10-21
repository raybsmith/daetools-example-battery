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
    out = 1 * S/m
    return out


def D(c):
    out = 1e-10 * m**2/s
    return out


def thermodynamic_factor(c):
    out = 1
    return out


def t_p(c):
    out = 0.3
    return out


def Ds_n(c):
    out = 1e-12 * m**2/s
    return out


def Ds_p(c):
    out = 1e-12 * m**2/s
    return out


def U_n(c):
    out = 0.1 * V
    return out


def U_p(c):
    out = 2.0 * V
    return out


class portFromMacro(daePort):
    def __int__(self, Name, PortType, Model, Description):
        self.c_2 = daeVariable("c_2", conc_t, self, "Concentration in electrolyte")
        self.phi_2 = daeVariable("phi_2", elec_pot_t, self, "Electric potential in electrolyte")
        self.phi_1 = daeVariable("phi_1", elec_pot_t, self, "Electric potential in bulk electrode")


class portFromParticle(daePort):
    def __int__(self, Name, PortType, Model, Description):
        self.j_p = daeVariable("j_p", rxn_t, self, "Reaction rate at particle surface")


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
        self.i_0 = daeParameter("j_0", mol/(m**2 * s), self, "Exchange current density / F")
        self.alpha = daeParameter("alpha", unit(), self, "Reaction symmetry factor")

        # Ports
        self.portIn_2 = portFromElyte("portInLyte", eInletPort, self, "inlet port from elyte")
        self.portIn_1 = portFromEtrode("portInEtrode", eInletPort, self, "inet port from e- conducting phase")
        self.portOut = portFromParticle("portOut", eOutletPort, self, "outlet to elyte")
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
        r = eq.DistributeOnDomain(self.r, eLowerBound)
        c = self.c(r)
        eq.Residual = d(c, self.r, eCFDM)

        eq = self.CreateEquation("SurfaceGradient", "D_s*dc/dr = j_+ at r=R_p")
        r = eq.DistributeOnDomain(self.r, eUpperBound)
        c = self.c(r)
        eq.Residual = D_s(c) * d(c, self.r, eCFDM) - self.j_p()

        eq = self.CreateEquation("SurfaceRxn", "Reaction rate")
        eta = self.phi_1() - self.phi_2() - U(c)
        eq.Residual = self.j_p() - self.j_0() * (Exp(-alpha*eta) - Exp((1 - alpha)*eta))

        # Set output port info
        eq = self.CreateEquation("portOut")
        eq.Residual = self.portOut.j_p() - self.j_p()


class ModCell(daeModel):
    def __init__(self, Name, Parent=None, Description="", process_info=None):
        daeModel.__init__(self, Name, Parent, Description)
        self.process_info = process_info

        # Domains where variables are distributed
        self.x_n = daeDomain("x_n", self, m, "X domain in negative electrode")
        self.x_s = daeDomain("x_s", self, m, "X domain in separator")
        self.x_p = daeDomain("x_p", self, m, "X domain in positive electrode")

        # Sub-models
        self.particle_n = ModParticle("particle_n", self, Ds=Ds_n, U=U_n)
        self.particle_n.DistributeOnDomain(self.x_n)
        self.particle_p = ModParticle("particle_p", self, Ds=Ds_p, U=U_p)
        self.particle_p.DistributeOnDomain(self.x_p)

        # Ports
        # negative electrode
        self.portOut_n = portFromEtrode("portOutMacro_n", eOutletPort, self, "Port to particles")
        self.portOut_n.DistributeOnDomain(self.x_n)
        self.portIn_n = portFromParticle("portIn_n", eInletPort, self, "Port from particles")
        self.portIn_n.DistributeOnDomain(self.x_n)
        # positive electrode
        self.portOut_p = portFromEtrode("portOutMacro_p", eOutletPort, self, "Port to particles")
        self.portOut_p.DistributeOnDomain(self.x_p)
        self.portIn_p = portFromParticle("portIn_p", eInletPort, self, "Port from particles")
        self.portIn_p.DistributeOnDomain(self.x_p)

        # Connect ports
        # Note: Are these distributed correctly?
        # There should be one port defined for each "particle" sub-model, one of which exists at each grid point within
        # the electrodes, and each of these ports should be connected to the corresponding port in the particle model.
        # negative electrode
        self.ConnectPorts(self.PortOutElyte_n, self.particle_n.portIn_2)
        self.ConnectPorts(self.PortOutEtrode_n, self.particle_n.portIn_1)
        self.ConnectPorts(self.PortIn_n, self.particle_n.portOut)
        # positive electrode
        self.ConnectPorts(self.PortOutElyte_p, self.particle_p.portIn_2)
        self.ConnectPorts(self.PortOutEtrode_p, self.particle_p.portIn_1)
        self.ConnectPorts(self.PortIn_p, self.particle_p.portOut)

        # Variables
        # Concentration/potential in different regions of electrolyte and electrode
        self.c_n = daeVariable("c_n", conc_t, self, "Concentration in the elyte in negative")
        self.phi1_n = daeVariable("phi2_n", elec_pot_t, self, "Electric potential in bulk sld in negative")
        self.phi2_n = daeVariable("phi1_n", elec_pot_t, self, "Electric potential in the elyte in negative")
        self.c_n.DistributeOnDomain(self.x_n)
        self.phi1_n.DistributeOnDomain(self.x_n)
        self.phi2_n.DistributeOnDomain(self.x_n)
        self.c_s = daeVariable("c_s", conc_t, self, "Concentration in the elyte in separator")
        self.phi2_s = daeVariable("phi1_s", elec_pot_t, self, "Electric potential in the elyte in separator")
        self.c_s.DistributeOnDomain(self.x_s)
        self.phi2_s.DistributeOnDomain(self.x_s)
        self.c_p = daeVariable("c_p", conc_t, self, "Concentration in the elyte in positive")
        self.phi1_p = daeVariable("phi2_p", elec_pot_t, self, "Electric potential in bulk sld in positive")
        self.phi2_p = daeVariable("phi1_p", elec_pot_t, self, "Electric potential in the elyte in positive")
        self.c_p.DistributeOnDomain(self.x_p)
        self.phi1_p.DistributeOnDomain(self.x_p)
        self.phi2_p.DistributeOnDomain(self.x_p)
        # Applied potential at the negative electrode
        self.phiCC_n = daeVariable("phiCC_n", elec_pot_t, self, "phi at negative current collector")
        self.phiCC_p = daeVariable("phiCC_p", elec_pot_t, self, "phi at positive current collector")
        self.V = daeVariable("V", elec_pot_t, self, "Applied voltage")
        self.current = dae.daeVariable("current", dae.no_t, self, "Total current of the cell")

        # Parameters
        self.F = daeParameter("F", A*s/mol, self, "Faraday's constant")
        self.R = daeParameter("R", J/(mol*K), self, "Gas constant")
        self.T = daeParameter("T", K, self, "Temperature")
        self.a_n = daeParameter("a_n", 1/m, "Reacting area per electrode volume, negative electrode")
        self.a_p = daeParameter("a_p", 1/m, "Reacting area per electrode volume, positive electrode")
        self.BruggExp_n = daeParameter("BruggExp_n", unit(), "Bruggeman exponent in x_n")
        self.BruggExp_s = daeParameter("BruggExp_s", unit(), "Bruggeman exponent in x_s")
        self.BruggExp_p = daeParameter("BruggExp_p", unit(), "Bruggeman exponent in x_p")
        self.poros_n = daeParameter("poros_n", unit(), "porosity in x_n")
        self.poros_s = daeParameter("poros_s", unit(), "porosity in x_s")
        self.poros_p = daeParameter("poros_p", unit(), "porosity in x_p")
        self.currset = daeParameter("currset", A/m**3, "current per volume of active material")
        self.Vset = daeParameter("Vset", V, "applied voltage set point")

    def DeclareEquations(self):
        dae.daeModel.DeclareEquations(self)
        V_thm = self.R() * self.T() / self.F()

        # Set output port info
        # negative electrode, c, phi1, phi2
        eq = self.CreateEquation("portOut_n_c")
        x_n = eq.DistributeOnDomain(self.x_n)
        eq.Residual = self.portOut_n(x_n).c_2 - self.c_n(x_n)
        eq = self.CreateEquation("portOutElyte_n_phi")
        x_n = eq.DistributeOnDomain(self.x_n)
        eq.Residual = self.portOut_n(x_n).phi_2 - self.phi2_n(x_n)
        eq = self.CreateEquation("portOutEtrode_n")
        x_n = eq.DistributeOnDomain(self.x_n)
        eq.Residual = self.portOut_n(x_n).phi_1 - self.phi1_n(x_n)
        # positive electrode, c, phi1, phi2
        eq = self.CreateEquation("portOut_p_c")
        x_p = eq.DistributeOnDomain(self.x_p)
        eq.Residual = self.portOut_p(x_p).c_2 - self.c_p(x_p)
        eq = self.CreateEquation("portOutElyte_p_phi")
        x_p = eq.DistributeOnDomain(self.x_p)
        eq.Residual = self.portOut_p(x_p).phi_2 - self.phi2_p(x_p)
        eq = self.CreateEquation("portOutEtrode_p")
        x_p = eq.DistributeOnDomain(self.x_p)
        eq.Residual = self.portOut_p(x_p).phi_1 - self.phi1_p(x_p)

        def i_lyte(kappa, dphidx, t_p, TF, c, dcdx):
            i = -kappa * (dphidx + 2*V_thm*(1 - t_p)*TF*(1/c)*dcdx)
            return i

        def set_up_cons_eq(name, domain, cvar, phivar, poros, BruggExp):
            eq = self.CreateEquation(name)
            x = eq.DistributeOnDomain(domain, eOpenOpen)
            c = cvar(x)
            phi = phivar(x)
            eff_factor = poros / (poros**BruggExp)
            kappa_eff = eff_factor * kappa(c)
            D_eff = eff_factor * D(c)
            dcdx = d(c, domain, eCFDM)
            dphidx = d(phi, domain, eCFDM)
            return eq, c, phi, dcdx, dphidx, kappa_eff, D_eff

        # Electrolyte: mass and charge conservation in separator
        # mass
        eq, c, phi, dcdx, dphidx, kappa_eff, D_eff = set_up_cons_eq(
            "massCons_s", self.x_s, self.c_s, self.phi_s, self.poros_s(), self.BruggExp_s())
        i = i_lyte(kappa_eff, dphidx, t_p(c), thermodynamic_factor(c), c, dcdx)
        t_m = 1 - t_p(c)
        dt_mdx = d(t_m, self.x_s, eCFDM)
        didx = d(i, self.x_s, eCFDM)
        eq.Residual = dt(c) - (d(D_eff*dcdx) + (t_m*didx + i*dt_mdx)/self.F())
        # charge
        eq, c, phi, dcdx, dphidx, kappa_eff, D_eff = set_up_cons_eq(
            "chargeCons_s", self.x_s, self.c_s, self.phi_s, self.poros_s(), self.BruggExp_s())
        i = i_lyte(kappa_eff, dphidx, t_p(c), thermodynamic_factor(c), c, dcdx)
        eq.Residual = d(i, self.x_s, eCFDM)

        # Electrolyte: mass and charge conservation in negative electrode
        # mass
        eq, c, phi, dcdx, dphidx, kappa_eff, D_eff = set_up_cons_eq(
            "massCons_n", self.x_n, self.c_n, self.phi_n, self.poros_n(), self.BruggExp_n())
        i = i_lyte(kappa_eff, dphidx, t_p(c), thermodynamic_factor(c), c, dcdx)
        t_m = 1 - t_p(c)
        dt_mdx = d(t_m, self.x_s, eCFDM)
        didx = d(i, self.x_s, eCFDM)
        eq.Residual = dt(c) - (d(D_eff*dcdx) + (t_m*didx + i*dt_mdx)/self.F())
        # charge
        eq, c, phi, dcdx, dphidx, kappa_eff, D_eff = set_up_cons_eq(
            "chargeCons_n", self.x_n, self.c_n, self.phi_n, self.poros_n(), self.BruggExp_n())
        i = i_lyte(kappa_eff, dphidx, t_p(c), thermodynamic_factor(c), c, dcdx)
        eq.Residual = d(i, self.x_n, eCFDM) - self.a_n()*self.portIn_n(self.x_n).j_p

        # Electrolyte: mass and charge conservation in positive electrode
        # mass
        eq, c, phi, dcdx, dphidx, kappa_eff, D_eff = set_up_cons_eq(
            "massCons_p", self.x_p, self.c_p, self.phi2_p, self.poros_p(), self.BruggExp_p())
        i = i_lyte(kappa_eff, dphidx, t_p(c), thermodynamic_factor(c), c, dcdx)
        t_m = 1 - t_p(c)
        dt_mdx = d(t_m, self.x_s, eCFDM)
        didx = d(i, self.x_s, eCFDM)
        eq.Residual = dt(c) - (d(D_eff*dcdx) + (t_m*didx + i*dt_mdx)/self.F())
        # charge
        eq, c, phi, dcdx, dphidx, kappa_eff, D_eff = set_up_cons_eq(
            "chargeCons_p", self.x_p, self.c_p, self.phi2_p, self.poros_p(), self.BruggExp_p())
        i = i_lyte(kappa_eff, dphidx, t_p(c), thermodynamic_factor(c), c, dcdx)
        eq.Residual = d(i, self.x_p, eCFDM) - self.a_p()*self.portIn_n(self.x_p).j_p

        # Electrolyte: current collector BC's on concentration:
        eq = self.CreateEquation("BC_c_CC_n", "BC for c at the negative current collector")
        x_n = eq.DistributeOnDomain(self.x_n, eLowerBound)
        eq.Residual = d(self.c(x_n), self.x_n, eCFDM)
        eq = self.CreateEquation("BC_c_CC_p", "BC for c at the positive current collector")
        x_p = eq.DistributeOnDomain(self.x_p, eUpperBound)
        eq.Residual = d(self.c(x_p), self.x_p, eCFDM)

        # Electrolyte: current collector BC's on phi in electrolyte:
        eq = self.CreateEquation("BC_phi_CC_n", "BC for phi at the negative current collector")
        x_n = eq.DistributeOnDomain(self.x_n, eLowerBound)
        eq.Residual = self.phi2_n(x_n)  # arbitrary datum for phi field
        eq = self.CreateEquation("BC_phi_CC_p", "BC for phi at the positive current collector")
        x_p = eq.DistributeOnDomain(self.x_p, eUpperBound)
        eq.Residual = d(self.phi2_p(x_n), self.x_p, eCFDM)

        # TODO:
        #  figure out continuity to link the sections

        # Electrode: charge conservation:
        # We assume infinite conductivity in the electron conducting phase for simplicity
        # negative
        eq = self.CreateEquation("phi1_n")
        x_n = eq.DistributeOnDomain(self.x_n, eOpenOpen)
        eq.Residual = d2(self.phi1_n(x_n), self.x_n, eCFDM)
        # At current collector, phi1 = phiCC
        eq = self.CreateEquation("phi1_n_left")
        x_n = eq.DistributeOnDomain(self.x_n, eLowerBound)
        eq.Residual = self.phi1_n(x_n) - self.phiCC_n()
        # At electrode-separator interface, no electric current can pass, so dphi/dx = 0
        eq = self.CreateEquation("phi1_n_right")
        x_n = eq.DistributeOnDomain(self.x_n, eUpperBound)
        eq.Residual = d(self.phi1_n(x_n), self.x_n, eCFDM)
        # positive
        eq = self.CreateEquation("phi1_p")
        x_p = eq.DistributeOnDomain(self.x_p, eOpenOpen)
        eq.Residual = d2(self.phi1_p(x_p), self.x_p, eCFDM)
        # At electrode-separator interface, no electric current can pass, so dphi/dx = 0
        eq = self.CreateEquation("phi1_n_left")
        x_n = eq.DistributeOnDomain(self.x_n, eLowerBound)
        eq.Residual = d(self.phi1_n(x_n), self.x_n, eCFDM)
        # At current collector, phi1 = phiCC
        eq = self.CreateEquation("phi1_n_right")
        x_n = eq.DistributeOnDomain(self.x_n, eUpperBound)
        eq.Residual = self.phi1_n(x_n) - self.phiCC_n()

        # Define the total current.
        eq = self.CreateEquation("Total_Current")
        eq.Residual = self.current()
        # TODO: Substract integral of a_p*j_p

        # Define the measured voltage
        eq = self.CreateEquation("Voltage")
        eq.Residual = self.phiCC_p() - self.phiCC_n()

        pinfo = self.process_info
        tend, tramp = pinfo["tend"], pinfo["tramp"]
        if pinfo["profileType"] == "CC":
            # Total Current Constraint Equation
            eq = self.CreateEquation("Total_Current_Constraint")
            eq.Residual = self.current() - self.currset()*(1 - np.exp(-dae.Time()/(tend*tramp)))
        elif pinfo["profileType"] == "CV":
            # Keep applied potential constant
            eq = self.CreateEquation("applied_potential")
            eq.Residual = self.V() - self.Vset()*(1 - np.exp(-dae.Time()/(tend*tramp)))

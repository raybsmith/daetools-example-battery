from daetools.pyDAE import *
import numpy as np

from dae.pyUnits import m, s, K, mol, J, A
V = J/(A*s)

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
        self.j_0 = daeParameter("j_0", mol/(m**2 * s), self, "Exchange current density / F")
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
        self.current = dae.daeVariable("current", dae.no_t, self, "Total current of the cell")

        # Parameters
        self.F = daeParameter("F", A*s/mol, self, "Faraday's constant")
        self.R = daeParameter("R", J/(mol*K), self, "Gas constant")
        self.T = daeParameter("T", K, self, "Temperature")
        self.a_n = daeParameter("a_n", 1/m, "Reacting area per electrode volume, negative electrode")
        self.a_p = daeParameter("a_p", 1/m, "Reacting area per electrode volume, positive electrode")
        self.L_n = daeParameter("L_n", m, "Length of negative electrode")
        self.L_s = daeParameter("L_s", m, "Length of separator")
        self.L_p = daeParameter("L_p", m, "Length of positive electrode")
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
        pinfo = self.process_info
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
            "massCons_s", self.x_s, self.c_s, self.phi2_s, self.poros_s(), self.BruggExp_s())
        i = i_lyte(kappa_eff, dphidx, t_p(c), thermodynamic_factor(c), c, dcdx)
        t_m = 1 - t_p(c)
        dt_mdx = d(t_m, self.x_s, eCFDM)
        didx = d(i, self.x_s, eCFDM)
        eq.Residual = dt(c) - (d(D_eff*dcdx) + (t_m*didx + i*dt_mdx)/self.F())
        # charge
        eq, c, phi, dcdx, dphidx, kappa_eff, D_eff = set_up_cons_eq(
            "chargeCons_s", self.x_s, self.c_s, self.phi2_s, self.poros_s(), self.BruggExp_s())
        i = i_lyte(kappa_eff, dphidx, t_p(c), thermodynamic_factor(c), c, dcdx)
        eq.Residual = d(i, self.x_s, eCFDM)

        # Electrolyte: mass and charge conservation in negative electrode
        # mass
        eq, c, phi, dcdx, dphidx, kappa_eff, D_eff = set_up_cons_eq(
            "massCons_n", self.x_n, self.c_n, self.phi2_n, self.poros_n(), self.BruggExp_n())
        i = i_lyte(kappa_eff, dphidx, t_p(c), thermodynamic_factor(c), c, dcdx)
        t_m = 1 - t_p(c)
        dt_mdx = d(t_m, self.x_s, eCFDM)
        didx = d(i, self.x_s, eCFDM)
        eq.Residual = dt(c) - (d(D_eff*dcdx) + (t_m*didx + i*dt_mdx)/self.F())
        # charge
        eq, c, phi, dcdx, dphidx, kappa_eff, D_eff = set_up_cons_eq(
            "chargeCons_n", self.x_n, self.c_n, self.phi2_n, self.poros_n(), self.BruggExp_n())
        i = i_lyte(kappa_eff, dphidx, t_p(c), thermodynamic_factor(c), c, dcdx)
        # Note: How do we properly manage distributed ports in the equation?
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

        # Tie regions together: Continuity of field variables at the electrode-separator interfaces
        N_n, N_s, N_p = self.x_n.NumberOfPoints, self.x_s.NumberOfPoints, self.x_p.NumberOfPoints
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

        # Tie regions together: Conservation equations at the electrode-separator interfaces
        # We use non-uniform finite difference approximations
        def dfdx(f_left, f_cent, f_right, h_left, h_right):
            dfac = 1 + h_right/h_left
            a = -h_right/(h_left**2*dfac)
            b = (h_right/h_left**2 - 1/h_right)/dfac
            c = 1/(h_right*dfac)
            df = a*f_left + b*f_cent + c*f_right
            return df
        h_n = self.L_n() / (N_n - 1)
        h_s = self.L_s() / (N_s - 1)
        h_p = self.L_p() / (N_p - 1)
        # negative-separator
        eff_factor_left = self.poros_n() / (self.poros_n()**self.BruggExp_n())
        eff_factor_cent = eff_factor_left
        eff_factor_right = self.poros_s() / (self.poros_s()**self.BruggExp_s())
        c_left = self.c_n(N_n - 2)
        c_cent = self.c_n(N_n - 1)
        c_right = self.c_s(1)
        D_eff_left = eff_factor_left * D(c_left)
        D_eff_cent = eff_factor_cent * D(c_cent)
        D_eff_right = eff_factor_right * D(c_right)
        kappa_eff_left = eff_factor_left * kappa(c_left)
        kappa_eff_cent = eff_factor_cent * kappa(c_cent)
        kappa_eff_right = eff_factor_right * kappa(c_right)
        dc_left = dfdx(self.c_n(N_n - 3), self.c_n(N_n - 2), self.c_n(N_n - 1), h_n, h_n)
        dc_cent = dfdx(self.c_n(N_n - 2), self.c_n(N_n - 1), self.c_s(1), h_n, h_s)
        dc_right = dfdx(self.c_n(N_n - 1), self.c_s(1), self.c_s(2), h_s, h_s)
        dphi_left = dfdx(self.phi2_n(N_n - 3), self.phi2_n(N_n - 2), self.phi2_n(N_n - 1), h_n, h_n)
        dphi_cent = dfdx(self.phi2_n(N_n - 2), self.phi2_n(N_n - 1), self.phi2_s(1), h_n, h_s)
        dphi_right = dfdx(self.phi2_n(N_n - 1), self.phi2_s(1), self.phi2_s(2), h_s, h_s)
        i_left = i_lyte(kappa_eff_left, dphi_left, t_p(c_left), thermodynamic_factor(c_left), c_left, dc_left)
        i_cent = i_lyte(kappa_eff_cent, dphi_cent, t_p(c_cent), thermodynamic_factor(c_cent), c_cent, dc_cent)
        i_right = i_lyte(kappa_eff_right, dphi_right, t_p(c_right), thermodynamic_factor(c_right), c_right, dc_right)
        t_m_left = 1 - t_p(c_left)
        t_m_cent = 1 - t_p(c_cent)
        t_m_right = 1 - t_p(c_right)
        dDc = dfdx(D_eff_left*dc_left, D_eff_cent*dc_cent, D_eff_right*dc_right, h_n, h_s)
        di = dfdx(i_left, i_cent, i_right, h_n, h_s)
        dt_m = dfdx(t_m_left, t_m_cent, t_m_right, h_n, h_s)
        eq = self.CreateEquation("massCons_ns")
        eq.Residual = dt(self.c_n(N_n - 1)) - (dDc + (t_m_cent*di + i_cent*dt_m) / self.F())
        eq = self.CreateEquation("chargeCons_ns")
        eq.Residual = di - self.a_n()*self.portIn_n(N_n - 1).j_p
        # separator-positive
        eff_factor_left = self.poros_s() / (self.poros_s()**self.BruggExp_s())
        eff_factor_cent = self.poros_p() / (self.poros_p()**self.BruggExp_p())
        eff_factor_right = eff_factor_cent
        c_left = self.c_s(N_n - 2)
        c_cent = self.c_p(0)
        c_right = self.c_p(1)
        D_eff_left = eff_factor_left * D(c_left)
        D_eff_cent = eff_factor_cent * D(c_cent)
        D_eff_right = eff_factor_right * D(c_right)
        kappa_eff_left = eff_factor_left * kappa(c_left)
        kappa_eff_cent = eff_factor_cent * kappa(c_cent)
        kappa_eff_right = eff_factor_right * kappa(c_right)
        dc_left = dfdx(self.c_s(N_n - 3), self.c_s(N_n - 2), self.c_p(0), h_s, h_s)
        dc_cent = dfdx(self.c_s(N_n - 2), self.c_p(0), self.c_p(1), h_s, h_p)
        dc_right = dfdx(self.c_p(0), self.c_p(1), self.c_p(2), h_p, h_p)
        dphi_left = dfdx(self.phi2_n(N_n - 3), self.phi2_n(N_n - 2), self.phi2_p(0), h_s, h_s)
        dphi_cent = dfdx(self.phi2_n(N_n - 2), self.phi2_p(0), self.phi2_p(1), h_s, h_p)
        dphi_right = dfdx(self.phi2_p(0), self.phi2_p(1), self.phi2_p(2), h_p, h_p)
        i_left = i_lyte(kappa_eff_left, dphi_left, t_p(c_left), thermodynamic_factor(c_left), c_left, dc_left)
        i_cent = i_lyte(kappa_eff_cent, dphi_cent, t_p(c_cent), thermodynamic_factor(c_cent), c_cent, dc_cent)
        i_right = i_lyte(kappa_eff_right, dphi_right, t_p(c_right), thermodynamic_factor(c_right), c_right, dc_right)
        t_m_left = 1 - t_p(c_left)
        t_m_cent = 1 - t_p(c_cent)
        t_m_right = 1 - t_p(c_right)
        dDc = dfdx(D_eff_left*dc_left, D_eff_cent*dc_cent, D_eff_right*dc_right, h_s, h_p)
        di = dfdx(i_left, i_cent, i_right, h_s, h_p)
        dt_m = dfdx(t_m_left, t_m_cent, t_m_right, h_s, h_p)
        eq = self.CreateEquation("massCons_sp")
        eq.Residual = dt(self.c_p(0)) - (dDc + (t_m_cent*di + i_cent*dt_m) / self.F())
        eq = self.CreateEquation("chargeCons_sp")
        eq.Residual = di - self.a_p()*self.portIn_p(0).j_p

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
        # TODO: Substract integral of F*a_p*j_p

        # Define the measured voltage
        eq = self.CreateEquation("Voltage")
        eq.Residual = self.phiCC_p() - self.phiCC_n()

        tend, tramp = pinfo["tend"], pinfo["tramp"]
        if pinfo["profileType"] == "CC":
            # Total Current Constraint Equation
            eq = self.CreateEquation("Total_Current_Constraint")
            eq.Residual = self.current() - self.currset()*(1 - np.exp(-dae.Time()/(tend*tramp)))
        elif pinfo["profileType"] == "CV":
            # Keep applied potential constant
            eq = self.CreateEquation("applied_potential")
            eq.Residual = self.V() - self.Vset()*(1 - np.exp(-dae.Time()/(tend*tramp)))


class SimBattery(dae.daeSimulation):
    def __init__(self, process_info):
        dae.daeSimulation.__init__(self)
        # Define the model we're going to simulate
        self.m = mod.ModCell("ModCell", process_info=process_info)
        self.L_n = 100e-6 * m
        self.L_s = 80e-6 * m
        self.L_p = 100e-6 * m
        self.Rp_n = 10e-6 * m
        self.Rp_p = 10e-6 * m
        self.csmax_n = 13e3 * mol/m**3
        self.csmax_p = 5e3 * mol/m**3
        ff0_n = 0.01
        ff0_p = 0.99

    def SetUpParametersAndDomains(self):
        # Domains in ModCell
        self.m.x_n.CreateStructuredGrid(15, 0, L_n)
        self.m.x_s.CreateStructuredGrid(16, 0, L_s)
        self.m.x_p.CreateStructuredGrid(17, 0, L_p)
        # Domains in each particle
        for indx_n in range(self.m.x_n.NumberOfPoints):
            self.m.particle_n(indx_n).r.CreateStructuredGrid(20, 0, self.Rp_n)
        for indx_p in range(self.m.x_p.NumberOfPoints):
            self.m.particle_p(indx_p).r.CreateStructuredGrid(21, 0, self.Rp_p)
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
        self.m.a_n.SetValue((1-self.m.poros_n())*3/self.Rp_n)
        self.m.a_p.SetValue((1-self.m.poros_p())*3/self.Rp_p)
        self.m.currset.SetValue(1e-4 * A/m**3)
        self.m.Vset.SetValue(1.9 * V)
        # Parameters in each particle
        for indx_n in range(self.m.x_n.NumberOfPoints):
            p = self.m.particle_n(indx_n)
            N = p.r.NumberOfPoints
            rvec = np.empty(N, dtype=object)
            rvec[:] = np.linspace(0, Rp_n, N) * m
            p.w.SetValues(rvec**2)
            p.j_0.SetValue(1e-4 * mol/(m**2 * s))
            p.alpha.SetValue(0.5)
        for indx_p in range(self.m.x_p.NumberOfPoints):
            p = self.m.particle_p(indx_p)
            N = p.r.NumberOfPoints
            rvec = np.empty(N, dtype=object)
            rvec[:] = np.linspace(0, Rp_p, N) * m
            p.w.SetValues(rvec**2)
            p.j_0.SetValue(1e-4 * mol/(m**2 * s))
            p.alpha.SetValue(0.5)

    def SetUpVariables(self):
        # ModCell
        for indx_x_n in range(1, self.m.x_n.NumberOfPoints-1):
            self.m.c_n(indx_x_n).SetInitialCondition(indx_x_n, 1e3 * mol/m**3)
            self.m.phi1_n(indx_x_n).SetInitialGuess(indx_x_n, U_n(ff0_n*csmax_n))
        for indx_x_s in range(1, self.m.x_s.NumberOfPoints-1):
            self.m.c_s(indx_x_s).SetInitialCondition(indx_x_s, 1e3 * mol/m**3)
        for indx_x_p in range(1, self.m.x_p.NumberOfPoints-1):
            self.m.c_p(indx_x_p).SetInitialCondition(indx_x_p, 1e3 * mol/m**3)
            self.m.phi1_p(indx_x_p).SetInitialGuess(indx_x_p, U_p(ff0_p*csmax_p))
        self.m.phiCC_n.SetInitialGuess(U_n(ff0_n*csmax_n))
        self.m.phiCC_p.SetInitialGuess(U_p(ff0_p*csmax_p))
        # particles
        for indx_n in range(self.m.x_n.NumberOfPoints):
            p = self.m.particle_n(indx_n)
            for indx_r in range(1, p.r.NumberOfPoints-1):
                p.c.SetInitialCondition(indx_r, ff0_n*csmax_n)
        for indx_p in range(self.m.x_p.NumberOfPoints):
            p = self.m.particle_p(indx_p)
            for indx_r in range(1, p.r.NumberOfPoints-1):
                p.c.SetInitialCondition(indx_r, ff0_p*csmax_p)

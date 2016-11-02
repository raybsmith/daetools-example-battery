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
                "tend": 1e4 * s,
                }

def kappa(c):
    """Return the conductivity of the electrolyte in S/m as a function of concentration in M."""
    out = 0.1  # S/m
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
    out = 1e-12 * y/y  # m**2/s
    return out

def Ds_p(y):
    """Return diffusivity (in m^2/s) as a function of solid filling fraction, y."""
    out = 1e-12 * y/y  # m**2/s
    return out

def U_n(y):
    """Return the equilibrium potential (V vs Li) of the negative electrode active material
    as a function of solid filling fraction, y.
    """
    out = 0.0  # V
    return out

def U_p(y):
    """Return the equilibrium potential (V vs Li) of the positive electrode active material
    as a function of solid filling fraction, y.
    """
    out = 0.000  # V
    return out

class ModParticle(daeModel):
    def __init__(self, Name, pindx_1, pindx_2, c_2, phi_2, phi_1, Ds, U, Parent=None, Description=""):
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
        self.R = daeParameter("R", m, self, "Radius of particle")

        self.pindx_1 = pindx_1
        self.pindx_2 = pindx_2
        self.phi_2 = phi_2
        self.c_2 = c_2
        self.phi_1 = phi_1

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        eq = self.CreateEquation("mass_cons")
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
        eta = self.phi_1(self.pindx_1) - self.phi_2(self.pindx_2) - self.U_ref()*self.U(c_surf/self.c_ref())
        eta_ndim = eta / self.V_thermal()
#        eq.Residual = self.j_p() - self.j_0() * (np.exp(-self.alpha()*eta_ndim) - np.exp((1 - self.alpha())*eta_ndim))
        eq.Residual = self.j_p() + self.j_0() * eta_ndim

class ModCell(daeModel):
    def __init__(self, Name, Parent=None, Description="", process_info=process_info):
        daeModel.__init__(self, Name, Parent, Description)
        self.process_info = process_info

        # Domains where variables are distributed
        self.x_centers_n = daeDomain("x_centers_n", self, m, "X cell-centers domain in negative electrode")
        self.x_centers_p = daeDomain("x_centers_p", self, m, "X cell-centers domain in positive electrode")
        self.x_centers_full = daeDomain("x_centers_full", self, m, "X cell-centers domain over full cell")
        self.x_faces_full = daeDomain("x_faces_full", self, m, "X cell-faces domain over full cell")

        # Variables
        # Concentration/potential in different regions of electrolyte and electrode
        self.c = daeVariable("c", conc_t, self, "Concentration in the elyte")
        self.phi2 = daeVariable("phi2", elec_pot_t, self, "Electric potential in the elyte")
        self.i2 = daeVariable("i2", current_dens_t, self, "Electrolyte current density")
        self.c.DistributeOnDomain(self.x_centers_full)
        self.phi2.DistributeOnDomain(self.x_centers_full)
        self.i2.DistributeOnDomain(self.x_faces_full)
        self.phi1_n = daeVariable("phi1_n", elec_pot_t, self, "Electric potential in bulk sld, negative")
        self.phi1_p = daeVariable("phi1_p", elec_pot_t, self, "Electric potential in bulk sld, positive")
        self.phi1_n.DistributeOnDomain(self.x_centers_n)
        self.phi1_p.DistributeOnDomain(self.x_centers_p)
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
        self.xval_cells = daeParameter("xval_cells", m, self, "coordinate of cell centers")
        self.xval_faces = daeParameter("xval_faces", m, self, "coordinate of cell faces")
        self.xval_cells.DistributeOnDomain(self.x_centers_full)
        self.xval_faces.DistributeOnDomain(self.x_faces_full)

        # Sub-models
        N_n = self.process_info["N_n"]
        N_s = self.process_info["N_s"]
        N_p = self.process_info["N_p"]
        self.particles_n = np.empty(N_n, dtype=object)
        self.particles_p = np.empty(N_p, dtype=object)
        for indx in range(N_n):
            indx_1 = indx_2 = indx
            self.particles_n[indx] = ModParticle("particle_n_{}".format(indx), indx_1, indx_2, self.c,
                                                 self.phi2, self.phi1_n, Ds_n, U_n, Parent=self)
        for indx in range(N_p):
            indx_1 = indx
            indx_2 = N_n + N_s + indx
            self.particles_p[indx] = ModParticle("particle_p_{}".format(indx), indx_1, indx_2, self.c,
                                                 self.phi2, self.phi1_p, Ds_p, U_p, Parent=self)

    def DeclareEquations(self):
        daeModel.DeclareEquations(self)
        pinfo = self.process_info
        V_thm = self.R() * self.T() / self.F()
        N_n, N_p = self.x_centers_n.NumberOfPoints, self.x_centers_p.NumberOfPoints
        N_centers, N_faces = self.x_centers_full.NumberOfPoints, self.x_faces_full.NumberOfPoints
        N_s = N_centers - N_n - N_p
        center_coords = np.array([self.xval_cells(indx) for indx in range(N_centers)])
        face_coords = np.array([self.xval_faces(indx) for indx in range(N_faces)])
        h_centers = np.hstack((np.diff(center_coords)[0], np.diff(center_coords), np.diff(center_coords)[-1]))
        h_faces = np.diff(face_coords)
        # For convenience, make numpy arrays of variables at cell centers
        phi2 = np.array([self.phi2(indx) for indx in range(N_centers)])
        c = np.array([self.c(indx) for indx in range(N_centers)])
        dcdt = np.array([self.c.dt(indx) for indx in range(N_centers)])
        a = np.hstack((self.a_n()*np.ones(N_n), self.a_ref()*np.ones(N_s), self.a_p()*np.ones(N_p)))
        j_p = np.array([self.particles_n[indx].j_p() for indx in range(N_n)]
                       + N_s*[0 * self.j_ref()]
                       + [self.particles_p[indx].j_p() for indx in range(N_p)])
        eff_factor_tmp = np.hstack((self.poros_n() / (self.poros_n()**self.BruggExp_n()) * np.ones(N_n+1),
                                    self.poros_s() / (self.poros_s()**self.BruggExp_s()) * np.ones(N_s),
                                    self.poros_p() / (self.poros_p()**self.BruggExp_p()) * np.ones(N_p+1)))
        eff_factor = (2*eff_factor_tmp[1:]*eff_factor_tmp[:-1]) / (eff_factor_tmp[1:] + eff_factor_tmp[:-1])
        poros = np.hstack((self.poros_n()*np.ones(N_n),
                           self.poros_s()*np.ones(N_s),
                           self.poros_p()*np.ones(N_p)))

        # Boundary conditions on c and phi2 at current collectors.
        # To do these, create "ghost points" on the end of cell-center vectors
        ctmp = np.empty(N_centers + 2, dtype=object)
        ctmp[1:-1] = c
        phi2tmp = np.empty(N_centers + 2, dtype=object)
        phi2tmp[1:-1] = phi2
        # No ionic current passes into the current collectors, which requires
        # grad(c) = grad(phi2) = 0
        # at both current collectors. We apply this by using the ghost points.
        ctmp[0] = ctmp[1]
        ctmp[-1] = ctmp[-2]
        phi2tmp[0] = phi2tmp[1]
        phi2tmp[-1] = phi2tmp[-2]
        # We'll need the value of c at the faces as well. We use a harmonic mean.
        c_faces = (2*ctmp[1:]*ctmp[:-1])/(ctmp[1:] + ctmp[:-1])

        # Approximate the gradients of these field variables at the faces
        dc = np.diff(ctmp) / h_centers
        dlogc = np.diff(np.log(ctmp / self.c_ref())) / h_centers
        dphi2 = np.diff(phi2tmp) / h_centers

        # Effective transport properties are required at faces between cells
        D_eff = eff_factor * self.D_ref() * D(c_faces / self.c_ref())
        kappa_eff = eff_factor * self.cond_ref() * kappa(c_faces / self.c_ref())

        # Flux of charge (current density) at faces
        i = -kappa_eff * (dphi2 - 2*V_thm*(1 - t_p(c_faces))*thermodynamic_factor(c_faces)*dlogc)
        # Flux of anions at faces
        N_m = -D_eff*dc - (1 - t_p(c_faces)) * i / self.F()

        # Store values for the current density
        for indx in range(N_faces):
            eq = self.CreateEquation("i2_{}".format(indx))
            eq.Residual = self.i2(indx) - i[indx]
        # Divergence of fluxes
        di = np.diff(i) / h_faces
        dN_m = np.diff(N_m) / h_faces
        # Electrolyte: mass and charge conservation
        for indx in range(N_centers):
            eq = self.CreateEquation("mass_cons_m_{}".format(indx), "anion mass conservation")
            eq.Residual = poros[indx]*dcdt[indx] + dN_m[indx]
            eq = self.CreateEquation("charge_cons_{}".format(indx), "charge conservation")
            eq.Residual = -di[indx] - self.F()*a[indx]*j_p[indx]

        # Arbitrary datum for electric potential.
        # We apply this in the electrolyte at an arbitrary location, the negative current collector
        eq = self.CreateEquation("phi2_datum")
#        eq.Residual = phi2[0]
        eq.Residual = self.phiCC_n()

        # Electrode: charge conservation
        phi1_n = np.array([self.phi1_n(indx) for indx in range(N_n)])
        phi1_p = np.array([self.phi1_p(indx) for indx in range(N_p)])
        # We assume infinite conductivity in the electron conducting phase for simplicity
        # negative
        for indx in range(N_n):
            eq = self.CreateEquation("phi1_n_{}".format(indx))
            eq.Residual = phi1_n[indx] - self.phiCC_n()
        for indx in range(N_p):
            eq = self.CreateEquation("phi1_p_{}".format(indx))
            eq.Residual = phi1_p[indx] - self.phiCC_p()

        # Define the total current.
        eq = self.CreateEquation("Total_Current")
        eq.Residual = self.current() + np.sum(self.F()*a[:N_n]*j_p[:N_n]*h_centers[:N_n])

        # Define the measured voltage
        eq = self.CreateEquation("Voltage")
        eq.Residual = self.V() - (self.phiCC_p() - self.phiCC_n())

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
        self.L_n = 300e-6 * m
        self.L_s = 200e-6 * m
        self.L_p = 300e-6 * m
        self.L_tot = self.L_n + self.L_s + self.L_p
        self.N_n = 10
        self.N_s = 10
        self.N_p = 10
        self.NR_n = 10
        self.NR_p = 10
        self.R_n = 1e-6 * m
        self.R_p = 1e-6 * m
        self.csmax_n = 13e3 * mol/m**3
        self.csmax_p = 5e3 * mol/m**3
        self.ff0_n = 0.99
        self.ff0_p = 0.01
        self.process_info = process_info
        self.process_info["N_n"] = self.N_n
        self.process_info["N_s"] = self.N_s
        self.process_info["N_p"] = self.N_p
        self.m = ModCell("ModCell", process_info=self.process_info)

    def SetUpParametersAndDomains(self):
        h_n = self.L_n / self.N_n
        h_s = self.L_s / self.N_s
        h_p = self.L_p / self.N_p
        xvec_centers_n = [h_n*(0.5 + indx) for indx in range(self.N_n)]
        xvec_centers_s = [self.L_n + h_s*(0.5 + indx) for indx in range(self.N_s)]
        xvec_centers_p = [(self.L_n + self.L_s) + h_p*(0.5 + indx) for indx in range(self.N_p)]
        xvec_centers = xvec_centers_n + xvec_centers_s + xvec_centers_p
        xvec_faces = [0 * m] + [h_n*(1 + indx) for indx in range(self.N_n)]
        xvec_faces += [self.L_n + h_s*(1 + indx) for indx in range(self.N_s)]
        xvec_faces += [(self.L_n + self.L_s) + h_p*(1 + indx) for indx in range(self.N_p)]
#        cvals = [val.value for val in xvec_centers]
#        fvals = [val.value for val in xvec_faces]
#        print("Cell centers:\n", cvals)
#        print("Cell faces:\n", fvals)
#        print("cell-center-spacing:\n", np.diff(np.array(cvals)))
#        print("cell-widths:\n", np.diff(np.array(fvals)))
#        zz
        # Domains in ModCell
        self.m.x_centers_n.CreateStructuredGrid(self.N_n - 1, 0, 1)
        self.m.x_centers_p.CreateStructuredGrid(self.N_p - 1, 0, 1)
        self.m.x_centers_full.CreateStructuredGrid(self.N_n + self.N_s + self.N_p - 1, 0, 1)
        self.m.x_faces_full.CreateStructuredGrid(self.N_n + self.N_s + self.N_p, 0, 1)
        self.m.x_centers_n.Points = [x.value for x in xvec_centers_n]
        self.m.x_centers_p.Points = [x.value for x in xvec_centers_p]
        self.m.x_centers_full.Points = [x.value for x in xvec_centers]
        self.m.x_faces_full.Points = [x.value for x in xvec_faces]
        # Domains in each particle
        for indx_n in range(self.m.x_centers_n.NumberOfPoints):
            self.m.particles_n[indx_n].r.CreateStructuredGrid(self.NR_n - 1, 0, self.R_n.value)
        for indx_p in range(self.m.x_centers_p.NumberOfPoints):
            self.m.particles_p[indx_p].r.CreateStructuredGrid(self.NR_p - 1, 0, self.R_p.value)
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
        self.m.a_n.SetValue((1-self.m.poros_n.GetValue())*3/self.R_n)
        self.m.a_p.SetValue((1-self.m.poros_p.GetValue())*3/self.R_p)
        self.m.D_ref.SetValue(1 * m**2/s)
        self.m.cond_ref.SetValue(1 * S/m)
        self.m.c_ref.SetValue(1000 * mol/m**3)
        self.m.j_ref.SetValue(1 * mol/(m**2 * s))
        self.m.a_ref.SetValue(1 * m**(-1))
        self.m.currset.SetValue(1e+1 * A/m**2)
        self.m.Vset.SetValue(1.9 * V)
        self.m.tau_ramp.SetValue(1e-1 * process_info["tend"])
        self.m.xval_cells.SetValues(np.array(xvec_centers))
        self.m.xval_faces.SetValues(np.array(xvec_faces))
        # Parameters in each particle
        for indx_n in range(self.m.x_centers_n.NumberOfPoints):
            p = self.m.particles_n[indx_n]
            N = p.r.NumberOfPoints
            rvec = np.empty(N, dtype=object)
            rvec[:] = np.linspace(0, self.R_n.value, N) * m
            p.w.SetValues(rvec**2)
            p.j_0.SetValue(1e-4 * mol/(m**2 * s))
            p.alpha.SetValue(0.5)
            p.c_ref.SetValue(self.csmax_n)
            p.D_ref.SetValue(1 * m**2/s)
            p.U_ref.SetValue(1 * V)
            p.V_thermal.SetValue(self.m.R.GetValue()*self.m.T.GetValue()/self.m.F.GetValue())
            p.R.SetValue(self.R_n)
        for indx_p in range(self.m.x_centers_p.NumberOfPoints):
            p = self.m.particles_p[indx_p]
            N = p.r.NumberOfPoints
            rvec = np.empty(N, dtype=object)
            rvec[:] = np.linspace(0, self.R_p.value, N) * m
            p.w.SetValues(rvec**2)
            p.j_0.SetValue(1e-4 * mol/(m**2 * s))
            p.alpha.SetValue(0.5)
            p.c_ref.SetValue(self.csmax_p)
            p.D_ref.SetValue(1 * m**2/s)
            p.U_ref.SetValue(1 * V)
            p.V_thermal.SetValue(self.m.R.GetValue()*self.m.T.GetValue()/self.m.F.GetValue())
            p.R.SetValue(self.R_p)

    def SetUpVariables(self):
        cs0_n = self.ff0_n * self.csmax_n
        cs0_p = self.ff0_p * self.csmax_p
        # ModCell
        for indx in range(self.m.x_centers_full.NumberOfPoints):
            self.m.c.SetInitialCondition(indx, 1e3 * mol/m**3)
        self.m.phi1_n.SetInitialGuesses(U_n(self.ff0_n) * V)
        self.m.phiCC_n.SetInitialGuess(U_n(self.ff0_n) * V)
        self.m.phi1_p.SetInitialGuesses(U_p(self.ff0_p) * V)
        self.m.phiCC_p.SetInitialGuess(U_p(self.ff0_p) * V)
        # particles
        for indx_n in range(self.m.x_centers_n.NumberOfPoints):
            p = self.m.particles_n[indx_n]
            for indx_r in range(1, p.r.NumberOfPoints-1):
                p.c.SetInitialCondition(indx_r, cs0_n)
        for indx_p in range(self.m.x_centers_p.NumberOfPoints):
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

#    # Save the model report and the runtime model report
#    simulation.m.SaveModelReport(simulation.m.Name + ".xml")
#    simulation.m.SaveRuntimeModelReport(simulation.m.Name + "-rt.xml")

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

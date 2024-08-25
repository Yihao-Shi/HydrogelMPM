# =================================== Set path =================================== #
import os, sys
current_path = os.path.realpath(__file__)
directory_path = os.path.dirname(current_path)
sys.path.append(directory_path)

# =================================== Import necessary packages =================================== #
import numpy as np
from pyevtk.hl import pointsToVTK, linesToVTK
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=False)

# =================================== Taichi types =================================== #
vec2f = ti.types.vector(2, float)
vec3f = ti.types.vector(3, float)
mat2x2 = ti.types.matrix(2, 2, float)
mat3x3 = ti.types.matrix(3, 3, float)
particleNum = ti.field(int, shape=())
# mass cut off
val_lim = 1e-15
# math.pi
pi = 3.1415926536

# =================================== User defined parameters =================================== #
# scheme 0-TL, 1-UL
scheme = 0
# open gui?
gui = True
# the order of b-spline function: 2-quadratic; 3-cubic
order = 2
# alpha pic
alpha = 1.
# the length along x- or y-axes
base = 5
# background damping
damp = 0.5
# the number of particle per cell in 1D (2D: npic * npic)
npic = 2
# sphere parameters
xcenter, ycenter = 2.5, 1.5
radius = 0.5
# the length of the cell and its inverse
dx, inv_dx = 0.02, 50
# timestep
dt = 1e-4
# particle density
p_rho = 1050
# gravity
gravity = vec2f(0., -0.)
# if open f-bar
fbar = True
# initial velocity
IniVel = [0., -0.1]
# simulation time
time = 1.25

# =================================== Hyperelastic model parameters =================================== #
# material type: 0->neo_hookean; 1->mooney_rivlin; 2->generalized_mooney_rivlin; 3->gent; 4->gent_gent; 5->gent_hydrogel
materialID = 5
# Young's modulus and Poisson's ratio
e, nu = 10000, 0.499
# lame constants
mu = 0.5 * e / (1 + nu)
la = e * nu / (1 + nu) / (1 - 2. * nu)
# mooney rivlin parameters
concentration = None
a10 = 0.2 * mu
a01 = 0.5 * mu - a10
a11 = .1 * mu
if concentration is not None:
    a10 = 1000 * (-12.76 + 53.67 * concentration ** 1.5)
    a01 = 1000 * (10.75 - 38.27 * concentration ** 1.5)
    a11 = 1000 * (-16.27 + 23.72 * concentration ** 1.5)
# gent model (tensile limit)
Jm = 10000.
c2 = 12.
# Hydrogel model
N = 1e25
T = 300
v = 1e-28
KT = 4.14e-21
Nv = N * v
chi = 0.2
lambda0 = 1.5
u = 0.
lambda0_1 = 1. / lambda0
lambda0_3 = 1. / lambda0 ** 3

# characteristic velocity/time
cv = np.sqrt(e / p_rho)
ct = 0.5 * radius / cv

# =================================== Linear contact parameters =================================== #
# stiffness parameter
kappa = 1e6
# friction parameter
fric = 0.0
# viscous damping coefficient
vratio = 0.4

# =================================== Pre-possessing =================================== #
# the left down corner of the bounding box 
px1, py1 = xcenter-radius, ycenter-radius
# the dimension of the bounding box
lx1, ly1 = 2.*radius, 2.*radius
# The number of activated cells
ex1, ey1 = lx1 * inv_dx, ly1 * inv_dx
# particle volume
p_vol = dx * dx / npic / npic 
# particle mass
p_mass = p_vol * p_rho
# particle radius
p_rad = dx / npic / 2.
# particle diameter
pdx = dx / npic
# The number of material points
n_particles = int(ex1 * ey1 * npic * npic)
# The number of total nodes
grid_x, grid_y = int(base * inv_dx + 1), int(base * inv_dx + 1)
# The number of total cells
cell_x, cell_y = int(base * inv_dx), int(base * inv_dx)

# =================================== Shape functions =================================== #
# ========================================================= #
#           Quadratic B-spline shape function               #
# ========================================================= #
@ti.func
def ShapeBsplineQ(xp, xg, idx, btype):
    nx = 0.
    d = (xp - xg) * idx
    if btype == 0:
        if d >= 0.5 and d < 1.5:
            nx = (0.5 * d - 1.5) * d + 1.125
        elif d >= -0.5 and d < 0.5:
            nx = -d * d + 0.75
        elif d >= -1.5 and d < 0.5:
            nx = (0.5 * d + 1.5) * d + 1.125
    elif btype == 1:
        if d >= 0. and d < 0.5:
            nx = 1. - d
        elif d >= 0.5 and d < 1.5:
            nx = (0.5 * d - 1.5) * d + 1.125
    elif btype == 2:
        if d >= -1. and d < -0.5:
            nx = 1. + d
        elif d >= -0.5 and d < 0.5:
            nx = -d * d + 0.75
        elif d >= 0.5 and d < 1.5:
            nx = (0.5 * d - 1.5) * d + 1.125
    elif btype == 3:
        if d >= -1.5 and d < -0.5:
            nx = (0.5 * d + 1.5) * d + 1.125
        elif d >= -0.5 and d < 0.5:
            nx = -d * d + 0.75
        elif d >= 0.5 and d < 1.:
            nx = 1. - d
    elif btype == 4:
        if d >= -1.5 and d < -0.5:
            nx = (0.5 * d + 1.5) * d + 1.125
        elif d >= -0.5 and d <= 0.:
            nx = 1. + d
    return nx

@ti.func
def GShapeBsplineQ(xp, xg, idx, btype):
    dnx = 0.
    d = (xp - xg) * idx
    if btype == 0:
        if d >= 0.5 and d < 1.5:
            dnx = d - 1.5
        elif d >= -0.5 and d < 0.5:
            dnx = -2. * d 
        elif d >= -1.5 and d < 0.5:
            dnx = d + 1.5
    elif btype == 1:
        if d >= 0. and d < 0.5:
            dnx = -1.
        elif d >= 0.5 and d < 1.5:
            dnx = d - 1.5
    elif btype == 2:
        if d >= -1. and d < -0.5:
            dnx = 1. 
        elif d >= -0.5 and d < 0.5:
            dnx = -2. * d
        elif d >= 0.5 and d < 1.5:
            dnx = d - 1.5
    elif btype == 3:
        if d >= -1.5 and d < -0.5:
            dnx = d + 1.5
        elif d >= -0.5 and d < 0.5:
            dnx = -2. * d
        elif d >= 0.5 and d < 1.:
            dnx = -1.
    elif btype == 4:
        if d >= -1.5 and d < -0.5:
            dnx = d + 1.5
        elif d >= -0.5 and d <= 0.:
            dnx = 1. 
    return dnx * idx

# ========================================================= #
#             Cubic B-spline shape function                 #
# ========================================================= #
@ti.func
def ShapeBsplineC(xp, xg, idx, btype):
    nx = 0.
    d = (xp - xg) * idx
    if d >= 1 and d < 2:
        if btype != 3:
            nx = ((-1./6. * d + 1) * d - 2) * d + 4.0/3.0
    elif d >= 0 and d < 1:
        if btype == 1:
            nx = (1./6. * d * d - 1) * d + 1
        elif btype == 3:
            nx = (1./3. * d - 1) * d * d + 2./3.
        else:
            nx = (0.5 * d - 1) * d * d + 2./3.
    elif d >= -1 and d < 0:
        if btype == 4:
            nx = (-1./6. * d * d + 1) * d + 1
        elif btype == 2:
            nx = (-1./3. * d - 1) * d * d + 2./3.
        else:
            nx = (-0.5 * d - 1) * d * d + 2./3.
    elif d >= -2 and d < -1:
        nx = ((1./6. * d + 1) * d + 2) * d + 4./3.
    return nx

@ti.func
def GShapeBsplineC(xp, xg, idx, btype):
    dnx = 0.
    d = (xp - xg) * idx
    if d >= 1 and d < 2:
        if btype != 3:
            dnx = (-0.5 * d + 2) * d - 2.
    elif d >= 0 and d < 1:
        if btype == 1:
            dnx = 0.5 * d * d - 1.
        elif btype == 3:
            dnx = d * (d - 2.)
        else:
            dnx = (3.0 / 2.0 * d - 2) * d
    elif d >= -1 and d < 0:
        if btype == 4:
            dnx = -0.5 * d * d + 1.
        elif btype == 2:
            dnx = (-d - 2.) * d
        else:
            dnx = (-3./2. * d - 2.) * d
    elif d >= -2 and d < -1:
        dnx = (0.5 * d + 2.) * d + 2.
    return dnx * idx

# The influenced node for each particles in the process of mapping state variables (1D)
influenced_node = 3
# shape function
shapeFN = ShapeBsplineQ
# gradient of shape function
dshapeFN = GShapeBsplineQ
if order == 3:
    influenced_node = 4
    shapeFN = ShapeBsplineC
    dshapeFN = GShapeBsplineC

# =================================== Essential data class =================================== #
@ti.dataclass
class Particle:
    x0: vec2f          # Initial position
    x: vec2f           # position
    v: vec2f           # velocity
    vol: float         # volume
    m: float           # mass
    cf: vec2f          # contact force
    stress: mat3x3     # stress tensor
    gradv: mat2x2      # velocity gradient
    td: mat2x2         # deformation gradient (modified)

@ti.dataclass
class Grid:
    m: float          # mass
    p: vec2f          # momentum
    f: vec2f          # force
    jac: float        # mapped jacobian

@ti.dataclass
class Line:           # ground boundary
    x0: vec2f         # left point
    x1: vec2f         # right point
    v: vec2f          # velocity
    norm: vec2f       # normal

# influenced node ID for each particle
LnID = ti.field(int)
# the shape function values
shape = ti.field(float)
# the gradient shape function values
dshape = ti.Vector.field(2, float)
ti.root.dense(ti.ij, (n_particles, int(influenced_node * influenced_node))).place(LnID, shape, dshape)
# the number of influenced nodes for each particle
offset = ti.field(int, shape=n_particles)
# determine the proper type of b-spline function to use 
gtype = ti.Vector.field(2, int, shape=((grid_x)*(grid_y)))

# particle data class
p = Particle.field(shape=n_particles)
# grid dataclass
g = Grid.field(shape=grid_x * grid_y)
# ground boundary dataclass
l = Line.field(shape=2)

# =================================== Utils =================================== #
@ti.func
def sign(x):
    if x != 0:
        x /= ti.abs(x)
    return x 

@ti.func
def highdim(mat):
    return mat3x3([mat[0, 0], mat[0, 1], 0.], [mat[1, 0], mat[1, 1], 0.], [0., 0., 1.])

# =================================== Initialization =================================== #
@ti.kernel
def line_init():
    l[0].x0 = [0, py1]
    l[0].x1 = [base, py1]
    l[0].v = [0., 0.0]
    l[0].norm = [0., 1.]
    l[1].x0 = [0, py1+ly1]
    l[1].x1 = [base, py1+ly1]
    l[1].v = vec2f(IniVel)
    l[1].norm = [0., -1.]

@ti.func
def rectangle(x):
    return True

@ti.func
def circle(x):
    return ((x[0] - (px1 + 0.5 * lx1)) / (0.5 * lx1)) ** 2 + ((x[1] - (py1 + 0.5 * ly1)) / (0.5 * ly1)) ** 2 < 1.

@ti.kernel
def sdf_init():                                                             # determine the node type
    for i in range(gtype.shape[0]):
        ix = i % (grid_x)
        iy = i // (grid_x)

        xtype = min(2, ix) - min(grid_x - 1 - ix, 2) 
        ytype = min(2, iy) - min(grid_x - 1 - iy, 2) 
        if xtype < 0: xtype += 3
        elif xtype > 0: xtype += 2
        if ytype < 0: ytype += 3
        elif ytype > 0: ytype += 2
        gtype[i] = [xtype, ytype]

@ti.kernel
def particle_init():                                                         # particle initialize
    for i in range(n_particles):
        aa = i % (ex1 * npic)
        bb = i // (ex1 * npic)
        if not rectangle(vec2f(px1 + dx / npic / 2. + aa * pdx, py1 + dx / npic / 2. + bb * pdx)): continue
        ids = ti.atomic_add(particleNum[None], 1)
        p[ids].x = vec2f(px1 + dx / npic / 2. + aa * pdx, py1 + dx / npic / 2. + bb * pdx)
        p[ids].x0 = p[ids].x
        p[ids].vol = p_vol
        p[ids].m = p_mass
        p[ids].gradv = mat2x2([0, 0], [0, 0])
        if materialID == 6:
            p[ids].td = mat2x2([lambda0, 0], [0, lambda0])
        else:
            p[ids].td = mat2x2([1, 0], [0, 1])

@ti.kernel
def grid_init():
    # mapping particle mass to grid
    for i in range(particleNum[None]):
        # loop over each influenced nodes for each particle
        for j in range(offset[i]):
            g[LnID[i, j]].m += shape[i, j] * p[i].m

# =================================== Hyperelastic constitutive model =================================== #
@ti.func
def I1(td):
    sums = 0.
    for i in ti.static(range(td.n)):
        for j in ti.static(range(td.m)):
            sums += td[i, j] * td[i, j]
    return sums

@ti.func
def I2(td):
    I_1 = I1(td)
    sums = 0.
    for i in ti.static(range(td.n)):
        rowsum = 0.
        for j in ti.static(range(td.m)):
            rowsum += td[i, j] * td[i, j]
        sums += rowsum * rowsum
    return 0.5 * (I_1 ** 2 - sums)

@ti.func
def I3(td):
    return td.determinant() ** 2

@ti.func
def I1dev(AJ, td):
    return AJ ** (-2./3.) * I1(td)

@ti.func
def I2dev(AJ, td):
    return AJ ** (-4./3.) * I2(td)

@ti.func
def I3dev(AJ, td=None):
    return AJ

@ti.func
def dI1dF(td):
    return 2. * td

@ti.func
def dI2dF(td):
    i1 = I1(td)
    return 2. * (td * i1 - td @ td @ td.transpose())

@ti.func
def dJdF(td):
    return td.determinant() * td.inverse().transpose()

@ti.func
def getPK1(dUdI1, dUdI2, dUdJ, td):
    return 2. * (dUdI1 * dI1dF(td) + dUdI2 * dI2dF(td) + dUdJ * dJdF(td))

@ti.func
def neohookean(td):
    AJ = td.determinant()
    dUdI1 = 0.5 * mu
    dUdI2 = 0.
    dUdJ = (la * ti.log(AJ) - mu) / AJ
    return getPK1(dUdI1, dUdI2, dUdJ, td)

@ti.func
def mooney_rivlin(td):
    AJ = td.determinant()
    dUdI1 = a10
    dUdI2 = a01
    dUdJ = (la * ti.log(AJ) - mu) / AJ
    return getPK1(dUdI1, dUdI2, dUdJ, td)

@ti.func
def generalized_mooney_rivlin(td):
    AJ = td.determinant()
    I_1 = I1(td)
    I_2 = I2(td)
    dUdI1 = a10 + a11 * (I_2 - 3.)
    dUdI2 = a01 + a11 * (I_1 - 3.)
    dUdJ = (la * ti.log(AJ) - mu) / AJ
    return getPK1(dUdI1, dUdI2, dUdJ, td)

@ti.func
def gent(td):
    AJ = td.determinant()
    dUdI1 = 0.5 * mu * Jm / (Jm - I1(td) + 3.)
    dUdI2 = 0.
    dUdJ = 0.5 * la * (AJ - 1./AJ)
    return getPK1(dUdI1, dUdI2, dUdJ, td)

@ti.func
def gent_gent(td):
    AJ = td.determinant()
    dUdI1 = 0.5 * mu * Jm / (Jm - I1(td) + 3.)
    dUdI2 = 1.5 * c2 / I2(td)
    dUdJ = 0.5 * la * (AJ - 1./AJ)
    return getPK1(dUdI1, dUdI2, dUdJ, td)

@ti.func
def gent_hydrogel(td):
    # Large deformation and crack propagation analyses of hedrogel by peridynamics. EFM.
    AJ = td.determinant()
    I_1 = I1(td)
    dUdI1 = 0.5 * mu * AJ ** (-2./3.) * Jm / (Jm - I_1 * AJ ** (-2./3.) + 3.)
    dUdI2 = 0.
    dUdJ = -1./3. * mu * AJ ** (-5./3.) * Jm * I_1 / (Jm - I_1 * AJ ** (-2./3.) + 3.) + 0.5 * la * (AJ - 1./AJ)
    return getPK1(dUdI1, dUdI2, dUdJ, td)

@ti.func
def hydrogel(td):
    # A theory of coupled diffusion and large deformation in polymeric gels. JMPS.
    AJ = td.determinant()
    dUdI1 = 0.5 * N * KT * lambda0_1 
    dUdI2 = 0.
    dUdJ = (KT / v - N * KT) / AJ * lambda0_3 + KT / v * (ti.log((AJ - lambda0_3) / (lambda0_3 * AJ)) + chi * lambda0_3 * lambda0_3 / (AJ * AJ)) - u / v
    return getPK1(dUdI1, dUdI2, dUdJ, td)

# =================================== Main kernels =================================== #
@ti.kernel
def calculate_init():
    offset.fill(0)
    for i in range(particleNum[None]):
        pos = p[i].x

        # compute the base node
        base_bound = ti.Vector([0, 0])
        if ti.static(order == 2):
            base_bound = ti.floor(pos * inv_dx - 0.5, int)
        elif ti.static(order == 3):
            base_bound = ti.floor(pos * inv_dx, int) - 1

        for a, b in ti.ndrange(influenced_node, influenced_node):
            grid_idx = base_bound[0] + a
            grid_idy = base_bound[1] + b
            if grid_idx < 0 or grid_idx >= grid_x: continue
            if grid_idy < 0 or grid_idy >= grid_y: continue
            
            # calcualte linearize node ID
            nodeID = int(grid_idx + grid_idy * grid_x)

            # calcualte shape function and its derivative
            sx = shapeFN(pos[0], grid_idx * dx, inv_dx, int(gtype[nodeID][0]))
            sy = shapeFN(pos[1], grid_idy * dx, inv_dx, int(gtype[nodeID][1]))
            gsx = dshapeFN(pos[0], grid_idx * dx, inv_dx, int(gtype[nodeID][0]))
            gsy = dshapeFN(pos[1], grid_idy * dx, inv_dx, int(gtype[nodeID][1]))
            s = sx * sy
            gs = vec2f(gsx * sy, gsy * sx)
            
            # if shape function value equal to zero, pass
            if s <= val_lim: continue

            # calculate shape function values and store it to global memory
            count = ti.atomic_add(offset[i], 1)
            LnID[i, count] = nodeID
            shape[i, count] = s
            dshape[i, count] = gs

@ti.func
def apply_bc(i):                           # apply boundary conditions on nodes
    ig = i % grid_x
    jg = i // grid_x
    '''if jg<=20:
        g[i].p = vec2f(0, 0)
        g[i].f = vec2f(0, 0)'''

@ti.kernel
def solve():
    # update boundaries
    l[0].x0[1] += l[0].v[1] * dt
    l[0].x1[1] += l[0].v[1] * dt
    l[1].x0[1] += l[1].v[1] * dt
    l[1].x1[1] += l[1].v[1] * dt

    # reset nodal variables
    for i in g:
        if ti.static(scheme == 1):
            g[i].m = 0.
        g[i].p = vec2f(0, 0)
        g[i].f = vec2f(0, 0)
        g[i].jac = 0.
    
    if ti.static(scheme == 1):
        # mapping particle mass to grid
        for i in range(particleNum[None]):
            # loop over each influenced nodes for each particle
            for j in range(offset[i]):
                g[LnID[i, j]].m += shape[i, j] * p[i].m

    # calculate normal contact force using linear contact model: f_n = -k_n * d - 2. * sqrt(m * k_n) * v_n
    for i in range(particleNum[None]):
        # calculate normal gap
        eta1 = p[i].x[1] - p_rad - l[0].x0[1] 
        eta2 = l[1].x0[1] - p[i].x[1] - p_rad
        if eta1 < 0.:
            p[i].cf = [0, -kappa * eta1 - 2. * vratio * ti.sqrt(p[i].m * kappa) * p[i].v[1]]
        elif eta2 < 0.:
            p[i].cf = [0, kappa * eta2 - 2. * vratio * ti.sqrt(p[i].m * kappa) * p[i].v[1]]
        else:
            p[i].cf = [0., 0.]

    # mapping particle momentum to grid
    for i in range(particleNum[None]):
        # loop over each influenced nodes for each particle
        for j in range(offset[i]):
            ix = LnID[i, j] % grid_x
            iy = LnID[i, j] // grid_x
            g[LnID[i, j]].p += shape[i, j] * p[i].m * (p[i].v + p[i].gradv @ (vec2f(ix, iy) * dx - p[i].x0))
        
    # mapping particle force (external and internal) to grid
    for i in range(particleNum[None]):
        # loop over each influenced nodes for each particle
        for j in range(offset[i]):
            stress = p[i].stress
            plane_stress = mat2x2([stress[0, 0], stress[0, 1]], [stress[1, 0], stress[1, 1]])
            g[LnID[i, j]].f += shape[i, j] * (p[i].m * gravity + p[i].cf) - p[i].vol * plane_stress @ dshape[i, j] 

    # solve momentum balance equation on grid
    for i in g:
        # if nodal mass is equal to zero, pass
        if g[i].m > val_lim:
            # apply background damping force
            for d in ti.static(range(2)):
                if g[i].p[d] * g[i].f[d] > 0.:
                    g[i].f[d] -= damp * abs(g[i].f[d]) * sign(g[i].p[d])
            # solve p = f * dt
            g[i].p += g[i].f * dt
            # apply boundary condition
            apply_bc(i)
    
    # mapping updated nodal variables to particles
    for i in range(particleNum[None]):
        acc = vec2f(0, 0)
        vel = vec2f(0, 0)
        # loop over each influenced nodes for each particle
        for j in range(offset[i]):
            acc += shape[i, j] * g[LnID[i, j]].f / g[LnID[i, j]].m
            vel += shape[i, j] * g[LnID[i, j]].p / g[LnID[i, j]].m
        p[i].v = (1. - alpha) * (p[i].v + acc * dt) + alpha * vel

    # musl remapping
    for i in g:
        if g[i].m > val_lim:
            g[i].p = vec2f(0, 0)

    for i in range(particleNum[None]):
        # loop over each influenced nodes for each particle
        for j in range(offset[i]):
            ix = LnID[i, j] % grid_x
            iy = LnID[i, j] // grid_x
            g[LnID[i, j]].p += shape[i, j] * p[i].m * (p[i].v + p[i].gradv @ (vec2f(ix, iy) * dx - p[i].x0))

    # calculate particle velocity gradient
    for i in range(particleNum[None]):
        # velocity gradient
        td_rate = mat2x2([[0, 0], [0, 0]])                                                        
        # loop over each influenced nodes for each particle                                                    
        for j in range(offset[i]):
            # calculate velocity gradient on each particle
            td_rate += (g[LnID[i, j]].p / g[LnID[i, j]].m).outer_product(dshape[i, j])            
        # pull gradv to global memory
        p[i].gradv = td_rate
        if ti.static(scheme == 1):
            p[i].vol *= (mat2x2([1, 0], [0, 1]) + p[i].gradv * dt).determinant()    
                                                                             
    if ti.static(fbar):
        # mapping jacobian from particle to grid
        for i in range(particleNum[None]):              
            # The determinant of deformation gradient at this timestep
            jac_rate = 0.
            if ti.static(scheme == 0):
                jac_rate = (mat2x2([1, 0], [0, 1]) + p[i].gradv @ p[i].td.inverse() * dt).determinant()    
            elif ti.static(scheme == 1):
                jac_rate = (mat2x2([1, 0], [0, 1]) + p[i].gradv * dt).determinant()    
            # loop over each influenced nodes for each particle                                                       
            for j in range(offset[i]):                                   
                # Mapping Jacobian from particles to grids
                g[LnID[i, j]].jac += shape[i, j] * p[i].m * jac_rate                    

        # update stress-strain
        for i in range(particleNum[None]):                                 
            # The determinant of deformation gradient at last timestep                                              
            jac_rate_bar = 0.
            for j in range(offset[i]):
                jac_rate_bar += shape[i, j] * g[LnID[i, j]].jac / g[LnID[i, j]].m
            
            # decide if use f bar
            td_rate = mat2x2([1, 0], [0, 1]) + p[i].gradv @ p[i].td.inverse() * dt
            jac_rate = td_rate.determinant()  
            p[i].td = ((jac_rate_bar / jac_rate) ** 0.5 * td_rate) @ p[i].td
    else:
        for i in range(particleNum[None]):
            # Calculate origin deformation gradient
            if ti.static(scheme == 0):
                p[i].td += p[i].gradv * dt  
            elif ti.static(scheme == 1):
                p[i].td = (mat2x2([1, 0], [0, 1]) + p[i].gradv * dt) @ p[i].td
            
    # Constitutive model (PK1 stress)
    for i in range(particleNum[None]):   
        if ti.static(materialID == 0):
            p[i].stress = neohookean(highdim(p[i].td))
        elif ti.static(materialID == 1):
            p[i].stress = mooney_rivlin(highdim(p[i].td))
        elif ti.static(materialID == 2):
            p[i].stress = generalized_mooney_rivlin(highdim(p[i].td))
        elif ti.static(materialID == 3):
            p[i].stress = gent(highdim(p[i].td))
        elif ti.static(materialID == 4):
            p[i].stress = gent_gent(highdim(p[i].td))
        elif ti.static(materialID == 5):
            p[i].stress = gent_hydrogel(highdim(p[i].td))
        elif ti.static(materialID == 6):
            p[i].stress = hydrogel(highdim(p[i].td))

    # mapping updated nodal variables to particles
    for i in range(particleNum[None]):
        vel = vec2f(0, 0)
        # loop over each influenced nodes for each particle
        for j in range(offset[i]):
            vel += shape[i, j] * g[LnID[i, j]].p / g[LnID[i, j]].m
        p[i].x += vel * dt

# =================================== Post-possessing =================================== #
# copy variables to visualize field
@ti.kernel
def copy():
    for i in range(particleNum[None]):
        visp[i] = ti.cast(vec3f([p[i].x[0], p[i].x[1], 0]) / base, ti.f32)
    for i in range(l.shape[0]):
        visl[2 * i] = ti.cast(vec3f([l[i].x0[0], l[i].x0[1], 0]) / base, ti.f32)
        visl[2 * i + 1] = ti.cast(vec3f([l[i].x1[0], l[i].x1[1], 0]) / base, ti.f32)

def output(step):
    if not os.path.exists(directory_path+'/impact'):
        os.makedirs(directory_path+'/impact')
    print("Current step:", step, '\n')
    pos = p.x.to_numpy()
    posx = np.ascontiguousarray(pos[0:particleNum[None],0])
    posy = np.ascontiguousarray(pos[0:particleNum[None],1])
    posz = np.zeros(posx.shape[0])
    stress = p.stress.to_numpy()[0:particleNum[None],:,:]
    pressure = np.ascontiguousarray(1./3.*(stress[:,0,0]+stress[:,1,1]+stress[:,2,2]))
    vonmises = np.ascontiguousarray(np.sqrt(0.5 * ((stress[:,0,0]-stress[:,1,1])**2+(stress[:,0,0]-stress[:,2,2])**2+(stress[:,2,2]-stress[:,1,1])**2+\
                                                   3.*stress[:,0,1]*stress[:,0,1]+3.*stress[:,1,0]*stress[:,1,0]+3.*stress[:,2,0]*stress[:,2,0]+3.*stress[:,0,2]*stress[:,0,2]+3.*stress[:,1,2]*stress[:,1,2]+3.*stress[:,2,1]*stress[:,2,1])))
    pointsToVTK(directory_path+f'/impact/hydrogel{step:06d}', posx, posy, posz, data={"pressure":pressure, "vonmises": vonmises})
    x0 = l.x0.to_numpy()
    x1 = l.x1.to_numpy()
    x = np.zeros(4)
    y = np.zeros(4)
    z = np.zeros(4)
    x[0] = x0[0, 0]
    x[1] = x1[0, 0]
    x[2] = x0[1, 0]
    x[3] = x1[1, 0]
    y[0] = x0[0, 1]
    y[1] = x1[0, 1]
    y[2] = x0[1, 1]
    y[3] = x1[1, 1]
    linesToVTK(directory_path+f'/impact/boundary{step:06d}', x, y, z)

if __name__ == "__main__":
    # initialize
    printNum = 0
    save_interval = int(0.02 * time // dt)
    line_init()
    particle_init()
    sdf_init()
    calculate_init()
    grid_init()

    # taichi gui
    if gui:
        window = ti.ui.Window('Window Title', (892, 892))
        visp = ti.Vector.field(3, ti.f32, particleNum[None])
        visl = ti.Vector.field(3, ti.f32, 2 * l.shape[0])
    
    # main loop
    step = 0
    while step<int(time // dt):
        if gui:
            copy()
            canvas = window.get_canvas()
            canvas.set_background_color((0, 0, 0))
            canvas.circles(visp, p_rad/base, (1, 1, 1))
            canvas.lines(visl, width=0.001)
            window.show()
        for _ in range(int(0.02 * time // dt)):
            if not gui and step % save_interval == 0:
                output(printNum)
                printNum += 1
            # main kernels
            solve()
            step+=1

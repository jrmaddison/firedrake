from firedrake import *
from firedrake.adjoint import *

import numpy as np


def taylor_test_reverse_over_forward(forward, m, zeta, eps_vals):
    e_0 = []
    e_1 = []
    e_2 = []
    for eps in eps_vals:
        dm = Function(zeta.function_space(), name="dm")
        dm.assign(eps * zeta)

        get_working_tape().clear_tape()
        m.block_variable.tlm_value = dm

        continue_annotation()
        with reverse_over_forward():
            J = forward(m)
        pause_annotation()

        ddJ, dJ = compute_gradient(J.block_variable.tlm_value,
                                   [Control(m), Control(dm)])

        with dm.dat.vec_ro as dm_v, dJ.dat.vec_ro as dJ_v:
            correction_1 = dm_v.dot(dJ_v)
        with dm.dat.vec_ro as dm_v, ddJ.dat.vec_ro as ddJ_v:
            correction_2 = 0.5 * dm_v.dot(ddJ_v)

        e = forward(m + dm) - J
        e_0.append(abs(e))
        e_1.append(abs(e - correction_1))
        e_2.append(abs(e - correction_1 - correction_2))
    e_0 = np.array(e_0)
    e_1 = np.array(e_1)
    e_2 = np.array(e_2)

    order_0 = np.log(e_0[1:] / e_0[:-1]) / np.log(eps_vals[1:] / eps_vals[:-1])
    order_1 = np.log(e_1[1:] / e_1[:-1]) / np.log(eps_vals[1:] / eps_vals[:-1])
    order_2 = np.log(e_2[1:] / e_2[:-1]) / np.log(eps_vals[1:] / eps_vals[:-1])

    print(f"{e_0=}")
    print(f"{order_0=}")
    print(f"{e_1=}")
    print(f"{order_1=}")
    print(f"{e_2=}")
    print(f"{order_2=}")
    return order_0, order_1, order_2


def test_solve_assemble():
    np.random.seed(876951)

    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    m = Function(space, name="m")
    m.interpolate(X[0] * sin(pi * X[0]) * sin(2 * pi * X[0]))
    zeta = Function(space, name="zeta")
    zeta.dat.data[:] = np.random.random(zeta.dat.data_ro.shape)

    eps_vals = 1.0e-3 * np.array([2 ** -p for p in range(5)], dtype=float)

    def forward(m):
        u = Function(space, name="u")
        solve(inner(trial, test) * dx == inner(m, test) * dx,
              u)

        J = assemble(((u + Constant(1.0)) ** 4) * dx)
        return J

    order_0, order_1, order_2 = taylor_test_reverse_over_forward(forward, m, zeta, eps_vals)

    assert order_0.min() >= 1.00
    assert order_1.min() >= 2.00
    assert order_2.min() >= 3.00

    get_working_tape().clear_tape()


def test_Function_assign():
    np.random.seed(97520)

    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    m = Function(space, name="m")
    m.interpolate(X[0] * sin(pi * X[0]) * sin(2 * pi * X[0]))
    zeta = Function(space, name="zeta")
    zeta.dat.data[:] = np.random.random(zeta.dat.data_ro.shape)

    eps_vals = 1.0e-3 * np.array([2 ** -p for p in range(5)], dtype=float)

    def forward(m):
        u = Function(space, name="u")
        v = Function(space, name="v")
        v.assign(m)
        u.assign(2 * v + m)

        J = assemble(((u + Constant(1.0)) ** 4) * dx)
        return J

    order_0, order_1, order_2 = taylor_test_reverse_over_forward(forward, m, zeta, eps_vals)

    assert order_0.min() >= 1.00
    assert order_1.min() >= 2.00
    assert order_2.min() >= 2.99

    get_working_tape().clear_tape()


def test_Function_interpolate():
    np.random.seed(357286)

    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space_1 = FunctionSpace(mesh, "Lagrange", 1)
    space_2 = FunctionSpace(mesh, "Lagrange", 2)

    m = Function(space_1, name="m")
    m.interpolate(X[0] * sin(pi * X[0]) * sin(2 * pi * X[0]))
    zeta = Function(space_1, name="zeta")
    zeta.dat.data[:] = np.random.random(zeta.dat.data_ro.shape)

    eps_vals = 1.0e-3 * np.array([2 ** -p for p in range(5)], dtype=float)

    def forward(m):
        u = Function(space_2, name="u")
        v = Function(space_2, name="v")
        v.interpolate(m)
        u.interpolate(2 * v + m)

        J = assemble(((u + Constant(1.0)) ** 4) * dx)
        return J

    order_0, order_1, order_2 = taylor_test_reverse_over_forward(forward, m, zeta, eps_vals)

    assert order_0.min() >= 1.00
    assert order_1.min() >= 2.00
    assert order_2.min() >= 2.99

    get_working_tape().clear_tape()

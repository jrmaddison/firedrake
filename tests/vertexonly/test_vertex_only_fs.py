from firedrake import *
import pytest
import numpy as np
from mpi4py import MPI


# Utility Functions

@pytest.fixture(params=["interval",
                        "square",
                        "squarequads",
                        "extruded",
                        pytest.param("extrudedvariablelayers", marks=pytest.mark.skip(reason="Extruded meshes with variable layers not supported and will hang when created in parallel")),
                        "cube",
                        "tetrahedron",
                        pytest.param("immersedsphere", marks=pytest.mark.xfail(reason="immersed parent meshes not supported")),
                        "periodicrectangle",
                        "shiftedmesh"])
def parentmesh(request):
    if request.param == "interval":
        return UnitIntervalMesh(1)
    elif request.param == "square":
        return UnitSquareMesh(1, 1)
    elif request.param == "squarequads":
        return UnitSquareMesh(2, 2, quadrilateral=True)
    elif request.param == "extruded":
        return ExtrudedMesh(UnitSquareMesh(2, 2), 3)
    elif request.param == "extrudedvariablelayers":
        return ExtrudedMesh(UnitIntervalMesh(3), np.array([[0, 3], [0, 3], [0, 2]]), np.array([3, 3, 2]))
    elif request.param == "cube":
        return UnitCubeMesh(1, 1, 1)
    elif request.param == "tetrahedron":
        return UnitTetrahedronMesh()
    elif request.param == "immersedsphere":
        return UnitIcosahedralSphereMesh()
    elif request.param == "periodicrectangle":
        return PeriodicRectangleMesh(3, 3, 1, 1)
    elif request.param == "shiftedmesh":
        m = UnitSquareMesh(10, 10)
        m.coordinates.dat.data[:] -= 0.5
        return m


@pytest.fixture(params=[0, 1, 100], ids=lambda x: f"{x}-coords")
def vertexcoords(request, parentmesh):
    size = (request.param, parentmesh.geometric_dimension())
    return pseudo_random_coords(size)


def pseudo_random_coords(size):
    """
    Get an array of pseudo random coordinates with coordinate elements
    between -0.5 and 1.5. The random numbers are consistent for any
    given `size` since `numpy.random.seed(0)` is called each time this
    is used.
    """
    np.random.seed(0)
    a, b = -0.5, 1.5
    return (b - a) * np.random.random_sample(size=size) + a


# Function Space Generation Tests

def functionspace_tests(vm):
    # Prep
    num_cells = len(vm.coordinates.dat.data_ro)
    num_cells_mpi_global = MPI.COMM_WORLD.allreduce(num_cells, op=MPI.SUM)
    num_cells_halo = len(vm.coordinates.dat.data_ro_with_halos) - num_cells
    # Can create DG0 function space
    V = FunctionSpace(vm, "DG", 0)
    # Can't create with degree > 0
    with pytest.raises(ValueError):
        V = FunctionSpace(vm, "DG", 1)
    # Can create function on function spaces
    f = Function(V)
    g = Function(V)
    # Make expr which is x in 1D, x*y in 2D, x*y*z in 3D
    from functools import reduce
    from operator import mul
    expr = reduce(mul, SpatialCoordinate(vm))
    # Can interpolate and Galerkin project expressions onto functions
    f.interpolate(expr)
    g.project(expr)
    # Should have 1 DOF per cell so check DOF DataSet
    assert f.dof_dset.size == g.dof_dset.size == vm.cell_set.size == num_cells
    assert f.dof_dset.total_size == g.dof_dset.total_size == vm.cell_set.total_size == num_cells + num_cells_halo
    # The function should take on the value of the expression applied to
    # the vertex only mesh coordinates (with no change to coordinate ordering)
    # Reshaping because for all meshes, we want (-1, gdim) but
    # when gdim == 1 PyOP2 doesn't distinguish between dats with shape
    # () and shape (1,).
    assert np.allclose(f.dat.data_ro, np.prod(vm.coordinates.dat.data_ro.reshape(-1, vm.geometric_dimension()), axis=1))
    # Galerkin Projection of expression is the same as interpolation of
    # that expression since both exactly point evaluate the expression.
    assert np.allclose(f.dat.data_ro, g.dat.data_ro)
    # Assembly works as expected - global assembly (integration) of a
    # constant on a vertex only mesh is evaluation of that constant
    # num_vertices (globally) times
    f.interpolate(Constant(2, domain=vm))
    assert np.isclose(assemble(f*dx), 2*num_cells_mpi_global)
    if "input_ordering" in vm.name:
        with pytest.raises(AttributeError):
            W = FunctionSpace(vm.input_ordering, "DG", 0)
        return
    # Can interpolate onto the input ordering VOM and we retain values from the
    # expresson on the main VOM
    W = FunctionSpace(vm.input_ordering, "DG", 0)
    h = Function(W)
    h.dat.data_wo_with_halos[:] = -1
    h.interpolate(g)
    # Exclude points which we know are missing - these should all be equal to -1
    input_ordering_parent_cell_nums = vm.input_ordering.topology_dm.getField("parentcellnum")
    vm.input_ordering.topology_dm.restoreField("parentcellnum")
    idxs_to_include = input_ordering_parent_cell_nums != -1
    assert np.allclose(h.dat.data_ro_with_halos[idxs_to_include], np.prod(vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include].reshape(-1, vm.input_ordering.geometric_dimension()), axis=1))
    assert np.all(h.dat.data_ro_with_halos[~idxs_to_include] == -1)
    # check we can interpolate expressions
    h2 = Function(W)
    h2.interpolate(2*g*Constant(1, domain=vm))
    assert np.allclose(h2.dat.data_ro_with_halos[idxs_to_include], 2*np.prod(vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include].reshape(-1, vm.input_ordering.geometric_dimension()), axis=1))
    # Check that the opposite works
    g.dat.data_wo_with_halos[:] = -1
    g.interpolate(h)
    assert np.allclose(g.dat.data_ro_with_halos, np.prod(vm.coordinates.dat.data_ro_with_halos.reshape(-1, vm.geometric_dimension()), axis=1))
    # Can equivalently create interpolators and use them. NOTE the
    # transpose interpolator is equivilent to the inverse here because the
    # inner product matrix in the reisz representer is the identity. TODO: when
    # we introduce cofunctions, this will need to be rewritten.
    I_io = Interpolator(TestFunction(V), W)
    h = I_io.interpolate(g)
    assert np.allclose(h.dat.data_ro_with_halos[idxs_to_include], np.prod(vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include].reshape(-1, vm.input_ordering.geometric_dimension()), axis=1))
    assert np.all(h.dat.data_ro_with_halos[~idxs_to_include] == 0)
    I2_io = Interpolator(2*TestFunction(V)*Constant(1, domain=vm), W)
    h2 = I2_io.interpolate(g)
    assert np.allclose(h2.dat.data_ro_with_halos[idxs_to_include], 2*np.prod(vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include].reshape(-1, vm.input_ordering.geometric_dimension()), axis=1))
    g = I_io.interpolate(h, transpose=True)
    assert np.allclose(g.dat.data_ro_with_halos, np.prod(vm.coordinates.dat.data_ro_with_halos.reshape(-1, vm.geometric_dimension()), axis=1))
    with pytest.raises(NotImplementedError):
        # Can't use transpose on interpolators with expressions yet
        g2 = I2_io.interpolate(h, transpose=True)
        assert np.allclose(g2.dat.data_ro_with_halos, 2*np.prod(vm.coordinates.dat.data_ro_with_halos.reshape(-1, vm.geometric_dimension()), axis=1))

    I_io_transpose = Interpolator(TestFunction(W), V)
    I2_io_transpose = Interpolator(2*TestFunction(W)*Constant(1, domain=vm.input_ordering), V)
    h = I_io_transpose.interpolate(g, transpose=True)
    assert np.allclose(h.dat.data_ro_with_halos[idxs_to_include], np.prod(vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include].reshape(-1, vm.input_ordering.geometric_dimension()), axis=1))
    assert np.all(h.dat.data_ro_with_halos[~idxs_to_include] == 0)
    with pytest.raises(NotImplementedError):
        # Can't use transpose on interpolators with expressions yet
        h2 = I2_io_transpose.interpolate(g, transpose=True)
        assert np.allclose(h2.dat.data_ro_with_halos[idxs_to_include], 2*np.prod(vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include].reshape(-1, vm.input_ordering.geometric_dimension()), axis=1))
    g = I_io_transpose.interpolate(h)
    assert np.allclose(g.dat.data_ro_with_halos, np.prod(vm.coordinates.dat.data_ro_with_halos.reshape(-1, vm.geometric_dimension()), axis=1))
    g2 = I2_io_transpose.interpolate(h)
    assert np.allclose(g2.dat.data_ro_with_halos, 2*np.prod(vm.coordinates.dat.data_ro_with_halos.reshape(-1, vm.geometric_dimension()), axis=1))


def vectorfunctionspace_tests(vm):
    # Prep
    gdim = vm.geometric_dimension()
    num_cells = len(vm.coordinates.dat.data_ro)
    num_cells_mpi_global = MPI.COMM_WORLD.allreduce(num_cells, op=MPI.SUM)
    num_cells_halo = len(vm.coordinates.dat.data_ro_with_halos) - num_cells
    # Can create DG0 function space
    V = VectorFunctionSpace(vm, "DG", 0)
    # Can't create with degree > 0
    with pytest.raises(ValueError):
        V = VectorFunctionSpace(vm, "DG", 1)
    # Can create functions on function spaces
    f = Function(V)
    g = Function(V)
    # Can interpolate and Galerkin project onto functions
    x = SpatialCoordinate(vm)
    f.interpolate(2*x)
    g.project(2*x)
    # Should have 1 DOF per cell so check DOF DataSet
    assert f.dof_dset.size == g.dof_dset.size == vm.cell_set.size == num_cells
    assert f.dof_dset.total_size == g.dof_dset.total_size == vm.cell_set.total_size == num_cells + num_cells_halo
    # The function should take on the value of the expression applied to
    # the vertex only mesh coordinates (with no change to coordinate ordering)
    assert np.allclose(f.dat.data_ro, 2*vm.coordinates.dat.data_ro)
    # Galerkin Projection of expression is the same as interpolation of
    # that expression since both exactly point evaluate the expression.
    assert np.allclose(f.dat.data_ro, g.dat.data_ro)
    # Assembly works as expected - global assembly (integration) of a
    # constant on a vertex only mesh is evaluation of that constant
    # num_vertices (globally) times. Note that we get a vertex cell for
    # each geometric dimension so we have to sum over geometric
    # dimension too.
    f.interpolate(Constant([1] * gdim, domain=vm))
    assert np.isclose(assemble(inner(f, f)*dx), num_cells_mpi_global*gdim)
    if "input_ordering" in vm.name:
        with pytest.raises(AttributeError):
            W = FunctionSpace(vm.input_ordering, "DG", 0)
        return
    # Can interpolate onto the input ordering VOM and we retain values from the
    # expresson on the main VOM
    W = VectorFunctionSpace(vm.input_ordering, "DG", 0)
    h = Function(W)
    h.dat.data_wo_with_halos[:] = -1
    h.interpolate(g)
    # Exclude points which we know are missing - these should all be equal to -1
    input_ordering_parent_cell_nums = vm.input_ordering.topology_dm.getField("parentcellnum")
    vm.input_ordering.topology_dm.restoreField("parentcellnum")
    idxs_to_include = input_ordering_parent_cell_nums != -1
    assert np.allclose(h.dat.data_ro[idxs_to_include], 2*vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include])
    assert np.all(h.dat.data_ro_with_halos[~idxs_to_include] == -1)
    # check we can interpolate expressions
    h2 = Function(W)
    h2.interpolate(2*g*Constant(1, domain=vm))
    assert np.allclose(h2.dat.data_ro[idxs_to_include], 4*vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include])
    # Check that the opposite works
    g.dat.data_wo_with_halos[:] = -1
    g.interpolate(h)
    assert np.allclose(g.dat.data_ro_with_halos, 2*vm.coordinates.dat.data_ro_with_halos)
    # Can equivalently create interpolators and use them. NOTE the
    # transpose interpolator is equivilent to the inverse here because the
    # inner product matrix in the reisz representer is the identity. TODO: when
    # we introduce cofunctions, this will need to be rewritten.
    I_io = Interpolator(TestFunction(V), W)
    h = I_io.interpolate(g)
    assert np.allclose(h.dat.data_ro[idxs_to_include], 2*vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include])
    assert np.all(h.dat.data_ro_with_halos[~idxs_to_include] == 0)
    I2_io = Interpolator(2*TestFunction(V)*Constant(1, domain=vm), W)
    h2 = I2_io.interpolate(g)
    assert np.allclose(h2.dat.data_ro[idxs_to_include], 4*vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include])
    g = I_io.interpolate(h, transpose=True)
    assert np.allclose(g.dat.data_ro_with_halos, 2*vm.coordinates.dat.data_ro_with_halos)
    with pytest.raises(NotImplementedError):
        # Can't use transpose on interpolators with expressions yet
        g2 = I2_io.interpolate(h, transpose=True)
        assert np.allclose(g2.dat.data_ro_with_halos, 4*vm.coordinates.dat.data_ro_with_halos)

    I_io_transpose = Interpolator(TestFunction(W), V)
    I2_io_transpose = Interpolator(2*TestFunction(W)*Constant(1, domain=vm.input_ordering), V)
    h = I_io_transpose.interpolate(g, transpose=True)
    assert np.allclose(h.dat.data_ro[idxs_to_include], 2*vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include])
    assert np.all(h.dat.data_ro_with_halos[~idxs_to_include] == 0)
    with pytest.raises(NotImplementedError):
        # Can't use transpose on interpolators with expressions yet
        h2 = I2_io_transpose.interpolate(g, transpose=True)
        assert np.allclose(h2.dat.data_ro[idxs_to_include], 4*vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include])
    g = I_io_transpose.interpolate(h)
    assert np.allclose(g.dat.data_ro_with_halos, 2*vm.coordinates.dat.data_ro_with_halos)
    g2 = I2_io_transpose.interpolate(h)
    assert np.allclose(g2.dat.data_ro_with_halos, 4*vm.coordinates.dat.data_ro_with_halos)


def test_functionspaces(parentmesh, vertexcoords):
    vm = VertexOnlyMesh(parentmesh, vertexcoords, missing_points_behaviour=None)
    functionspace_tests(vm)
    vectorfunctionspace_tests(vm)
    functionspace_tests(vm.input_ordering)
    vectorfunctionspace_tests(vm.input_ordering)


@pytest.mark.parallel
def test_functionspaces_parallel(parentmesh, vertexcoords):
    test_functionspaces(parentmesh, vertexcoords)


@pytest.mark.parallel(nprocs=2)
def test_simple_line():
    m = UnitIntervalMesh(4)
    points = np.asarray([[0.125], [0.375], [0.625]])
    vm = VertexOnlyMesh(m, points, redundant=True)
    V = FunctionSpace(vm, "DG", 0)
    f = Function(V)
    g = Function(V)
    x = SpatialCoordinate(vm)
    expr = x**2
    # Can interpolate and Galerkin project expressions onto functions
    f.interpolate(expr)
    g.project(expr)

    assert np.allclose(f.dat.data_ro, vm.coordinates.dat.data_ro**2)
    # Galerkin Projection of expression is the same as interpolation of
    # that expression since both exactly point evaluate the expression.
    assert np.allclose(f.dat.data_ro, g.dat.data_ro)


@pytest.mark.parallel(nprocs=2)
def test_input_ordering_missing_point():
    m = UnitIntervalMesh(4)
    points = np.asarray([[0.125], [0.375], [0.625], [5.0]])
    data = np.asarray([1.0, 2.0, 3.0, 4.0])
    vm = VertexOnlyMesh(m, points, missing_points_behaviour=None, redundant=True)

    # put data on the input ordering
    P0DG_input_ordering = FunctionSpace(vm.input_ordering, "DG", 0)
    data_input_ordering = Function(P0DG_input_ordering)
    if vm.comm.rank == 0:
        data_input_ordering.dat.data_wo[:] = data
    else:
        data_input_ordering.dat.data_wo[:] = []
        assert not len(data_input_ordering.dat.data_ro)

    # shouldn't have any halos
    assert np.array_equal(data_input_ordering.dat.data_ro_with_halos, data_input_ordering.dat.data_ro)

    # Interpolate it onto the immersed vertex-only mesh
    P0DG = FunctionSpace(vm, "DG", 0)
    data_on_vm = Function(P0DG).interpolate(data_input_ordering)

    # Check that the data is correct
    for data_at_point, point in zip(data_on_vm.dat.data_ro_with_halos, vm.coordinates.dat.data_ro_with_halos):
        assert data_at_point == data[points.flatten() == point]

    # change the data on the immersed vertex-only mesh
    data_on_vm.assign(2*data_on_vm)

    # interpolate it back onto the input ordering and make sure we get what we
    # expect and that the point which was missing still has it's original value
    data_input_ordering.interpolate(data_on_vm)
    if vm.comm.rank == 0:
        assert np.allclose(data_input_ordering.dat.data_ro[0:3], 2*data[0:3])
        assert np.allclose(data_input_ordering.dat.data_ro[3], data[3])
    else:
        assert not len(data_input_ordering.dat.data_ro)

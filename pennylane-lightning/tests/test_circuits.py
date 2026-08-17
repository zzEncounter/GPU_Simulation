import numpy as np
import pennylane as qml
import pytest
from pennylane_lightning_baseline import build_hamiltonian, get_circuit


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ra-hea", 8),
        ("su2_hea", 16),
        ("rzz", 24),
        ("qaoa", 4),
        ("qaoa-ns", 16),
        ("qaoa_ns", 16),
        ("xxz", 24),
        ("mera", 8),
        ("eqnn", 6),
    ],
)
def test_parameter_counts(name, expected):
    assert get_circuit(name).parameter_count(4, 2) == expected


@pytest.mark.parametrize(
    ("name", "op_names"),
    [
        ("ra-hea", ["RY"] * 4 + ["CNOT"] * 4),
        ("su2-hea", ["RY"] * 4 + ["RZ"] * 4 + ["CNOT"] * 4),
        ("rzz-hea", ["RX"] * 4 + ["RZ"] * 4 + ["IsingZZ"] * 4),
    ],
)
def test_one_layer_gate_order_and_unique_parameters(name, op_names):
    spec = get_circuit(name)
    params = np.arange(spec.parameter_count(4, 1), dtype=np.float64)
    with qml.tape.QuantumTape() as tape:
        spec.apply(params, 4, 1)

    assert [op.name for op in tape.operations] == op_names
    parameterised = [op.data[0] for op in tape.operations if op.num_params == 1]
    np.testing.assert_array_equal(parameterised, params)


def test_rzz_checkerboard_edges():
    spec = get_circuit("rzz-hea")
    params = np.zeros(spec.parameter_count(6, 1))
    with qml.tape.QuantumTape() as tape:
        spec.apply(params, 6, 1)
    edges = [tuple(op.wires) for op in tape.operations if op.name == "IsingZZ"]
    assert edges == [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (5, 0)]


def test_qaoa_shared_parameters_and_cost_then_mixer_order():
    spec = get_circuit("qaoa")
    params = np.array([0.25, -0.75])
    with qml.tape.QuantumTape() as tape:
        spec.apply(params, 4, 1)
    assert [op.name for op in tape.operations] == (
        ["Hadamard"] * 4 + ["IsingZZ"] * 4 + ["RX"] * 4
    )
    angles = [op.data[0] for op in tape.operations if op.num_params == 1]
    np.testing.assert_allclose(angles, [params[1]] * 4 + [params[0]] * 4)


def test_qaoa_ns_independent_parameters_and_cost_then_mixer_order():
    spec = get_circuit("qaoa_ns")
    params = np.arange(spec.parameter_count(4, 1), dtype=np.float64)
    with qml.tape.QuantumTape() as tape:
        spec.apply(params, 4, 1)
    assert [op.name for op in tape.operations] == (
        ["Hadamard"] * 4 + ["IsingZZ"] * 4 + ["RX"] * 4
    )
    angles = [op.data[0] for op in tape.operations if op.num_params == 1]
    np.testing.assert_array_equal(angles, [*params[4:], *params[:4]])
    assert [tuple(op.wires) for op in tape.operations[4:8]] == [
        (0, 1), (1, 2), (2, 3), (3, 0)
    ]


def test_xxz_hva_checkerboard_pairing_and_axis_major_parameters():
    spec = get_circuit("xxz-hva")
    params = np.arange(spec.parameter_count(4, 1), dtype=np.float64)
    with qml.tape.QuantumTape() as tape:
        spec.apply(params, 4, 1)
    assert [op.name for op in tape.operations[:2]] == ["PauliX", "PauliX"]
    bonds = tape.operations[2:]
    assert [op.name for op in bonds] == ["IsingXX", "IsingYY", "IsingZZ"] * 4
    assert [tuple(op.wires) for op in bonds[::3]] == [
        (0, 1),
        (2, 3),
        (1, 2),
        (3, 0),
    ]
    np.testing.assert_array_equal(
        [op.data[0] for op in bonds],
        [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11],
    )


@pytest.mark.parametrize(
    ("qubits", "layers", "expected_parameters", "expected_pairs"),
    [
        (
            6,
            3,
            16,
            [(1, 2), (3, 4), (0, 1), (2, 3), (4, 5),
             (3, 5), (1, 3), (3, 5)],
        ),
        (
            8,
            3,
            22,
            [(1, 2), (3, 4), (5, 6), (0, 1), (2, 3), (4, 5), (6, 7),
             (3, 5), (1, 3), (5, 7), (3, 7)],
        ),
    ],
)
def test_mera_topology_and_parameter_order(
    qubits, layers, expected_parameters, expected_pairs
):
    spec = get_circuit("mera")
    params = np.arange(expected_parameters, dtype=np.float64)
    with qml.tape.QuantumTape() as tape:
        spec.apply(params, qubits, layers)

    assert [op.name for op in tape.operations] == ["RY", "RY", "CNOT"] * len(
        expected_pairs
    )
    assert [
        tuple(tape.operations[index + 2].wires)
        for index in range(0, len(tape.operations), 3)
    ] == expected_pairs
    np.testing.assert_array_equal(
        [op.data[0] for op in tape.operations if op.name == "RY"], params
    )


def test_mera_rejects_noncanonical_layer_count():
    with pytest.raises(ValueError, match="invalid parameter count"):
        get_circuit("mera").parameter_count(6, 2)


def test_rzz_rejects_odd_ring():
    with pytest.raises(ValueError, match="even number of qubits"):
        get_circuit("rzz-hea").parameter_count(5, 2)


def test_fixed_hamiltonian_has_periodic_terms():
    hamiltonian = build_hamiltonian(4)
    assert len(hamiltonian.terms()[0]) == 8
    assert list(hamiltonian.terms()[0]) == [-1.0] * 8
    assert tuple(hamiltonian.terms()[1][3].wires) == (3, 0)


def test_qaoa_and_xxz_use_circuit_specific_hamiltonians():
    qaoa = build_hamiltonian(4, "qaoa")
    assert list(qaoa.terms()[0]) == [0.5, -0.5] * 4
    qaoa_ns = build_hamiltonian(4, "qaoa-ns")
    assert list(qaoa_ns.terms()[0]) == list(qaoa.terms()[0])
    xxz = build_hamiltonian(4, "xxz-hva")
    assert list(xxz.terms()[0]) == [1.0, 1.0, 0.5] * 4
    mera = build_hamiltonian(6, "mera")
    assert list(mera.terms()[0]) == [1.0]
    assert mera.terms()[1][0].name == "PauliZ"
    assert tuple(mera.terms()[1][0].wires) == (5,)


@pytest.mark.parametrize("qubits", [2, 3, 4, 5])
def test_eqnn_round_robin_shared_parameters(qubits):
    spec = get_circuit("equivariant-qnn")
    params = np.array([0.1, 0.2, 0.3])
    with qml.tape.QuantumTape() as tape:
        spec.apply(params, qubits, 1)
    assert [op.name for op in tape.operations[: 2 * qubits]] == [
        "RX"
    ] * qubits + ["RY"] * qubits
    pair_ops = tape.operations[2 * qubits :]
    assert len(pair_ops) == 3 * qubits * (qubits - 1) // 2
    assert [op.name for op in pair_ops[::3]] == ["CNOT"] * (qubits * (qubits - 1) // 2)
    assert [op.name for op in pair_ops[1::3]] == ["RZ"] * (qubits * (qubits - 1) // 2)
    assert [op.name for op in pair_ops[2::3]] == ["CNOT"] * (qubits * (qubits - 1) // 2)
    assert {tuple(op.wires) for op in pair_ops[::3]} == {
        (left, right) for left in range(qubits) for right in range(left + 1, qubits)
    }
    assert all(op.data[0] == 0.3 for op in pair_ops[1::3])


def test_eqnn_hamiltonian_is_mean_x():
    hamiltonian = build_hamiltonian(4, "eqnn")
    assert list(hamiltonian.terms()[0]) == [0.25] * 4
    assert [term.name for term in hamiltonian.terms()[1]] == ["PauliX"] * 4

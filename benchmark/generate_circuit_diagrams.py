#!/usr/bin/env python3
"""Generate representative SVG diagrams for the three added circuits."""

from pathlib import Path


OUT = Path(__file__).parent / "results" / "figures"
WIDTH = 1200
INK = "#24323d"
MUTED = "#6b747c"
WIRE = "#8a949c"
BLUE = "#d8eef4"
BLUE_STROKE = "#176b87"
RED = "#f8e1dc"
RED_STROKE = "#b84b38"
GREEN = "#e5efd9"
GREEN_STROKE = "#5b7f36"


def header(height, title, subtitle):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#24323d;letter-spacing:0}.title{font-size:24px;font-weight:700}.sub{font-size:14px;fill:#6b747c}.q{font-size:15px;font-weight:600}.gate{font-size:14px;font-weight:700}.small{font-size:12px;fill:#59636b}.phase{font-size:13px;font-weight:700}</style>',
        f'<text class="title" x="70" y="36">{title}</text>',
        f'<text class="sub" x="70" y="59">{subtitle}</text>',
    ]


def wires(svg, count, y0, gap, x0=92, x1=1140):
    for q in range(count):
        y = y0 + q * gap
        svg.append(f'<text class="q" x="70" y="{y + 5}" text-anchor="end">q{q}</text>')
        svg.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{WIRE}" stroke-width="1.5"/>')


def gate(svg, x, y, label, fill, stroke, width=54):
    svg.append(f'<rect x="{x-width/2}" y="{y-19}" width="{width}" height="38" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>')
    svg.append(f'<text class="gate" x="{x}" y="{y+5}" text-anchor="middle">{label}</text>')


def cnot(svg, x, y_control, y_target, stroke=BLUE_STROKE):
    svg.append(f'<line x1="{x}" y1="{y_control}" x2="{x}" y2="{y_target}" stroke="{stroke}" stroke-width="2"/>')
    svg.append(f'<circle cx="{x}" cy="{y_control}" r="5" fill="{stroke}"/>')
    svg.append(f'<circle cx="{x}" cy="{y_target}" r="11" fill="#fff" stroke="{stroke}" stroke-width="2"/>')
    svg.append(f'<line x1="{x-7}" y1="{y_target}" x2="{x+7}" y2="{y_target}" stroke="{stroke}" stroke-width="2"/>')
    svg.append(f'<line x1="{x}" y1="{y_target-7}" x2="{x}" y2="{y_target+7}" stroke="{stroke}" stroke-width="2"/>')


def cz(svg, x, y1, y2, stroke=GREEN_STROKE):
    svg.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{stroke}" stroke-width="2"/>')
    svg.append(f'<circle cx="{x}" cy="{y1}" r="6" fill="{stroke}"/>')
    svg.append(f'<circle cx="{x}" cy="{y2}" r="6" fill="{stroke}"/>')


def rzz(svg, x, y1, y2):
    svg.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{RED_STROKE}" stroke-width="2"/>')
    for y in (y1, y2):
        svg.append(f'<rect x="{x-18}" y="{y-13}" width="36" height="26" rx="4" fill="{RED}" stroke="{RED_STROKE}" stroke-width="1.7"/>')
        svg.append(f'<text class="gate" x="{x}" y="{y+5}" text-anchor="middle">ZZ</text>')
    svg.append(f'<text class="small" x="{x+24}" y="{(y1+y2)/2+4}">g</text>')


def band(svg, x1, x2, y1, y2, label, fill, stroke):
    svg.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" rx="5" fill="{fill}" fill-opacity="0.22" stroke="{stroke}" stroke-dasharray="5 4"/>')
    svg.append(f'<text class="phase" x="{(x1+x2)/2}" y="{y1-8}" text-anchor="middle" fill="{stroke}">{label}</text>')


def write(name, svg):
    svg.append('</svg>')
    path = OUT / name
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(path)


def mera():
    height, y0, gap = 570, 105, 48
    svg = header(height, "MERA circuit (8-qubit example)", "Each pair applies RY on both wires followed by CNOT; U stages select the next active wires")
    wires(svg, 8, y0, gap)
    stages = [
        ("D0", [(1, 2), (3, 4), (5, 6)]),
        ("U0", [(0, 1), (2, 3), (4, 5), (6, 7)]),
        ("D1", [(3, 5)]),
        ("U1", [(1, 3), (5, 7)]),
        ("D2", []),
        ("U2", [(3, 7)]),
    ]
    x = 150
    for label, pairs in stages:
        x1, x2 = x - 38, x + 75
        color = BLUE_STROKE if label.startswith("D") else GREEN_STROKE
        fill = BLUE if label.startswith("D") else GREEN
        band(svg, x1, x2, 82, 468, label, fill, color)
        if not pairs:
            svg.append(f'<text class="small" x="{x+18}" y="285" text-anchor="middle">no pair</text>')
        for left, right in pairs:
            gate(svg, x, y0 + left * gap, "RY", fill, color, 42)
            gate(svg, x, y0 + right * gap, "RY", fill, color, 42)
            cnot(svg, x + 48, y0 + left * gap, y0 + right * gap, color)
        x += 165
    svg.append(f'<text class="small" x="70" y="520">Active wires: [0..7] -> [1,3,5,7] -> [3,7] -> [7]</text>')
    svg.append(f'<text class="small" x="1135" y="520" text-anchor="end">Observable: Z on q7</text>')
    write("mera_circuit.svg", svg)


def eqnn():
    height, y0, gap = 390, 112, 58
    svg = header(height, "Equivariant QNN macro-layer (4-qubit example)", "Shared RX(a), shared RY(b), then all-to-all RZZ(g) in three round-robin matching phases")
    wires(svg, 4, y0, gap)
    for q in range(4):
        gate(svg, 150, y0 + q * gap, "RX(a)", BLUE, BLUE_STROKE, 62)
        gate(svg, 235, y0 + q * gap, "RY(b)", GREEN, GREEN_STROKE, 62)
    phases = [
        ("phase 0", [(0, 3), (1, 2)]),
        ("phase 1", [(0, 2), (1, 3)]),
        ("phase 2", [(0, 1), (2, 3)]),
    ]
    for p, (label, pairs) in enumerate(phases):
        start = 350 + p * 245
        band(svg, start - 42, start + 152, 82, 310, label, RED, RED_STROKE)
        for offset, (left, right) in enumerate(pairs):
            rzz(svg, start + offset * 105, y0 + left * gap, y0 + right * gap)
    svg.append(f'<text class="small" x="70" y="350">Each RZZ(g) is logically CX - RZ(g) - CX; SAD applies the all-to-all product with the closed-form phase.</text>')
    svg.append(f'<text class="small" x="1135" y="350" text-anchor="end">Observable: (1/n) sum X_i</text>')
    write("equivariant-qnn_circuit.svg", svg)


def data_reuploading():
    height, y0, gap = 555, 108, 48
    svg = header(height, "Data Re-uploading circuit (6-qubit example)", "Two consecutive layers show the alternating even/odd brickwork CZ topology")
    wires(svg, 6, y0, gap)
    layers = [
        ("layer l (even)", 0, [(0, 1), (2, 3), (4, 5)]),
        ("layer l+1 (odd)", 560, [(1, 2), (3, 4), (5, 0)]),
    ]
    for label, offset, pairs in layers:
        x0 = 130 + offset
        band(svg, x0 - 25, x0 + 480, 78, 382, label, GREEN, GREEN_STROKE)
        for q in range(6):
            gate(svg, x0 + 35, y0 + q * gap, "RZ(z)", BLUE, BLUE_STROKE, 58)
            gate(svg, x0 + 125, y0 + q * gap, "RY(y)", GREEN, GREEN_STROKE, 58)
            gate(svg, x0 + 215, y0 + q * gap, "RZ(x)", BLUE, BLUE_STROKE, 58)
        for index, (left, right) in enumerate(pairs):
            cz(svg, x0 + 305 + index * 62, y0 + left * gap, y0 + right * gap)
        svg.append(f'<text class="small" x="{x0+245}" y="410" text-anchor="middle">RZ -> RY -> RZ -> CZ matching</text>')
    svg.append(f'<text class="small" x="70" y="480">Even pairs: (0,1), (2,3), (4,5)   |   Odd pairs: (1,2), (3,4), (5,0)</text>')
    svg.append(f'<text class="small" x="1135" y="480" text-anchor="end">Observable: Z on q0</text>')
    write("data-reuploading_circuit.svg", svg)


def qaoa_ns():
    height, y0, gap = 430, 108, 52
    svg = header(
        height,
        "Non-shared-angle QAOA (4-qubit example)",
        "Each layer applies independent RX(beta[l,q]) mixers and ring RZZ(gamma[l,q]) cost terms",
    )
    wires(svg, 4, y0, gap)
    layers = [
        ("layer l", 0, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ("layer l+1", 500, [(0, 1), (1, 2), (2, 3), (3, 0)]),
    ]
    for label, offset, pairs in layers:
        x0 = 135 + offset
        band(svg, x0 - 28, x0 + 415, 78, 342, label, RED, RED_STROKE)
        for q in range(4):
            gate(svg, x0 + 42, y0 + q * gap, "RZZ", RED, RED_STROKE, 58)
            gate(svg, x0 + 150, y0 + q * gap, "RX", BLUE, BLUE_STROKE, 50)
        for index, (left, right) in enumerate(pairs):
            x = x0 + 260 + index * 35
            svg.append(
                f'<line x1="{x}" y1="{y0 + left * gap}" x2="{x}" '
                f'y2="{y0 + right * gap}" stroke="{RED_STROKE}" stroke-width="2"/>'
            )
        svg.append(
            f'<text class="small" x="{x0 + 205}" y="368" text-anchor="middle">'
            "ring edges: (0,1),(1,2),(2,3),(3,0)</text>"
        )
    svg.append(
        '<text class="small" x="70" y="400">Each layer has 2n independent parameters: '
        'beta[l,0..n-1] and gamma[l,0..n-1].</text>'
    )
    svg.append(
        '<text class="small" x="1135" y="400" text-anchor="end">Observable: ring cost Hamiltonian</text>'
    )
    write("qaoa-ns_circuit.svg", svg)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mera()
    eqnn()
    data_reuploading()
    qaoa_ns()


if __name__ == "__main__":
    main()

"""
Model 3D mobil sport dari titik-titik (point cloud), dibuat dengan rumus (tanpa file 3D).

Setiap bagian punya: nama, titik-titik (N x 3), warna (BGR), dan arah "bongkar"
(ke mana bagian itu bergeser saat mobil dibongkar).
Satuan: meter kira-kira. x = depan(+)/belakang(-), y = atas, z = kanan(+)/kiri(-).
"""
import math

import numpy as np

RNG = np.random.default_rng(11)

# ------------------------------------------------------------------ bentuk bodi
L_FRONT, L_REAR = 2.2, -2.2
WHEEL_X = (1.38, -1.38)
WHEEL_Z = 0.80
WHEEL_R = 0.35
WHEEL_Y = WHEEL_R


def _interp(x, xs, ys):
    return np.interp(x, xs, ys)


def body_top(x):
    """Tinggi atap/kap pada posisi x (siluet mobil sport)."""
    xs = [-2.2, -1.95, -1.5, -0.95, -0.55, -0.1, 0.35, 0.75, 1.3, 1.9, 2.2]
    ys = [0.66, 0.80, 0.86, 0.95, 1.12, 1.17, 1.08, 0.80, 0.70, 0.60, 0.46]
    return _interp(x, xs, ys)


def body_bottom(x):
    return _interp(x, [-2.2, -1.9, 1.9, 2.2], [0.30, 0.18, 0.18, 0.26])


def body_half_w(x, y=None):
    w = _interp(x, [-2.2, -1.9, -1.2, 1.4, 2.0, 2.2], [0.80, 0.93, 0.97, 0.95, 0.88, 0.70])
    return w


def cabin_w(x):
    """Lebar kabin (bagian atas lebih ramping dari bodi)."""
    return _interp(x, [-0.95, -0.4, 0.4, 0.75], [0.62, 0.70, 0.72, 0.80])


def section(x, th):
    """Titik pada penampang bodi di posisi x, sudut th (0 = samping kanan, pi/2 = atas)."""
    top, bot = body_top(x), body_bottom(x)
    yc = (top + bot) / 2
    hh = (top - bot) / 2
    s, c = np.sin(th), np.cos(th)
    y = yc + hh * np.sign(s) * np.abs(s) ** 0.5
    # makin ke atas makin ramping (kabin)
    up = np.clip((y - 0.72) / 0.4, 0, 1)
    in_cab = (x > -0.95) & (x < 0.75)
    w = body_half_w(x) * (1 - up) + np.where(in_cab, cabin_w(x), body_half_w(x) * 0.8) * up
    z = w * np.sign(c) * np.abs(c) ** 0.35
    return y, z


def in_arch(x, y, z):
    """True kalau titik berada di lubang spakbor (tempat roda)."""
    hit = np.zeros(np.shape(x), bool)
    for wx in WHEEL_X:
        d = np.hypot(x - wx, y - WHEEL_Y)
        hit |= (d < WHEEL_R + 0.07) & (np.abs(z) > 0.55)
    return hit


def make_body():
    pts = []
    # garis penampang melintang (rangka kawat)
    for x in np.arange(L_REAR + 0.02, L_FRONT, 0.16):
        th = np.linspace(-math.pi / 2, 3 * math.pi / 2, 160)
        y, z = section(np.full_like(th, x), th)
        pts.append(np.stack([np.full_like(th, x), y, z], 1))
    # garis memanjang
    for th0 in np.linspace(-math.pi / 2, 3 * math.pi / 2, 30, endpoint=False):
        x = np.linspace(L_REAR + 0.02, L_FRONT - 0.02, 260)
        y, z = section(x, np.full_like(x, th0))
        pts.append(np.stack([x, y, z], 1))
    # isian acak
    x = RNG.uniform(L_REAR, L_FRONT, 3500)
    th = RNG.uniform(-math.pi / 2, 3 * math.pi / 2, 3500)
    y, z = section(x, th)
    pts.append(np.stack([x, y, z], 1))
    P = np.concatenate(pts)
    P = P[~in_arch(P[:, 0], P[:, 1], P[:, 2])]
    # kaca: bagian atas kabin
    glass = (P[:, 1] > 0.80) & (P[:, 0] > -0.9) & (P[:, 0] < 0.72) & (np.abs(P[:, 2]) < 0.74)
    roof = glass & (P[:, 0] > -0.45) & (P[:, 0] < 0.25) & (np.abs(P[:, 2]) < 0.5)
    glass &= ~roof
    return P[~glass], P[glass]


def ring(center, normal, r, n):
    normal = np.asarray(normal, float)
    normal /= np.linalg.norm(normal)
    a = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(normal, a)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.asarray(center) + r * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))


def seg(a, b, n):
    t = np.linspace(0, 1, n)[:, None]
    return np.asarray(a) * (1 - t) + np.asarray(b) * t


def box(c, size, n=900, edges=True):
    c, size = np.asarray(c, float), np.asarray(size, float)
    pts = []
    if edges:
        for i in range(3):
            for s1 in (-0.5, 0.5):
                for s2 in (-0.5, 0.5):
                    a = np.zeros(3)
                    a[i] = -0.5
                    j, k = [m for m in range(3) if m != i]
                    a[j], a[k] = s1, s2
                    b = a.copy()
                    b[i] = 0.5
                    pts.append(seg(c + a * size, c + b * size, 40))
    # titik di permukaan
    f = RNG.uniform(-0.5, 0.5, (n, 3))
    ax = RNG.integers(0, 3, n)
    f[np.arange(n), ax] = np.sign(f[np.arange(n), ax]) * 0.5
    pts.append(c + f * size)
    return np.concatenate(pts)


def cylinder(c, axis, r, length, n_ring=10, per=60, fill=300):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    pts = []
    for t in np.linspace(-0.5, 0.5, n_ring):
        pts.append(ring(np.asarray(c) + axis * length * t, axis, r, per))
    a = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(axis, a)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    th = RNG.uniform(0, 2 * math.pi, fill)
    tt = RNG.uniform(-0.5, 0.5, fill)
    pts.append(np.asarray(c) + np.outer(tt * length, axis) + r * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v)))
    return np.concatenate(pts)


def gear(c, axis, r, teeth, width=0.04):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    pts = []
    n = teeth * 12
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rr = r * (1 + 0.12 * (np.sin(t * teeth) > 0))
    a = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(axis, a)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    for w in (-width / 2, width / 2):
        pts.append(np.asarray(c) + axis * w + rr[:, None] * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v)))
    pts.append(ring(c, axis, r * 0.35, 40))
    for k in range(6):
        ang = k * math.pi / 3
        d = math.cos(ang) * u + math.sin(ang) * v
        pts.append(seg(np.asarray(c) + d * r * 0.35, np.asarray(c) + d * r * 0.95, 15))
    return np.concatenate(pts)


def helix(base, h, r, turns, n=220):
    t = np.linspace(0, 1, n)
    a = t * turns * 2 * math.pi
    return np.stack([base[0] + r * np.cos(a), base[1] + t * h, base[2] + r * np.sin(a)], 1)


def make_wheel(wx, wz):
    side = 1 if wz > 0 else -1
    c = np.array([wx, WHEEL_Y, wz])
    ax = np.array([0, 0, 1.0])
    pts = []
    # ban: torus
    for k, dz in enumerate(np.linspace(-0.12, 0.12, 7)):
        bulge = 1 - 0.5 * (dz / 0.12) ** 2 * 0.25
        pts.append(ring(c + ax * dz, ax, WHEEL_R * bulge, 140))
        pts.append(ring(c + ax * dz, ax, WHEEL_R * 0.66, 90))
    # alur ban
    for ang in np.linspace(0, 2 * math.pi, 48, endpoint=False):
        d = np.array([math.cos(ang), math.sin(ang), 0])
        pts.append(seg(c + d * WHEEL_R + ax * -0.12, c + d * WHEEL_R + ax * 0.12, 8))
    # velg: 5 jari-jari di sisi luar
    face = c + ax * 0.1 * side
    pts.append(ring(face, ax, WHEEL_R * 0.64, 100))
    pts.append(ring(face, ax, 0.06, 30))
    for k in range(5):
        ang = k * 2 * math.pi / 5
        for off in (-0.05, 0.05):
            d = np.array([math.cos(ang + off), math.sin(ang + off), 0])
            pts.append(seg(face + d * 0.06, face + d * WHEEL_R * 0.62, 22))
    return np.concatenate(pts)


def make_brake(wx, wz):
    side = 1 if wz > 0 else -1
    c = np.array([wx, WHEEL_Y, wz - 0.04 * side])
    ax = np.array([0, 0, 1.0])
    pts = [ring(c, ax, r, int(60 + 200 * r)) for r in np.linspace(0.08, 0.2, 6)]
    # kaliper
    pts.append(box(c + np.array([0.12, 0.13, 0.03 * side]), [0.08, 0.1, 0.06], 120))
    return np.concatenate(pts)


def build_car():
    parts = []

    def add(name, P, col, d, label=True):
        parts.append({"name": name, "P": np.asarray(P, np.float32), "col": col,
                      "dir": np.asarray(d, np.float32), "label": label})

    body, glass = make_body()
    add("Bodi", body, (240, 236, 232), (0, 1.55, 0))
    add("Kaca kabin", glass, (255, 215, 130), (0, 2.25, 0))

    # roda & rem
    for wx, nmx in ((WHEEL_X[0], "depan"), (WHEEL_X[1], "belakang")):
        for wz, nmz in ((WHEEL_Z, "kanan"), (-WHEEL_Z, "kiri")):
            sx = 0.25 if wx > 0 else -0.25
            sz = 1.0 if wz > 0 else -1.0
            lbl = (nmx == "depan" and nmz == "kanan")
            add(f"Roda {nmx} {nmz}", make_wheel(wx, wz), (235, 235, 235), (sx, 0, 1.15 * sz),
                label=lbl)
            add(f"Rem cakram {nmx} {nmz}", make_brake(wx, wz), (70, 70, 255), (sx * 0.6, 0, 0.62 * sz),
                label=lbl)
            add(f"Suspensi {nmx} {nmz}", helix((wx, WHEEL_Y + 0.08, wz * 0.72), 0.42, 0.07, 7),
                (60, 225, 255), (sx * 0.4, 0.55, 0.35 * sz), label=lbl)

    # mesin V8 (di tengah-belakang, di balik jok)
    ex, ey = -1.0, 0.52
    eng = [box((ex, ey, 0), (0.75, 0.3, 0.4), 700)]
    for bank in (-1, 1):
        axis = np.array([0, math.cos(0.5), bank * math.sin(0.5)])
        for i in range(4):
            cx = ex - 0.27 + i * 0.18
            cc = np.array([cx, ey + 0.2, bank * 0.16])
            eng.append(cylinder(cc, axis, 0.065, 0.22, 4, 36, 60))
    eng.append(box((ex, ey + 0.38, 0), (0.7, 0.08, 0.16), 250))   # intake
    add("Mesin V8", np.concatenate(eng), (220, 90, 255), (-0.25, 1.0, 0))

    # transmisi & roda gigi
    tr = [cylinder((-1.62, 0.42, 0), (1, 0, 0), 0.13, 0.42, 6, 50, 250)]
    for i, r in enumerate((0.11, 0.08, 0.13, 0.09)):
        tr.append(gear((-1.48 - i * 0.09, 0.42, 0.0), (1, 0, 0), r, 12 + i * 3))
    add("Transmisi", np.concatenate(tr), (60, 205, 255), (-1.25, 0.35, 0))

    # poros penggerak ke roda belakang
    sh = [seg((-1.62, 0.40, -0.75), (-1.62, 0.40, 0.75), 120)]
    sh.append(ring((-1.62, 0.40, 0.0), (0, 0, 1), 0.05, 30))
    add("Poros roda", np.concatenate(sh), (60, 205, 255), (-0.8, -0.35, 0))

    # knalpot
    ex_pts = []
    for z in (-0.32, 0.32):
        ex_pts.append(cylinder((-1.75, 0.32, z), (1, 0, 0), 0.06, 0.85, 8, 30, 120))
        ex_pts.append(cylinder((-2.2, 0.36, z), (1, 0, 0), 0.075, 0.12, 4, 30, 40))
    ex_pts.append(box((-1.95, 0.3, 0), (0.28, 0.12, 0.5), 260))
    add("Knalpot", np.concatenate(ex_pts), (255, 220, 80), (-1.75, -0.25, 0))

    # rangka sasis
    ch = []
    for z in (-0.55, 0.55):
        ch.append(seg((-2.0, 0.22, z), (2.0, 0.22, z), 300))
        ch.append(seg((-2.0, 0.30, z), (2.0, 0.30, z), 300))
    for x in np.linspace(-1.9, 1.9, 9):
        ch.append(seg((x, 0.22, -0.55), (x, 0.22, 0.55), 70))
    for x0, x1 in ((-1.9, -0.6), (-0.6, 0.7), (0.7, 1.9)):
        ch.append(seg((x0, 0.22, -0.55), (x1, 0.22, 0.55), 90))
    add("Rangka sasis", np.concatenate(ch), (215, 170, 120), (0, -0.95, 0))

    # jok
    seats = []
    for z in (-0.33, 0.33):
        seats.append(box((-0.25, 0.42, z), (0.45, 0.1, 0.42), 300))
        bp = box((-0.5, 0.72, z), (0.1, 0.55, 0.42), 300)
        seats.append(bp)
    add("Jok", np.concatenate(seats), (235, 120, 215), (0, 1.05, 0))

    # setir & dasbor
    st = [ring((0.32, 0.78, 0.33), (1, 0.5, 0), 0.16, 90)]
    for k in range(3):
        a = k * 2 * math.pi / 3 + math.pi / 2
        st.append(seg((0.32, 0.78, 0.33), (0.32 - 0.16 * math.sin(a) * 0.45, 0.78 + 0.16 * math.sin(a) * 0.9,
                                           0.33 + 0.16 * math.cos(a)), 20))
    st.append(seg((0.32, 0.78, 0.33), (0.6, 0.62, 0.33), 40))
    add("Setir", np.concatenate(st), (255, 255, 255), (0.55, 1.25, 0.2))
    da = [seg((0.55, 0.78, z0), (0.55, 0.78, z0 + 0.1), 5) for z0 in np.linspace(-0.7, 0.6, 14)]
    da.append(box((0.6, 0.72, 0), (0.25, 0.12, 1.3), 500, edges=True))
    add("Dasbor", np.concatenate(da), (200, 200, 255), (0.95, 0.9, 0))

    # radiator & lampu
    rad = [box((1.95, 0.42, 0), (0.06, 0.26, 1.0), 200)]
    for y in np.linspace(0.3, 0.54, 8):
        rad.append(seg((1.95, y, -0.5), (1.95, y, 0.5), 80))
    for z in np.linspace(-0.5, 0.5, 14):
        rad.append(seg((1.95, 0.3, z), (1.95, 0.54, z), 20))
    add("Radiator", np.concatenate(rad), (255, 230, 120), (1.45, 0, 0))
    hl = np.concatenate([ring((2.08, 0.54, z), (1, 0.2, 0), r, 40) for z in (-0.6, 0.6) for r in (0.03, 0.06, 0.09)])
    add("Lampu depan", hl, (200, 255, 255), (0.9, 1.3, 0))
    tl = np.concatenate([seg((-2.17, 0.64, z0), (-2.17, 0.64, z0 + 0.35 * np.sign(z0)), 60) for z0 in (-0.35, 0.35)] +
                        [seg((-2.16, 0.6, z0), (-2.16, 0.6, z0 + 0.35 * np.sign(z0)), 60) for z0 in (-0.35, 0.35)])
    add("Lampu belakang", tl, (60, 60, 255), (-0.9, 1.3, 0))
    # pusatkan
    allp = np.concatenate([p["P"] for p in parts])
    ctr = (allp.min(0) + allp.max(0)) / 2
    for p in parts:
        p["P"] = p["P"] - ctr
        p["center"] = p["P"].mean(0)
    return parts


if __name__ == "__main__":
    ps = build_car()
    print(len(ps), "bagian,", sum(len(p["P"]) for p in ps), "titik")

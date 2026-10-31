#!/usr/bin/env python3
"""
Simulator Mesin 3D - Mobil Sport
================================
Model mobil sport dari titik-titik cahaya yang bisa diputar dan dibongkar
(exploded view) dengan gerakan tangan lewat webcam.

Gerakan tangan:
  Dua tangan          : jauhkan kedua tangan -> mobil terbongkar jadi bagian-bagiannya,
                        dekatkan -> mobil terakit lagi. Mobil berada di antara kedua tangan.
                        Miringkan kedua tangan (seperti memutar setir) -> mobil berputar.
  Satu tangan terbuka : geser tangan kiri-kanan / atas-bawah -> putar mobil.
  Cubit (jempol + telunjuk): jepit sebuah bagian lalu tarik keluar; lepas jepitan,
                        bagian itu tetap di tempat barunya. Bisa diulang untuk bagian lain.
  Satu tangan mengepal: mobil terakit kembali, semua bagian yang ditarik kembali ke tempatnya.
  Tanpa tangan        : mobil berputar pelan.

Tombol: L = nama bagian, H = titik tangan, I = info/fps, F = layar penuh/jendela,
        R = reset, Q/Esc = keluar.
Program terbuka layar penuh; jendela biasa: python simulator_mesin.py --jendela
Tanpa kamera (untuk mencoba): python simulator_mesin.py --demo
"""
import argparse
import math
import os
import shutil
import ssl
import subprocess
import time
import urllib.request

import cv2
import numpy as np

from model_mobil import build_car

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
WIN = "Simulator Mesin 3D"
HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
             (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
             (18, 19), (19, 20), (0, 17)]
GOLD = (90, 200, 245)
WHITE = (240, 240, 240)


PINCH_ON = 0.40   # jempol-telunjuk dianggap menjepit (relatif ukuran telapak)
PINCH_OFF = 0.58  # dilepas bila lebih dari ini
PINCH_NEAR = 0.75  # jari mulai mendekat untuk mencubit: mobil berhenti berputar
GRAB_WINDOW = 0.6  # detik setelah mulai mencubit, bagian masih bisa diambil


def smoothstep(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def rot(yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
    return Rx @ Ry


def approach(cur, target, rate, dt):
    return cur + (target - cur) * (1 - math.exp(-rate * dt))


class Simulator:
    def __init__(self):
        self.parts = build_car()
        P, D, C, idx = [], [], [], []
        for i, p in enumerate(self.parts):
            n = len(p["P"])
            P.append(p["P"])
            D.append(np.repeat(p["dir"][None], n, 0))
            C.append(np.repeat(np.array(p["col"], np.float32)[None], n, 0))
            idx.append(np.full(n, i))
        self.P = np.concatenate(P).astype(np.float32)
        self.Dir = np.concatenate(D).astype(np.float32)
        self.C = np.concatenate(C)
        self.pidx = np.concatenate(idx)
        self.psize = np.sqrt(np.bincount(self.pidx).astype(np.float32))
        self.phase = np.random.default_rng(3).uniform(0, 6.283, len(self.P)).astype(np.float32)
        self.proj = None
        self.R = rot(0.6, 0.28)
        self.size = None
        self.reset()
        self.hands = []
        self.show_labels = True
        self.show_skel = True
        self.show_info = False
        self.status = ""

    def reset(self):
        self.yaw, self.pitch = 0.6, 0.28
        self.e = 0.0
        self.e_target = 0.0
        self.center = None
        self.center_t = None
        self.drag_prev = None
        self.last_seen = -10.0
        self.pull = np.zeros((len(self.parts), 3), np.float32)  # geseran tiap bagian yang ditarik
        self.grab = None          # {"part": i, "pt": titik jepit}
        self.pinch_prev = []      # (titik jepit, waktu mulai) frame sebelumnya
        self.now = 0.0
        self.returning = False

    def set_size(self, w, h):
        if self.size == (w, h):
            return
        self.size = (w, h)
        self.w, self.h = w, h
        self.center = np.array([w / 2, h / 2], float)
        self.center_t = self.center.copy()

    # ------------------------------------------------------------ tangan
    def classify(self, sc):
        """Kembalikan (pose, titik jepit, rasio jepit). Pose: pinch / fist / open."""
        d = lambda a, b: np.linalg.norm(sc[a] - sc[b])
        palm = d(0, 9) + 1e-6
        pt = (sc[4] + sc[8]) / 2
        ratio = d(4, 8) / palm
        index_out = d(8, 0) / (d(5, 0) + 1e-6)   # telunjuk tidak tertekuk ke telapak
        was = any(np.linalg.norm(pt - q) < self.h * 0.1 for q, _ in self.pinch_prev)
        if (ratio < PINCH_ON and index_out > 1.05) or (was and ratio < PINCH_OFF):
            return "pinch", pt, ratio
        r = [d(t, 0) / (d(p, 0) + 1e-6) for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))]
        if all(x < 1.0 for x in r):
            return "fist", None, ratio
        return "open", None, ratio

    def process_hands(self, result, now):
        self.hands = []
        lms = (getattr(result, "hand_landmarks", None) if result is not None else None) or []
        for lm in lms:
            sc = np.array([[p.x * self.w, p.y * self.h] for p in lm])
            palm = sc[[0, 5, 9, 13, 17]].mean(0)
            pose, pt, ratio = self.classify(sc)
            self.hands.append({"sc": sc, "palm": palm, "pose": pose, "pt": pt, "ratio": ratio})
        self.hands.sort(key=lambda hd: hd["palm"][0])  # kiri ke kanan

    def pick(self, x, y):
        """Bagian yang paling banyak titiknya di dekat titik jepit."""
        if self.proj is None:
            return -1
        px, py, ok = self.proj
        r = max(36.0, self.h * 0.06)
        near = ok & ((px - x) ** 2 + (py - y) ** 2 < r * r)
        if not near.any():  # tidak ada titik dekat: ambil bagian yang pusatnya terdekat
            best, bd = -1, self.h * 0.1
            for i in range(len(self.parts)):
                cx, cy, _ = self.project(self.part_pos(i)[None])
                dd = math.hypot(cx[0] - x, cy[0] - y)
                if dd < bd:
                    best, bd = i, dd
            return best
        cnt = np.bincount(self.pidx[near], minlength=len(self.parts))
        return int((cnt / self.psize).argmax())  # bagian kecil tidak kalah oleh bodi yang besar

    def part_pos(self, i):
        p = self.parts[i]
        return (p["center"] + p["dir"] * self.e + self.pull[i]) @ self.R.T

    def update_pinch(self):
        """Cubit (jempol + telunjuk) sebuah bagian lalu tarik keluar."""
        pins = [h["pt"] for h in self.hands if h["pose"] == "pinch"]
        if self.grab is not None:
            near = [q for q in pins if np.linalg.norm(q - self.grab["pt"]) < self.h * 0.15]
            if not near:
                self.grab = None                    # jepitan dilepas: bagian tetap di tempatnya
            else:
                q = min(near, key=lambda v: np.linalg.norm(v - self.grab["pt"]))
                d = q - self.grab["pt"]
                c = self.part_pos(self.grab["part"])
                z = 9.0 - c[2]
                f = 0.42 * self.w / 4.6 * 9.0
                cam = np.array([d[0] * z / f, -d[1] * z / f, 0.0], np.float32)
                self.pull[self.grab["part"]] += cam @ self.R  # kembali ke koordinat model
                self.grab["pt"] = q
                self.returning = False
        # catat sejak kapan tiap jepitan dimulai
        cur = []
        for q in pins:
            t0 = self.now
            for o, ot in self.pinch_prev:
                if np.linalg.norm(q - o) < self.h * 0.1:
                    t0 = ot
                    break
            cur.append((q, t0))
        if self.grab is None:
            for q, t0 in cur:
                if self.now - t0 <= GRAB_WINDOW:  # jepitan yang baru dimulai bisa mengambil bagian
                    i = self.pick(q[0], q[1])
                    if i >= 0:
                        self.grab = {"part": i, "pt": q}
                        break
        self.pinch_prev = cur

    def update(self, now, dt, demo=False):
        self.now = now
        if not demo:
            self.update_pinch()
        hs = [h for h in self.hands if h["pose"] != "pinch"]
        if demo:
            self.e_target = 0.5 - 0.5 * math.cos(now * 0.5)
            self.yaw += 0.3 * dt
        elif len(hs) >= 2:
            a, b = hs[0]["palm"], hs[1]["palm"]
            dist = np.linalg.norm(a - b) / self.w
            self.e_target = float(np.clip((dist - 0.22) / 0.38, 0, 1))
            self.center_t = (a + b) / 2
            ang = math.atan2(b[1] - a[1], b[0] - a[0])  # miring seperti setir
            if abs(ang) > 0.12:
                self.yaw += (ang - math.copysign(0.12, ang)) * 2.5 * dt
            self.drag_prev = None
            self.last_seen = now
        elif len(hs) == 1:
            h = hs[0]
            self.last_seen = now
            if h["pose"] == "fist":
                self.e_target = 0.0
                self.returning = True               # kepal: semua bagian kembali ke tempatnya
                self.drag_prev = None
            elif self.grab is None and h["ratio"] > PINCH_NEAR:  # jari mendekat untuk mencubit: diam
                p = h["palm"]
                if self.drag_prev is not None:
                    d = p - self.drag_prev
                    self.yaw += d[0] / self.w * 6.0
                    self.pitch = float(np.clip(self.pitch + d[1] / self.h * 3.0, -0.5, 1.1))
                self.drag_prev = p
            else:
                self.drag_prev = None              # sedang/akan mencubit: mobil tidak diputar
        else:
            self.drag_prev = None
            if self.hands:
                self.last_seen = now
            elif now - self.last_seen > 1.0:  # tanpa tangan: berputar pelan, kembali ke tengah
                self.yaw += 0.25 * dt
                self.center_t = np.array([self.w / 2, self.h / 2], float)
        self.e = approach(self.e, self.e_target, 6.0, dt)
        if self.returning:
            self.pull *= math.exp(-5.0 * dt)
            if np.abs(self.pull).max() < 0.01:
                self.pull[:] = 0
                self.returning = False
        k = 1 - math.exp(-8 * dt)
        self.center = self.center + (self.center_t - self.center) * k

    # ------------------------------------------------------------ gambar
    def project(self, Q):
        sc = 0.42 * self.w / 4.6  # panjang mobil ~42% lebar layar
        Dcam = 9.0
        z = Dcam - Q[:, 2]
        f = sc * Dcam
        x = self.center[0] + f * Q[:, 0] / z
        y = self.center[1] - f * Q[:, 1] / z
        return x, y, z

    def draw_points(self, img, now):
        R = rot(self.yaw, self.pitch)
        self.R = R
        Q = (self.P + self.Dir * self.e + self.pull[self.pidx]) @ R.T
        mult = np.ones(len(self.P), np.float32)
        if self.grab is not None:
            mult[self.pidx == self.grab["part"]] = 1.9   # bagian yang dijepit menyala
        x, y, z = self.project(Q)
        xi = x.astype(np.int32)
        yi = y.astype(np.int32)
        ok = (xi >= 0) & (xi < self.w) & (yi >= 0) & (yi < self.h)
        depth = np.clip(0.55 + 0.18 * Q[:, 2], 0.3, 1.0)  # bagian belakang lebih redup
        tw = 0.8 + 0.2 * np.sin(now * 3.0 + self.phase)
        bright = (depth * tw * 0.6 * mult)[ok]
        idx = yi[ok] * self.w + xi[ok]
        buf = np.zeros((self.h * self.w, 3), np.float32)
        col = self.C[ok]
        for c in range(3):
            buf[:, c] = np.bincount(idx, weights=col[:, c] * bright, minlength=self.h * self.w)
        buf = buf.reshape(self.h, self.w, 3)
        small = cv2.resize(buf, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
        glow = cv2.GaussianBlur(small, (0, 0), 2.2)
        glow = cv2.resize(glow, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        out = img.astype(np.float32) + np.minimum(buf, 255) + glow * 3.5
        np.clip(out, 0, 255, out=out)
        img[:] = out.astype(np.uint8)
        self.proj = (x, y, ok)
        return R

    def draw_labels(self, img, R):
        if not self.show_labels:
            return
        a_exp = smoothstep(0.3, 0.6, self.e)
        grabbed = self.grab["part"] if self.grab is not None else -1
        ov = img.copy()
        faded = False
        for i, p in enumerate(self.parts):
            pulled = float(np.linalg.norm(self.pull[i]))
            solid = i == grabbed or pulled > 0.3       # bagian yang ditarik: nama selalu tampil
            if not solid and not (p["label"] and a_exp > 0.02):
                continue
            c = self.part_pos(i)
            x, y, _ = self.project(c[None])
            px, py = float(x[0]), float(y[0])
            d = np.array([px, py]) - self.center
            n = np.linalg.norm(d)
            d = d / n if n > 1 else np.array([1.0, 0.0])
            ax, ay = px + d[0] * 55, py + d[1] * 55
            side = 1 if d[0] >= 0 else -1
            name = p["name"] if solid else p["name"].replace(" depan kanan", "")
            col = GOLD if i == grabbed else WHITE
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ex = ax + side * (tw + 12)
            tx = int(ax + 6) if side > 0 else int(ex + 6)
            for canvas in ((img,) if solid else (ov,)):
                cv2.circle(canvas, (int(px), int(py)), 3, col, -1, cv2.LINE_AA)
                cv2.line(canvas, (int(px), int(py)), (int(ax), int(ay)), (200, 200, 200), 1, cv2.LINE_AA)
                cv2.line(canvas, (int(ax), int(ay)), (int(ex), int(ay)), (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(canvas, name, (tx, int(ay) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            faded |= not solid
        if faded:
            cv2.addWeighted(ov, a_exp, img, 1 - a_exp, 0, img)

    def draw_hud(self, img, now):
        ov = img.copy()
        for k, hd in enumerate(self.hands[:2]):
            c = tuple(int(v) for v in hd["palm"])
            r = int(self.h * 0.33)
            cv2.circle(ov, c, r, WHITE, 1, cv2.LINE_AA)
            a0 = (now * 40 * (1 if k == 0 else -1)) % 360
            cv2.ellipse(ov, c, (r + 10, r + 10), 0, a0, a0 + 110, GOLD, 2, cv2.LINE_AA)
            cv2.ellipse(ov, c, (r - 14, r - 14), 0, -a0, -a0 + 60, GOLD, 1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
        if self.show_skel:
            for hd in self.hands:
                sc = hd["sc"].astype(int)
                for a, b in HAND_CONN:
                    cv2.line(img, tuple(sc[a]), tuple(sc[b]), (225, 225, 225), 1, cv2.LINE_AA)
                for i in (4, 8, 12, 16, 20):
                    cv2.circle(img, tuple(sc[i]), 4, GOLD, -1, cv2.LINE_AA)
        for hd in self.hands:  # titik jepit
            if hd["pose"] == "pinch":
                q = tuple(int(v) for v in hd["pt"])
                held = self.grab is not None and np.linalg.norm(hd["pt"] - self.grab["pt"]) < 1
                cv2.circle(img, q, 14 if held else 10, GOLD, 2 if held else 1, cv2.LINE_AA)
        if self.show_info:  # tekan I: lihat bentuk tangan yang terbaca
            for hd in self.hands:
                x, y = hd["sc"][0].astype(int)
                cv2.putText(img, f"{hd['pose']} {hd['ratio']:.2f}", (x - 40, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, GOLD, 2, cv2.LINE_AA)
        if self.show_info and self.status:
            cv2.putText(img, self.status, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

    def render(self, img, now, dt, demo=False):
        self.update(now, dt, demo)
        img[:] = (img.astype(np.float32) * 0.55).astype(np.uint8)  # latar kamera diredupkan
        R = self.draw_points(img, now)
        self.draw_labels(img, R)
        self.draw_hud(img, now)


# ------------------------------------------------------------------ MediaPipe
def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return True
    print("File hand_landmarker.task belum ada. Mengunduh (sekitar 7 MB)...")
    try:
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        with urllib.request.urlopen(MODEL_URL, context=ctx) as r, open(MODEL_PATH, "wb") as f:
            shutil.copyfileobj(r, f)
        print("Model berhasil diunduh.")
        return True
    except Exception as exc:  # noqa: BLE001
        print("  urllib gagal:", exc)
    if shutil.which("curl"):
        try:
            subprocess.run(["curl", "-L", "-s", "-o", MODEL_PATH, MODEL_URL], check=True, timeout=120)
            if os.path.getsize(MODEL_PATH) > 1_000_000:
                print("Model berhasil diunduh (curl).")
                return True
        except Exception as exc:  # noqa: BLE001
            print("  curl gagal:", exc)
    print("Gagal mengunduh model. Salin hand_landmarker.task ke folder ini.")
    return False


def make_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp, vision.HandLandmarker.create_from_options(opts)


def screen_size():
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:  # noqa: BLE001
        pass
    return 1440, 900


def fit_cover(frame, aspect):
    h, w = frame.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect))
        x0 = (w - nw) // 2
        return frame[:, x0:x0 + nw]
    nh = int(round(w / aspect))
    y0 = (h - nh) // 2
    return frame[y0:y0 + nh]


def main():
    ap = argparse.ArgumentParser(description="Simulator Mesin 3D")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="tanpa kamera; mobil membongkar-merakit sendiri")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--jendela", action="store_true", help="jendela biasa, bukan layar penuh")
    args = ap.parse_args()
    sw, sh = screen_size()
    aspect = sw / sh
    print(f"Layar: {sw}x{sh}")
    print("Menyiapkan model mobil...")
    sim = Simulator()
    print(f"  {len(sim.parts)} bagian, {len(sim.P)} titik")

    cap = landmarker = mp = None
    if not args.demo:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            print("Kamera tidak bisa dibuka. Izinkan kamera untuk VS Code/Terminal di "
                  "System Settings > Privacy & Security > Camera. Menjalankan mode demo.")
            cap = None
            args.demo = True
        else:
            print("Kamera: OK")
            if ensure_model():
                try:
                    mp, landmarker = make_landmarker()
                    print("Pelacak tangan: OK")
                except Exception as exc:  # noqa: BLE001
                    print("Pelacak tangan GAGAL dimuat:", exc)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    fullscreen = not args.jendela
    if fullscreen:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WIN, int(720 * aspect), 720)
    demo_h = 720
    demo_w = int(round(demo_h * aspect))

    t0 = time.perf_counter()
    last = t0
    last_ts = -1
    fps = 0.0
    while True:
        now_abs = time.perf_counter()
        dt = min(0.05, now_abs - last)
        last = now_abs
        now = now_abs - t0
        fps = fps * 0.9 + (1 / max(dt, 1e-3)) * 0.1
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                print("Gagal membaca kamera.")
                break
            frame = fit_cover(cv2.flip(frame, 1), aspect)
        else:
            frame = np.full((demo_h, demo_w, 3), (40, 34, 30), np.uint8)
        sim.set_size(frame.shape[1], frame.shape[0])
        if landmarker is not None:
            ts = max(int(now * 1000), last_ts + 1)
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            sim.process_hands(res, now)
        sim.status = f"{len(sim.hands)} tangan | {fps:4.0f} fps"
        sim.render(frame, now, dt, demo=args.demo)
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("l"):
            sim.show_labels = not sim.show_labels
        elif key == ord("h"):
            sim.show_skel = not sim.show_skel
        elif key == ord("i"):
            sim.show_info = not sim.show_info
        elif key == ord("r"):
            sim.reset()
            sim.size = None
            sim.set_size(frame.shape[1], frame.shape[0])
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    if cap is not None:
        cap.release()
    if landmarker is not None:
        landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

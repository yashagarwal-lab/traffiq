"""
track_sim.py — Raycasting simulator for the TraffIQ competition track.
Renders a first-person 3D view (white walls, black floor, white ceiling)
and feeds frames to Model.predict() in real time.

Team SafeNSound — Sahil Sharma & Yash Agarwal
"""

import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import pygame
from Model import Model

# ═══════════════════════════════════════════════════════════════════════
#  TRACK PARAMETERS  —  tweak these to test different layouts
# ═══════════════════════════════════════════════════════════════════════

CELL            = 4       # cm per grid cell
TRACK_WIDTH_CM  = 60      # corridor width (cm)
TRACK_WIDTH     = TRACK_WIDTH_CM // CELL

CORRIDOR_0_LEN  = 340     # top horizontal → right  (cm)
CORRIDOR_1_LEN  = 270     # right vertical → down   (cm)
CORRIDOR_2_LEN  = 150     # U-turn horizontal → left(cm)
CORRIDOR_3_LEN  = 560     # main vertical → down    (cm)

TL_FRAC         = 0.45    # traffic light position on corridor 1
BOX_FRAC        = 0.25    # box position on corridor 3
STOP_FRAC       = 0.92    # stop sign position on corridor 3
BOX_LATERAL     = 4       # box offset from centre (cells, +ve = right)

TL_RED_DURATION = 20.0    # seconds the light stays red
CAR_MAX_SPEED   = 18.0    # grid-cells / second at full throttle
CAR_LENGTH_CM   = 25      # RC car body length (cm)
CAR_WIDTH_CM    = 16      # RC car body width (cm)
CAR_RADIUS      = CAR_WIDTH_CM / CELL / 2  # collision half-width in cells (2.0)

# ═══════════════════════════════════════════════════════════════════════
#  RENDERING
# ═══════════════════════════════════════════════════════════════════════

FOV       = math.radians(78)
FRAME_W   = 640
FRAME_H   = 480
HORIZON   = int(FRAME_H * 0.40)
NUM_RAYS  = 320

# ═══════════════════════════════════════════════════════════════════════
#  DERIVED GEOMETRY (auto-computed — don't touch)
# ═══════════════════════════════════════════════════════════════════════

HW        = TRACK_WIDTH // 2
C0_Y      = 5 + HW
C0_X_END  = 5 + CORRIDOR_0_LEN // CELL
C1_X      = C0_X_END - HW
C1_Y_END  = 5 + CORRIDOR_1_LEN // CELL
C2_Y      = C1_Y_END - HW
C2_X_ST   = C1_X - CORRIDOR_2_LEN // CELL
C3_X      = C2_X_ST + HW
C3_Y_END  = C2_Y + CORRIDOR_3_LEN // CELL

MAP_W     = C0_X_END + 15
MAP_H     = C3_Y_END + 15

TL_POS    = (float(C1_X), float(5 + int(TL_FRAC * CORRIDOR_1_LEN / CELL)))
BOX_POS   = (float(C3_X + BOX_LATERAL),
             float(C2_Y + int(BOX_FRAC * CORRIDOR_3_LEN / CELL)))
STOP_POS  = (float(C3_X),
             float(C2_Y + int(STOP_FRAC * CORRIDOR_3_LEN / CELL)))

CAR_START = (5.0 + CAR_RADIUS + 1, float(C0_Y))


# ═══════════════════════════════════════════════════════════════════════
#  MAP
# ═══════════════════════════════════════════════════════════════════════

def create_map():
    grid = np.ones((MAP_H, MAP_W), np.uint8)
    grid[C0_Y - HW : C0_Y + HW, 5 : C0_X_END]                       = 0  # corridor 0
    grid[5 : C1_Y_END, C1_X - HW : C1_X + HW]                       = 0  # corridor 1
    grid[C2_Y - HW : C2_Y + HW, C2_X_ST : C0_X_END]                 = 0  # corridor 2
    grid[C2_Y - HW : min(C3_Y_END, MAP_H - 1), C3_X - HW : C3_X + HW] = 0  # corridor 3
    return grid


# ═══════════════════════════════════════════════════════════════════════
#  RAY CASTING (DDA)
# ═══════════════════════════════════════════════════════════════════════

def dda(px, py, angle, grid):
    dx, dy = math.cos(angle), math.sin(angle)
    mx, my = int(px), int(py)
    ddx = abs(1 / dx) if abs(dx) > 1e-10 else 1e10
    ddy = abs(1 / dy) if abs(dy) > 1e-10 else 1e10

    if dx < 0: sx, sdx = -1, (px - mx) * ddx
    else:      sx, sdx = 1, (mx + 1 - px) * ddx
    if dy < 0: sy, sdy = -1, (py - my) * ddy
    else:      sy, sdy = 1, (my + 1 - py) * ddy

    side = 0
    for _ in range(250):
        if sdx < sdy:
            sdx += ddx; mx += sx; side = 0
        else:
            sdy += ddy; my += sy; side = 1
        if not (0 <= mx < MAP_W and 0 <= my < MAP_H):
            return 150.0, 0
        if grid[my, mx]:
            if side == 0:
                d = (mx - px + (1 - sx) / 2) / (dx or 1e-10)
            else:
                d = (my - py + (1 - sy) / 2) / (dy or 1e-10)
            return max(abs(d), 0.05), side
    return 150.0, 0


# ═══════════════════════════════════════════════════════════════════════
#  3-D FRAME RENDERER
# ═══════════════════════════════════════════════════════════════════════

def render_frame(cx, cy, ca, grid, tl_state):
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    frame[:HORIZON] = 220  # white ceiling

    dists  = np.empty(NUM_RAYS)
    sides  = np.empty(NUM_RAYS, np.int32)
    scale  = FRAME_W / NUM_RAYS

    for i in range(NUM_RAYS):
        ra = ca - FOV / 2 + FOV * i / NUM_RAYS
        d, s = dda(cx, cy, ra, grid)
        dists[i] = d * math.cos(ra - ca)  # fisheye fix
        sides[i] = s

    heights = np.clip(FRAME_H / np.maximum(dists, 0.1), 1, FRAME_H * 3).astype(int)
    shades  = np.clip(np.where(sides == 0, 215, 190) - (dists * 2).astype(int), 80, 220)

    for i in range(NUM_RAYS):
        x0 = int(i * scale)
        x1 = int((i + 1) * scale)
        h = heights[i]
        t = max(0, HORIZON - int(h * 0.35))
        b = min(FRAME_H, HORIZON + int(h * 0.65))
        frame[t:b, x0:x1] = int(shades[i])

    _render_sprites(frame, cx, cy, ca, tl_state, dists)
    return frame


def _project(ox, oy, cx, cy, ca):
    rx, ry = ox - cx, oy - cy
    cos_a, sin_a = math.cos(-ca), math.sin(-ca)
    depth   = rx * cos_a - ry * sin_a
    lateral = rx * sin_a + ry * cos_a
    if depth <= 0.3:
        return None, None, None
    hp = math.tan(FOV / 2)
    sx = int(FRAME_W / 2 + lateral / depth / hp * FRAME_W / 2)
    return sx, depth, FRAME_H / depth


def _render_sprites(frame, cx, cy, ca, tl_state, wall_dists):
    def _ray(sx):
        return int(np.clip(sx / (FRAME_W / NUM_RAYS), 0, NUM_RAYS - 1))

    # Traffic light
    sx, d, sc = _project(*TL_POS, cx, cy, ca)
    if sx is not None and d < 50 and d < wall_dists[_ray(sx)] + 1:
        sz = max(6, int(sc * 0.25))
        sy = HORIZON - int(sc * 0.3)
        hw, r = sz, max(4, sz // 2)
        cv2.rectangle(frame, (max(0, sx-hw), max(0, sy-sz*2)),
                      (min(FRAME_W, sx+hw), min(FRAME_H, sy+sz*2)), (25,25,25), -1)
        if tl_state == "red":
            cv2.circle(frame, (sx, sy-r-3), r, (255,20,20), -1)
            cv2.circle(frame, (sx, sy+r+3), r, (15,40,15), -1)
        else:
            cv2.circle(frame, (sx, sy-r-3), r, (40,15,15), -1)
            cv2.circle(frame, (sx, sy+r+3), r, (20,255,20), -1)

    # Box
    sx, d, sc = _project(*BOX_POS, cx, cy, ca)
    if sx is not None and d < 50 and d < wall_dists[_ray(sx)] + 1:
        sz = max(12, int(sc * 0.5))
        sy = HORIZON + int(sc * 0.15)
        cv2.rectangle(frame, (max(0,sx-sz), max(0,sy-sz)),
                      (min(FRAME_W,sx+sz), min(FRAME_H,sy+sz)), (150,120,90), -1)
        cv2.rectangle(frame, (max(0,sx-sz), max(0,sy-sz)),
                      (min(FRAME_W,sx+sz), min(FRAME_H,sy+sz)), (90,70,45), 3)

    # Stop sign
    sx, d, sc = _project(*STOP_POS, cx, cy, ca)
    if sx is not None and d < 50 and d < wall_dists[_ray(sx)] + 1:
        sz = max(10, int(sc * 0.4))
        sy = HORIZON
        pts = [[int(sx + sz*math.cos(math.pi/8 + k*math.pi/4)),
                int(sy + sz*math.sin(math.pi/8 + k*math.pi/4))] for k in range(8)]
        cv2.fillPoly(frame, [np.array(pts, np.int32)], (210,15,15))
        fs = max(0.25, sz / 45)
        cv2.putText(frame, "STOP", (sx-sz//2, sy+sz//4),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), max(1, sz//18))


# ═══════════════════════════════════════════════════════════════════════
#  CAR PHYSICS
# ═══════════════════════════════════════════════════════════════════════

class Car:
    def __init__(self):
        self.px, self.py = CAR_START
        self.angle = 0.0
        self.vel   = 0.0
        self.done  = False

    def step(self, speed, steer, dt, grid):
        if self.done:
            return
        target = speed * CAR_MAX_SPEED
        self.vel += (target - self.vel) * min(1.0, 4.0 * dt)
        self.angle += steer * 2.2 * dt

        nx = self.px + self.vel * math.cos(self.angle) * dt
        ny = self.py + self.vel * math.sin(self.angle) * dt

        if self._passable(nx, ny, grid):
            self.px, self.py = nx, ny
        elif self._passable(nx, self.py, grid):
            self.px = nx; self.vel *= 0.2
        elif self._passable(self.px, ny, grid):
            self.py = ny; self.vel *= 0.2
        else:
            self.vel *= 0.2

    @staticmethod
    def _passable(x, y, grid):
        r = CAR_RADIUS
        for ox, oy in ((-r,0),(r,0),(0,-r),(0,r),(0,0)):
            gx, gy = int(x + ox), int(y + oy)
            if not (0 <= gx < MAP_W and 0 <= gy < MAP_H) or grid[gy, gx]:
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════
#  MINIMAP
# ═══════════════════════════════════════════════════════════════════════

def render_minimap(grid, car, tl_state):
    S = 3
    m = np.zeros((MAP_H, MAP_W, 3), np.uint8)
    m[grid == 1] = 90; m[grid == 0] = 10
    m = cv2.resize(m, (MAP_W * S, MAP_H * S), interpolation=cv2.INTER_NEAREST)

    # obstacles
    c = (0,0,255) if tl_state == "red" else (0,255,0)
    cv2.circle(m, (int(TL_POS[0]*S), int(TL_POS[1]*S)), 5, c, -1)
    cv2.rectangle(m, (int((BOX_POS[0]-2)*S), int((BOX_POS[1]-2)*S)),
                  (int((BOX_POS[0]+2)*S), int((BOX_POS[1]+2)*S)), (90,120,150), -1)
    cv2.circle(m, (int(STOP_POS[0]*S), int(STOP_POS[1]*S)), 5, (0,0,210), -1)

    # car + heading + FOV
    cx, cy = int(car.px*S), int(car.py*S)
    cv2.circle(m, (cx,cy), 4, (255,170,0), -1)
    cv2.line(m, (cx,cy), (cx+int(12*math.cos(car.angle)),
             cy+int(12*math.sin(car.angle))), (0,255,255), 2)
    for off in (-FOV/2, FOV/2):
        a = car.angle + off
        cv2.line(m, (cx,cy), (cx+int(30*math.cos(a)), cy+int(30*math.sin(a))), (80,80,0), 1)

    # labels
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(m, "TL",   (int(TL_POS[0]*S)+8,   int(TL_POS[1]*S)+4),   f, 0.35, (0,255,255), 1)
    cv2.putText(m, "BOX",  (int(BOX_POS[0]*S)+8,   int(BOX_POS[1]*S)+4),  f, 0.35, (0,255,255), 1)
    cv2.putText(m, "STOP", (int(STOP_POS[0]*S)+8,   int(STOP_POS[1]*S)+4), f, 0.35, (0,255,255), 1)
    cv2.putText(m, "START",(int(CAR_START[0]*S),     int(CAR_START[1]*S)-10),f, 0.30, (0,200,200), 1)
    return m


# ═══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    model = Model(); model.load()
    grid  = create_map()
    car   = Car()

    pygame.init()
    S = 3
    mw, mh = MAP_W * S, MAP_H * S
    screen = pygame.display.set_mode((mw + FRAME_W + 15, max(mh, FRAME_H) + 70))
    pygame.display.set_caption("TraffIQ Sim — Team SafeNSound")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("monospace", 13)

    tl, t_sim = "red", 0.0
    dt = 1 / 30

    print("[R] Reset  [T] Toggle light  [ESC] Quit")

    running = True
    while running:
        t_sim += dt

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    car = Car(); model.stopped_for_red = False
                    model.stop_sign_seen = model.stop_sign_frames = model.frame_count = 0
                    tl, t_sim = "red", 0.0
                elif ev.key == pygame.K_t:
                    tl = "green" if tl == "red" else "red"

        if t_sim > TL_RED_DURATION and tl == "red":
            tl = "green"; print(">>> Light: GREEN")

        t0 = time.perf_counter()
        cam = render_frame(car.px, car.py, car.angle, grid, tl)
        r_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        spd, dr = model.predict(cam)
        p_ms = (time.perf_counter() - t0) * 1000

        car.step(spd, dr, dt, grid)
        if spd < 0.01 and car.py > C3_Y_END - 15 and not car.done:
            car.done = True
            print(f"🏁 FINISHED — {t_sim:.1f}s")

        # draw
        screen.fill((20,20,20))
        mini = render_minimap(grid, car, tl)
        screen.blit(pygame.surfarray.make_surface(
            cv2.cvtColor(mini, cv2.COLOR_BGR2RGB).swapaxes(0,1)), (0,0))
        screen.blit(pygame.surfarray.make_surface(cam.swapaxes(0,1)), (mw+15, 0))

        st = "STOPPED" if car.done else ("RED STOP" if model.stopped_for_red else "DRIVING")
        col = (100,255,100) if st == "DRIVING" else (255,100,100)
        y = max(mh, FRAME_H) + 10
        screen.blit(font.render(
            f"{st}  spd={spd:.2f}  str={dr:+.2f}  vel={car.vel:.1f}  "
            f"ang={math.degrees(car.angle)%360:.0f}°  t={t_sim:.1f}s", True, col), (10, y))
        screen.blit(font.render(
            f"render={r_ms:.0f}ms  predict={p_ms:.1f}ms  light={tl.upper()}  "
            f"stop_sign={model.stop_sign_seen}", True, (160,160,160)), (10, y+16))
        screen.blit(font.render(
            "[R] Reset   [T] Toggle light   [ESC] Quit", True, (100,100,100)), (10, y+32))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()

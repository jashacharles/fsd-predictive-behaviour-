import pygame
import random
import sys
import csv
from collections import deque
from pathlib import Path

import numpy as np

pygame.init()

# =========================
# CONFIG
# =========================

WIDTH, HEIGHT = 900, 700
FPS = 30

EPISODE_SECONDS = 10
EPISODE_FRAME_LIMIT = FPS * EPISODE_SECONDS



SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Behavior-Aware Driving Simulator")
CLOCK = pygame.time.Clock()

BLACK = (0, 0, 0)
ROAD = (80, 80, 80)
WHITE = (240, 240, 240)
YELLOW = (240, 220, 80)
ORANGE = (240, 120, 40)
RED = (220, 60, 60)
BLUE = (60, 140, 240)
GRAY = (170, 170, 170)
GREEN = (60, 200, 100)
PURPLE = (170, 80, 220)
CYAN = (60, 220, 220)
PINK = (240, 90, 160)

ROAD_WIDTH = 390
ROAD_X = WIDTH // 2 - ROAD_WIDTH // 2

LANE_COUNT = 3
LANE_WIDTH = ROAD_WIDTH // LANE_COUNT
LANES = [ROAD_X + LANE_WIDTH // 2 + i * LANE_WIDTH for i in range(LANE_COUNT)]

CAR_W, CAR_H = 45, 85
EGO_START_SCREEN_Y = HEIGHT - 130

SAFE_SPAWN_GAP = 230
INITIAL_SPAWN_GAP = 150
FOLLOW_DISTANCE = 180
HARD_BRAKE_DISTANCE = 120

LANE_CHANGE_REAR_GAP = 170
LANE_CHANGE_FRONT_GAP = 230
SIGNAL_FRAMES_BEFORE_LANE_CHANGE = 45
LANE_CHANGE_COOLDOWN_FRAMES = 90

# Keep the current view: not too much god-view.
VISIBLE_AHEAD_DISTANCE = 360
VISIBLE_BEHIND_DISTANCE = 60
DRIVER_VIEW_TOP_Y = EGO_START_SCREEN_Y - VISIBLE_AHEAD_DISTANCE

# The model receives a fixed-size snapshot of nearby visible cars every frame.
MAX_VISIBLE_CARS_FEATURES = 12
NO_CAR_SENTINEL = 9999.0

DATASET_PATH = Path("driving_data.csv")

SCENARIO_TEMPLATES = [
    {
        "id": "normal_front_stay",
        "description": "Normal car ahead with stable speed; usually stay in lane.",
        "expected_action": "stay",
    },
    {
        "id": "slow_front_open_lane",
        "description": "Slow car ahead with open adjacent lanes; lane change is usually reasonable.",
        "expected_action": "change_lane",
    },
    {
        "id": "brake_tapper_front",
        "description": "Brake-tapping car ahead; slow down or change lane.",
        "expected_action": "change_lane_or_brake",
    },
    {
        "id": "drifter_front",
        "description": "Drifting car ahead; lane change is usually reasonable.",
        "expected_action": "change_lane",
    },
    {
        "id": "blocked_adjacent_lane",
        "description": "Problem car ahead but adjacent lanes are blocked; staying or braking may be better.",
        "expected_action": "stay_or_brake",
    },
]


CURRENT_SCENARIO = random.choice(SCENARIO_TEMPLATES)

# Run modes:
# - default: 10-second scenario episode for training CSVs
# - full/eval/evaluation: full gameplay session for evaluation.csv
# - ghost: full gameplay session with live model predictions
RUN_MODE = sys.argv[1].lower() if len(sys.argv) > 1 else "scenario"
FULL_GAME_MODE = RUN_MODE in ["full", "eval", "evaluation"]
GHOST_MODE = RUN_MODE == "ghost"

if FULL_GAME_MODE or GHOST_MODE:
    DATASET_PATH = Path("evaluation.csv")
    CURRENT_SCENARIO = {
        "id": "full_game_evaluation",
        "description": "Full gameplay evaluation run.",
        "expected_action": "human_policy",
    }


def get_next_dataset_path(base_path):
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    index = 1

    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


# =========================
# CAR
# =========================

class Car:
    def __init__(self, lane, world_y, speed, color, is_ego=False):
        self.lane = lane
        self.x = LANES[lane]
        self.prev_x = self.x
        self.lateral_velocity = 0.0
        self.target_x = self.x
        self.world_y = world_y
        self.speed = speed
        self.desired_speed = speed
        self.color = color
        self.is_ego = is_ego
        self.blinker = None
        self.blinker_timer = 0
        self.blinker_visible = False
        self.pending_lane = None
        self.pending_direction = None
        self.signal_timer = 0
        self.lane_change_cooldown = 0
        self.movement_direction = 0
        self.behavior_type = "normal"
        self.drift_offset = 0
        self.drift_direction = random.choice([-1, 1])
        self.drift_timer = random.randint(30, 120)
        self.brake_tap_timer = random.randint(20, 90)

    def update_blinker(self):
        if self.blinker:
            self.blinker_timer += 1

            if self.blinker_timer > 20:
                self.blinker_visible = not self.blinker_visible
                self.blinker_timer = 0
        else:
            self.blinker_visible = False
            self.blinker_timer = 0

    def screen_y(self, ego_world_y):
        return EGO_START_SCREEN_Y - (self.world_y - ego_world_y)

    def rect(self, ego_world_y):
        r = pygame.Rect(0, 0, CAR_W, CAR_H)
        r.center = (self.x, self.screen_y(ego_world_y))
        return r

    def update_lane_change(self):
        previous_x = self.x
        lane_change_smoothing = 0.10 if not self.is_ego else 0.20
        self.x += (self.target_x - self.x) * lane_change_smoothing
        self.lateral_velocity = self.x - previous_x
        self.prev_x = previous_x

        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1

        if self.pending_lane is None and abs(self.target_x - self.x) < 2:
            self.blinker = None

        if self.is_ego and abs(self.target_x - self.x) < 1:
            self.x = self.target_x
            self.movement_direction = 0

    def move_forward(self):
        self.world_y += self.speed

    def draw(self, screen, ego_world_y):
        relative_y = self.world_y - ego_world_y

        if not self.is_ego:
            if relative_y > VISIBLE_AHEAD_DISTANCE or relative_y < -VISIBLE_BEHIND_DISTANCE:
                return

        r = self.rect(ego_world_y)

        pygame.draw.rect(screen, self.color, r, border_radius=8)
        pygame.draw.rect(screen, BLACK, r, 3, border_radius=8)

        pygame.draw.rect(screen, (25, 25, 25), (r.x + 10, r.y + 15, CAR_W - 20, 18), border_radius=4)

        if not self.is_ego:
            marker_font = pygame.font.SysFont("Arial", 14, bold=True)
            behavior_marker = {
                "normal": "N",
                "slow": "L",
                "brake_tapper": "B",
                "drifter": "D",
            }.get(self.behavior_type, "N")
            marker = marker_font.render(behavior_marker, True, WHITE)
            screen.blit(marker, (r.centerx - marker.get_width() // 2, r.centery - marker.get_height() // 2))

        pygame.draw.rect(screen, (35, 35, 35), (r.x + 12, r.y + CAR_H - 28, CAR_W - 24, 14), border_radius=4)

        if self.blinker_visible:
            blinker_color = (255, 180, 0)

            if self.blinker == "left":
                pygame.draw.circle(screen, blinker_color, (r.left + 2, r.top + 16), 8)
                pygame.draw.circle(screen, blinker_color, (r.left + 2, r.bottom - 16), 8)

            elif self.blinker == "right":
                pygame.draw.circle(screen, blinker_color, (r.right - 2, r.top + 16), 8)
                pygame.draw.circle(screen, blinker_color, (r.right - 2, r.bottom - 16), 8)


# =========================
# DRAWING
# =========================

def draw_road(road_offset, ego_speed, score, traffic_count, ghost_probability=None):
    SCREEN.fill(BLACK)
    pygame.draw.rect(SCREEN, ROAD, (ROAD_X, 0, ROAD_WIDTH, HEIGHT))

    pygame.draw.line(SCREEN, YELLOW, (ROAD_X, 0), (ROAD_X, HEIGHT), 5)
    pygame.draw.line(SCREEN, YELLOW, (ROAD_X + ROAD_WIDTH, 0), (ROAD_X + ROAD_WIDTH, HEIGHT), 5)

    for i in range(1, LANE_COUNT):
        x = ROAD_X + i * LANE_WIDTH
        for y in range(-120, HEIGHT, 120):
            pygame.draw.line(SCREEN, WHITE, (x, y + road_offset), (x, y + 65 + road_offset), 5)

    pygame.draw.rect(SCREEN, BLACK, (ROAD_X, 0, ROAD_WIDTH, DRIVER_VIEW_TOP_Y))
    pygame.draw.line(SCREEN, RED, (ROAD_X, DRIVER_VIEW_TOP_Y), (ROAD_X + ROAD_WIDTH, DRIVER_VIEW_TOP_Y), 3)

    font = pygame.font.SysFont("Arial", 25, bold=True)

    SCREEN.blit(font.render(f"Score: {score}", True, WHITE), (20, 20))
    SCREEN.blit(font.render(f"Ego speed: {ego_speed:.1f}", True, WHITE), (20, 55))
    SCREEN.blit(font.render(f"Traffic: {traffic_count}", True, WHITE), (20, 90))
    SCREEN.blit(font.render(f"Visible ahead: {VISIBLE_AHEAD_DISTANCE}", True, WHITE), (20, 125))
    if GHOST_MODE:
        SCREEN.blit(font.render("Mode: ghost model debug", True, WHITE), (20, 160))
    elif FULL_GAME_MODE:
        SCREEN.blit(font.render("Mode: full evaluation run", True, WHITE), (20, 160))
    else:
        remaining_seconds = max(0, EPISODE_SECONDS - score // FPS)
        SCREEN.blit(font.render(f"Countdown: {remaining_seconds}s", True, WHITE), (20, 160))
    SCREEN.blit(font.render("CSV labels: switch_lane + lane_direction", True, WHITE), (20, 195))
    SCREEN.blit(font.render(f"Scenario: {CURRENT_SCENARIO['id']}", True, WHITE), (20, 230))

    info_font = pygame.font.SysFont("Arial", 20)
    SCREEN.blit(info_font.render("Controls: ↑ accelerate | ↓ brake | ← → lane change", True, WHITE), (20, HEIGHT - 35))
    legend_font = pygame.font.SysFont("Arial", 18)
    SCREEN.blit(legend_font.render("N normal | L slow | B brake tapper | D drifter", True, WHITE), (20, HEIGHT - 62))

    if GHOST_MODE and ghost_probability is not None:
        decision = "CHANGE" if ghost_probability >= 0.5 else "STAY"
        decision_color = RED if ghost_probability >= 0.5 else GREEN
        SCREEN.blit(font.render(f"Model switch probability: {ghost_probability:.2f}", True, WHITE), (20, 270))
        SCREEN.blit(font.render(f"MODEL: {decision}", True, decision_color), (20, 305))


# =========================
# TRAFFIC LOGIC
# =========================

def can_spawn(lane, spawn_y, traffic, min_gap=SAFE_SPAWN_GAP):
    for car in traffic:
        if car.lane == lane and abs(car.world_y - spawn_y) < min_gap:
            return False
    return True


def create_behavior_car(lane, spawn_y):
    behavior_type = random.choices(
        ["normal", "slow", "brake_tapper", "drifter"],
        weights=[58, 14, 14, 14],
        k=1,
    )[0]

    if behavior_type == "normal":
        speed = random.uniform(10.0, 15.0)
        color = random.choice([BLUE, GREEN])
    elif behavior_type == "slow":
        speed = random.uniform(7.0, 10.0)
        color = GRAY
    elif behavior_type == "brake_tapper":
        speed = random.uniform(9.0, 14.0)
        color = RED
    else:
        speed = random.uniform(8.0, 14.0)
        color = CYAN

    car = Car(lane, spawn_y, speed, color)
    car.behavior_type = behavior_type
    car.desired_speed = speed
    return car


# --- Scenario-based car creation ---
def create_specific_car(lane, relative_y, behavior_type, ego_world_y=0):
    spawn_y = ego_world_y + relative_y

    if behavior_type == "normal":
        speed = random.uniform(12.0, 16.0)
        color = random.choice([BLUE, GREEN])
    elif behavior_type == "slow":
        speed = random.uniform(6.0, 8.5)
        color = GRAY
    elif behavior_type == "brake_tapper":
        speed = random.uniform(9.0, 13.0)
        color = RED
    elif behavior_type == "drifter":
        speed = random.uniform(8.0, 13.0)
        color = CYAN
    else:
        speed = random.uniform(10.0, 14.0)
        color = BLUE

    car = Car(lane, spawn_y, speed, color)
    car.behavior_type = behavior_type
    car.desired_speed = speed
    return car


def spawn_traffic(ego_world_y, traffic):
    for _ in range(10):
        lane = random.randint(0, LANE_COUNT - 1)
        spawn_y = ego_world_y + random.randint(550, 1600)

        if can_spawn(lane, spawn_y, traffic):
            return create_behavior_car(lane, spawn_y)

    return None


def seed_initial_traffic(ego_world_y, traffic, count=8, scenario=None):
    """Start each episode inside a meaningful traffic scenario."""
    scenario = scenario or CURRENT_SCENARIO
    traffic.clear()

    left_lane = 0
    same_lane = 1
    right_lane = 2

    if scenario["id"] == "normal_front_stay":
        traffic.append(create_specific_car(same_lane, 210, "normal", ego_world_y))
        traffic.append(create_specific_car(left_lane, 260, "normal", ego_world_y))
        traffic.append(create_specific_car(right_lane, 280, "normal", ego_world_y))

    elif scenario["id"] == "slow_front_open_lane":
        traffic.append(create_specific_car(same_lane, 180, "slow", ego_world_y))
        traffic.append(create_specific_car(left_lane, 340, "normal", ego_world_y))
        traffic.append(create_specific_car(right_lane, 370, "normal", ego_world_y))

    elif scenario["id"] == "brake_tapper_front":
        traffic.append(create_specific_car(same_lane, 180, "brake_tapper", ego_world_y))
        traffic.append(create_specific_car(left_lane, 340, "normal", ego_world_y))
        traffic.append(create_specific_car(right_lane, 360, "normal", ego_world_y))

    elif scenario["id"] == "drifter_front":
        traffic.append(create_specific_car(same_lane, 190, "drifter", ego_world_y))
        traffic.append(create_specific_car(left_lane, 330, "normal", ego_world_y))
        traffic.append(create_specific_car(right_lane, 350, "normal", ego_world_y))

    elif scenario["id"] == "blocked_adjacent_lane":
        traffic.append(create_specific_car(same_lane, 180, random.choice(["slow", "brake_tapper", "drifter"]), ego_world_y))
        traffic.append(create_specific_car(left_lane, 75, "normal", ego_world_y))
        traffic.append(create_specific_car(right_lane, 90, "normal", ego_world_y))
        traffic.append(create_specific_car(left_lane, 230, "normal", ego_world_y))
        traffic.append(create_specific_car(right_lane, 250, "normal", ego_world_y))

    while len(traffic) < count:
        new_car = spawn_traffic(ego_world_y, traffic)
        if new_car:
            traffic.append(new_car)
        else:
            break


def closest_car_ahead(car, cars):
    closest = None
    closest_gap = float("inf")

    for other in cars:
        if other is car:
            continue

        if other.lane == car.lane and other.world_y > car.world_y:
            gap = other.world_y - car.world_y

            if gap < closest_gap:
                closest = other
                closest_gap = gap

    return closest, closest_gap


def lane_is_safe(car, target_lane, traffic):
    for other in traffic:
        if other is car:
            continue

        if other.lane != target_lane:
            continue

        gap = other.world_y - car.world_y

        if gap >= 0 and gap < LANE_CHANGE_FRONT_GAP:
            return False

        if gap < 0 and abs(gap) < LANE_CHANGE_REAR_GAP:
            return False

    return True


def try_change_lane(car, traffic, urgency=0.015):
    if car.lane_change_cooldown > 0:
        return

    if car.pending_lane is not None:
        return

    if random.random() > urgency:
        return

    possible_lanes = []

    if car.lane > 0:
        possible_lanes.append(("left", car.lane - 1))

    if car.lane < LANE_COUNT - 1:
        possible_lanes.append(("right", car.lane + 1))

    random.shuffle(possible_lanes)

    for direction, target_lane in possible_lanes:
        if lane_is_safe(car, target_lane, traffic):
            car.blinker = direction
            car.pending_direction = direction
            car.pending_lane = target_lane
            car.signal_timer = 0
            return


def apply_reckless_behavior(car, all_cars):
    if car.behavior_type == "brake_tapper":
        car.brake_tap_timer -= 1

        if car.brake_tap_timer <= 0:
            car.speed = max(1.0, car.speed - random.uniform(4.0, 7.0))
            car.brake_tap_timer = random.randint(20, 70)
        else:
            car.speed = min(car.desired_speed, car.speed + 0.12)

    elif car.behavior_type == "drifter":
        car.drift_timer -= 1

        if car.drift_timer <= 0:
            car.drift_direction *= -1
            car.drift_timer = random.randint(25, 80)

        car.drift_offset += car.drift_direction * 0.55
        car.drift_offset = max(-LANE_WIDTH * 0.38, min(car.drift_offset, LANE_WIDTH * 0.38))
        car.target_x = LANES[car.lane] + car.drift_offset

        if abs(car.drift_offset) > LANE_WIDTH * 0.18:
            car.drift_direction *= -1


def update_pending_lane_change(car, traffic):
    if car.pending_lane is None:
        return

    car.signal_timer += 1

    if not lane_is_safe(car, car.pending_lane, traffic):
        car.pending_lane = None
        car.pending_direction = None
        car.signal_timer = 0
        car.blinker = None
        return

    if car.signal_timer >= SIGNAL_FRAMES_BEFORE_LANE_CHANGE:
        car.lane = car.pending_lane
        car.target_x = LANES[car.lane]
        car.pending_lane = None
        car.pending_direction = None
        car.signal_timer = 0
        car.lane_change_cooldown = LANE_CHANGE_COOLDOWN_FRAMES


def update_traffic(traffic, ego):
    all_cars = traffic + [ego]

    for car in traffic:
        car_ahead, gap = closest_car_ahead(car, all_cars)

        car.speed += (car.desired_speed - car.speed) * 0.03

        if car_ahead:
            if gap < HARD_BRAKE_DISTANCE:
                car.speed = max(1.5, car.speed - 0.35)
                try_change_lane(car, all_cars, urgency=0.08)

            elif gap < FOLLOW_DISTANCE:
                car.speed = min(car.speed, car_ahead.speed * 0.95)

                if car.desired_speed > car_ahead.speed + 1.0:
                    try_change_lane(car, all_cars, urgency=0.05)

        else:
            try_change_lane(car, all_cars, urgency=0.006)

        apply_reckless_behavior(car, all_cars)
        update_pending_lane_change(car, all_cars)
        car.update_blinker()
        car.move_forward()
        car.update_lane_change()


def remove_far_cars(traffic, ego_world_y):
    return [
        car for car in traffic
        if ego_world_y - 700 < car.world_y < ego_world_y + 1900
    ]


# =========================
# DATA RECORDING / SEQUENCE PREP
# =========================

ACTION_KEEP = 0
ACTION_ACCELERATE = 1
ACTION_BRAKE = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4

TRANSFORMER_FEATURE_COLUMNS = [
    "ego_lane",
    "ego_x",
    "ego_y",
    "ego_speed",
]

for visible_car_index in range(MAX_VISIBLE_CARS_FEATURES):
    TRANSFORMER_FEATURE_COLUMNS.extend([
        f"visible_car_{visible_car_index}_absolute_x",
        f"visible_car_{visible_car_index}_absolute_y",
        f"visible_car_{visible_car_index}_exist",
        f"visible_car_{visible_car_index}_lane",
        f"visible_car_{visible_car_index}_lateral_velocity",
        f"visible_car_{visible_car_index}_relative_speed",
        f"visible_car_{visible_car_index}_relative_x",
        f"visible_car_{visible_car_index}_relative_y",
        f"visible_car_{visible_car_index}_speed",
    ])


def get_manual_action_from_event(event):
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
            return ACTION_LEFT
        if event.key == pygame.K_RIGHT:
            return ACTION_RIGHT

    return None


def get_manual_action_from_keys():
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        return ACTION_ACCELERATE
    if keys[pygame.K_DOWN]:
        return ACTION_BRAKE

    return ACTION_KEEP


def all_visible_cars_features(ego, traffic):
    visible_cars = []

    for car in traffic:
        relative_y = car.world_y - ego.world_y

        if relative_y > VISIBLE_AHEAD_DISTANCE or relative_y < -VISIBLE_BEHIND_DISTANCE:
            continue

        visible_cars.append((abs(relative_y), car, relative_y))

    visible_cars.sort(key=lambda item: item[0])

    features = {}

    for index in range(MAX_VISIBLE_CARS_FEATURES):
        prefix = f"visible_car_{index}"

        if index < len(visible_cars):
            _, car, relative_y = visible_cars[index]
            features.update({
                f"{prefix}_absolute_x": round(car.x, 3),
                f"{prefix}_absolute_y": round(car.world_y, 3),
                f"{prefix}_exist": 1,
                f"{prefix}_lane": car.lane,
                f"{prefix}_lateral_velocity": round(car.lateral_velocity, 3),
                f"{prefix}_relative_speed": round(car.speed - ego.speed, 3),
                f"{prefix}_relative_x": round(car.x - ego.x, 3),
                f"{prefix}_relative_y": round(relative_y, 3),
                f"{prefix}_speed": round(car.speed, 3),
            })
        else:
            features.update({
                f"{prefix}_absolute_x": NO_CAR_SENTINEL,
                f"{prefix}_absolute_y": NO_CAR_SENTINEL,
                f"{prefix}_exist": 0,
                f"{prefix}_lane": -1,
                f"{prefix}_lateral_velocity": 0.0,
                f"{prefix}_relative_speed": 0.0,
                f"{prefix}_relative_x": NO_CAR_SENTINEL,
                f"{prefix}_relative_y": NO_CAR_SENTINEL,
                f"{prefix}_speed": 0.0,
            })

    return features


def extract_state_features(ego, traffic):
    features = {
        "ego_lane": ego.lane,
        "ego_x": round(ego.x, 3),
        "ego_y": round(ego.world_y, 3),
        "ego_speed": round(ego.speed, 3),
    }

    features.update(all_visible_cars_features(ego, traffic))
    return features


class DataRecorder:
    def __init__(self, path):
        self.path = path
        self.file = None
        self.writer = None
        self.frame_id = 0
        self.fieldnames = [
            "frame_id",
            "scenario_id",
            "scenario_description",
            "scenario_expected_action",
            "action",
            "switch_lane",
            "lane_direction",
            "change_active",
            "movement_state",
            "movement_class",
            *TRANSFORMER_FEATURE_COLUMNS,
        ]

    def start(self):
        self.path = get_next_dataset_path(self.path)
        self.file = self.path.open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        print(f"Recording dataset to {self.path.resolve()}")

    def record(self, ego, traffic, action):
        switch_lane = 1 if action in [ACTION_LEFT, ACTION_RIGHT] else 0

        if action == ACTION_LEFT:
            lane_direction = -1
        elif action == ACTION_RIGHT:
            lane_direction = 1
        else:
            lane_direction = 0

        movement_state = ego.movement_direction
        change_active = 1 if movement_state != 0 else 0

        if movement_state == -1:
            movement_class = 1
        elif movement_state == 1:
            movement_class = 2
        else:
            movement_class = 0

        row = {
            "frame_id": self.frame_id,
            "scenario_id": CURRENT_SCENARIO["id"],
            "scenario_description": CURRENT_SCENARIO["description"],
            "scenario_expected_action": CURRENT_SCENARIO["expected_action"],
            "action": action,
            "switch_lane": switch_lane,
            "lane_direction": lane_direction,
            "change_active": change_active,
            "movement_state": movement_state,
            "movement_class": movement_class,
        }
        row.update(extract_state_features(ego, traffic))

        self.writer.writerow(row)
        self.file.flush()
        self.frame_id += 1

    def stop(self):
        if self.file:
            self.file.close()


def build_transformer_sequences_from_csv(
    csv_path=DATASET_PATH,
    output_path="transformer_sequences.npz",
    seq_len=30,
    feature_columns=None,
    label_column="change_active",
    normalize=True,
):
    import numpy as np
    import pandas as pd

    feature_columns = feature_columns or TRANSFORMER_FEATURE_COLUMNS

    df = pd.read_csv(csv_path)
    missing = [col for col in [*feature_columns, label_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[feature_columns].values.astype("float32")
    y = df[label_column].values.astype("float32")

    X_sequences = []
    y_sequences = []

    for i in range(len(X) - seq_len):
        X_sequences.append(X[i:i + seq_len])
        y_sequences.append(y[i + seq_len])

    X_sequences = np.array(X_sequences, dtype="float32")
    y_sequences = np.array(y_sequences, dtype="float32")

    mean = None
    std = None
    if normalize and len(X_sequences) > 0:
        mean = X_sequences.mean(axis=(0, 1), keepdims=True)
        std = X_sequences.std(axis=(0, 1), keepdims=True) + 1e-6
        X_sequences = (X_sequences - mean) / std

    np.savez(
        output_path,
        X=X_sequences,
        y=y_sequences,
        feature_columns=np.array(feature_columns),
        seq_len=np.array(seq_len),
        mean=mean if mean is not None else np.array([]),
        std=std if std is not None else np.array([]),
    )

    print(f"Saved sequences to {output_path}")
    print(f"X shape: {X_sequences.shape}")
    print(f"y shape: {y_sequences.shape}")
    return X_sequences, y_sequences


# =========================
# EGO LOGIC
# =========================

def handle_ego_input(ego):
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        ego.speed += 0.08

    if keys[pygame.K_DOWN]:
        ego.speed -= 0.15

    ego.speed = max(0.5, min(ego.speed, 15.0))


def handle_lane_change(event, ego, traffic):
    if event.type != pygame.KEYDOWN:
        return

    if event.key == pygame.K_LEFT:
        previous_lane = ego.lane
        ego.lane = max(0, ego.lane - 1)
        ego.target_x = LANES[ego.lane]
        ego.pending_lane = None
        ego.pending_direction = None
        ego.signal_timer = 0
        ego.blinker = None
        ego.movement_direction = -1 if ego.lane != previous_lane else 0

    elif event.key == pygame.K_RIGHT:
        previous_lane = ego.lane
        ego.lane = min(LANE_COUNT - 1, ego.lane + 1)
        ego.target_x = LANES[ego.lane]
        ego.pending_lane = None
        ego.pending_direction = None
        ego.signal_timer = 0
        ego.blinker = None
        ego.movement_direction = 1 if ego.lane != previous_lane else 0


# =========================
# COLLISION
# =========================

def check_collision(ego, traffic):
    ego_rect = ego.rect(ego.world_y)

    for car in traffic:
        if ego_rect.colliderect(car.rect(ego.world_y)):
            return True

    return False


# =========================
# MAIN
# =========================

def main():
    ego = Car(
        lane=1,
        world_y=0,
        speed=6.5,
        color=ORANGE,
        is_ego=True,
    )

    traffic = []
    print(f"Mode: {RUN_MODE}")
    print(f"Scenario: {CURRENT_SCENARIO['id']} - {CURRENT_SCENARIO['description']}")
    if FULL_GAME_MODE or GHOST_MODE:
        seed_initial_traffic(0, traffic, count=12, scenario=CURRENT_SCENARIO)
    else:
        seed_initial_traffic(0, traffic, count=8, scenario=CURRENT_SCENARIO)
    score = 0
    spawn_timer = 0
    road_offset = 0

    recorder = DataRecorder(DATASET_PATH)
    recorder.start()

    ghost_model = None
    ghost_feature_buffer = deque(maxlen=30)
    ghost_probability = None

    if GHOST_MODE:
        from test_model import DrivingModelInference

        ghost_model = DrivingModelInference()
        ghost_feature_buffer = deque(maxlen=ghost_model.seq_len)
        print("Ghost model loaded successfully")

    try:
        while True:
            CLOCK.tick(FPS)
            score += 1
            if not (FULL_GAME_MODE or GHOST_MODE) and score >= EPISODE_FRAME_LIMIT:
                print("Episode complete. Final score:", score)
                pygame.quit()
                sys.exit()
            current_action = ACTION_KEEP

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                event_action = get_manual_action_from_event(event)
                if event_action is not None:
                    current_action = event_action

                handle_lane_change(event, ego, traffic)

            key_action = get_manual_action_from_keys()
            if key_action in [ACTION_ACCELERATE, ACTION_BRAKE]:
                current_action = key_action

            handle_ego_input(ego)
            ego.update_blinker()

            ego.move_forward()
            ego.update_lane_change()

            update_traffic(traffic, ego)

            spawn_timer += 1
            spawn_threshold = 35 if (FULL_GAME_MODE or GHOST_MODE) else 45
            if spawn_timer > spawn_threshold:
                new_car = spawn_traffic(ego.world_y, traffic)
                if new_car:
                    traffic.append(new_car)
                spawn_timer = 0

            traffic = remove_far_cars(traffic, ego.world_y)

            recorder.record(ego, traffic, current_action)

            if GHOST_MODE and ghost_model is not None:
                current_features = extract_state_features(ego, traffic)
                current_feature_vector = [
                    current_features[col]
                    for col in ghost_model.feature_columns
                ]
                ghost_feature_buffer.append(current_feature_vector)

                if len(ghost_feature_buffer) == ghost_model.seq_len:
                    sequence = np.array(ghost_feature_buffer, dtype=np.float32)
                    ghost_probability = ghost_model.predict_probability(sequence)

            road_offset = (road_offset + ego.speed) % 120

            if check_collision(ego, traffic):
                print("CRASH! Final score:", score)
                pygame.quit()
                sys.exit()

            draw_road(road_offset, ego.speed, score, len(traffic), ghost_probability)

            for car in traffic:
                car.draw(SCREEN, ego.world_y)

            ego.draw(SCREEN, ego.world_y)

            pygame.display.flip()
    finally:
        recorder.stop()


if __name__ == "__main__":
    main()

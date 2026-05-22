import pygame
import random
import sys

pygame.init()

# =========================
# CONFIG
# =========================

WIDTH, HEIGHT = 900, 700
FPS = 60

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

SAFE_SPAWN_GAP = 320
FOLLOW_DISTANCE = 180
HARD_BRAKE_DISTANCE = 120

LANE_CHANGE_REAR_GAP = 170
LANE_CHANGE_FRONT_GAP = 230
SIGNAL_FRAMES_BEFORE_LANE_CHANGE = 45
LANE_CHANGE_COOLDOWN_FRAMES = 90


# =========================
# CAR
# =========================

class Car:
    def __init__(self, lane, world_y, speed, color, is_ego=False):
        self.lane = lane
        self.x = LANES[lane]
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
        self.behavior_type = "normal"
        self.drift_offset = 0
        self.drift_direction = random.choice([-1, 1])
        self.drift_timer = random.randint(30, 120)
        self.speed_oscillation_timer = random.randint(20, 120)

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
        lane_change_smoothing = 0.10 if not self.is_ego else 0.20
        self.x += (self.target_x - self.x) * lane_change_smoothing

        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1

        # Only turn the blinker off after the physical lane change has finished.
        if self.pending_lane is None and abs(self.target_x - self.x) < 2:
            self.blinker = None

    def move_forward(self):
        self.world_y += self.speed

    def draw(self, screen, ego_world_y):
        r = self.rect(ego_world_y)

        pygame.draw.rect(screen, self.color, r, border_radius=8)
        pygame.draw.rect(screen, BLACK, r, 3, border_radius=8)

        # windshield
        pygame.draw.rect(
            screen,
            (25, 25, 25),
            (r.x + 10, r.y + 15, CAR_W - 20, 18),
            border_radius=4,
        )

        # behavior marker for non-ego cars
        if not self.is_ego:
            marker_font = pygame.font.SysFont("Arial", 14, bold=True)
            behavior_marker = {
                "normal": "N",
                "tailgater": "T",
                "aggressive": "A",
                "drifter": "D",
                "speed_oscillator": "S",
            }.get(self.behavior_type, "N")
            marker = marker_font.render(behavior_marker, True, WHITE)
            screen.blit(marker, (r.centerx - marker.get_width() // 2, r.centery - marker.get_height() // 2))

        # rear window
        pygame.draw.rect(
            screen,
            (35, 35, 35),
            (r.x + 12, r.y + CAR_H - 28, CAR_W - 24, 14),
            border_radius=4,
        )

        # blinkers
        if self.blinker_visible:
            blinker_color = (255, 180, 0)

            # left blinker
            if self.blinker == "left":
                pygame.draw.circle(screen, blinker_color, (r.left + 2, r.top + 16), 8)
                pygame.draw.circle(screen, blinker_color, (r.left + 2, r.bottom - 16), 8)

            # right blinker
            elif self.blinker == "right":
                pygame.draw.circle(screen, blinker_color, (r.right - 2, r.top + 16), 8)
                pygame.draw.circle(screen, blinker_color, (r.right - 2, r.bottom - 16), 8)

# =========================
# DRAWING
# =========================

def draw_road(road_offset, ego_speed, score, traffic_count):
    SCREEN.fill(BLACK)

    pygame.draw.rect(SCREEN, ROAD, (ROAD_X, 0, ROAD_WIDTH, HEIGHT))

    # borders
    pygame.draw.line(SCREEN, YELLOW, (ROAD_X, 0), (ROAD_X, HEIGHT), 5)
    pygame.draw.line(SCREEN, YELLOW, (ROAD_X + ROAD_WIDTH, 0), (ROAD_X + ROAD_WIDTH, HEIGHT), 5)

    # dashed lane lines
    for i in range(1, LANE_COUNT):
        x = ROAD_X + i * LANE_WIDTH
        for y in range(-120, HEIGHT, 120):
            pygame.draw.line(SCREEN, WHITE, (x, y + road_offset), (x, y + 65 + road_offset), 5)

    font = pygame.font.SysFont("Arial", 25, bold=True)

    SCREEN.blit(font.render(f"Score: {score}", True, WHITE), (20, 20))
    SCREEN.blit(font.render(f"Ego speed: {ego_speed:.1f}", True, WHITE), (20, 55))
    SCREEN.blit(font.render(f"Traffic: {traffic_count}", True, WHITE), (20, 90))

    info_font = pygame.font.SysFont("Arial", 20)
    SCREEN.blit(info_font.render("Controls: ↑ accelerate | ↓ brake | ← → lane change", True, WHITE), (20, HEIGHT - 35))
    legend_font = pygame.font.SysFont("Arial", 18)
    SCREEN.blit(legend_font.render("N normal | T tailgater | A aggressive | D drifter | S speed oscillator", True, WHITE), (20, HEIGHT - 62))


# =========================
# TRAFFIC LOGIC
# =========================

def can_spawn(lane, spawn_y, traffic):
    for car in traffic:
        if car.lane == lane and abs(car.world_y - spawn_y) < SAFE_SPAWN_GAP:
            return False
    return True


def spawn_traffic(ego_world_y, traffic):
    for _ in range(10):
        lane = random.randint(0, LANE_COUNT - 1)
        spawn_y = ego_world_y + random.randint(750, 1700)

        if can_spawn(lane, spawn_y, traffic):
            behavior_type = random.choices(
                ["normal", "tailgater", "aggressive", "drifter", "speed_oscillator"],
                weights=[60, 12, 12, 8, 8],
                k=1,
            )[0]

            if behavior_type == "normal":
                speed = random.uniform(4.0, 8.5)
                color = random.choice([BLUE, GRAY, GREEN])
            elif behavior_type == "tailgater":
                speed = random.uniform(7.0, 10.5)
                color = RED
            elif behavior_type == "aggressive":
                speed = random.uniform(7.5, 11.0)
                color = PURPLE
            elif behavior_type == "drifter":
                speed = random.uniform(4.5, 8.0)
                color = CYAN
            else:
                speed = random.uniform(5.5, 9.5)
                color = PINK

            car = Car(lane, spawn_y, speed, color)
            car.behavior_type = behavior_type
            car.desired_speed = speed
            return car

    return None


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

        # Car in front in target lane.
        if gap >= 0 and gap < LANE_CHANGE_FRONT_GAP:
            return False

        # Car behind in target lane.
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
            # Signal first. Do not move yet.
            car.blinker = direction
            car.pending_direction = direction
            car.pending_lane = target_lane
            car.signal_timer = 0
            return

def apply_reckless_behavior(car, all_cars):
    # Reckless does not mean suicidal. These behaviors add uncertainty,
    # but the normal safety checks still prevent most crashes.

    if car.behavior_type == "tailgater":
        car_ahead, gap = closest_car_ahead(car, all_cars)

        if car_ahead and gap < FOLLOW_DISTANCE * 1.2:
            # Tailgaters tolerate smaller gaps and react late.
            car.speed = min(car.speed + 0.04, car.desired_speed + 1.2)

            if gap < HARD_BRAKE_DISTANCE * 0.75:
                car.speed = max(1.5, car.speed - 0.55)

    elif car.behavior_type == "aggressive":
        # Aggressive drivers try to overtake more often.
        try_change_lane(car, all_cars, urgency=0.035)
        car.speed = min(car.speed + 0.03, car.desired_speed + 1.5)

    elif car.behavior_type == "drifter":
        # Drifters wander slightly inside their lane, then correct back.
        car.drift_timer -= 1

        if car.drift_timer <= 0:
            car.drift_direction *= -1
            car.drift_timer = random.randint(25, 80)

        car.drift_offset += car.drift_direction * 0.55
        car.drift_offset = max(-LANE_WIDTH * 0.38, min(car.drift_offset, LANE_WIDTH * 0.38))

        # Keep physical lane target, but visually/behaviorally wander around lane center.
        car.target_x = LANES[car.lane] + car.drift_offset

        # If drifting too much, correct more strongly.
        if abs(car.drift_offset) > LANE_WIDTH * 0.18:
            car.drift_direction *= -1

    elif car.behavior_type == "speed_oscillator":
        # Speed oscillators surge and slow down unpredictably.
        car.speed_oscillation_timer -= 1

        if car.speed_oscillation_timer <= 0:
            car.desired_speed += random.uniform(-2.2, 2.2)
            car.desired_speed = max(3.5, min(car.desired_speed, 11.5))
            car.speed_oscillation_timer = random.randint(35, 110)

def update_pending_lane_change(car, traffic):
    if car.pending_lane is None:
        return

    car.signal_timer += 1

    # Keep checking while signaling. If the lane becomes unsafe, cancel the lane change.
    if not lane_is_safe(car, car.pending_lane, traffic):
        car.pending_lane = None
        car.pending_direction = None
        car.signal_timer = 0
        car.blinker = None
        return

    # After signaling long enough, commit to the lane change.
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

        # Recover toward desired speed.
        car.speed += (car.desired_speed - car.speed) * 0.03

        if car_ahead:
            if gap < HARD_BRAKE_DISTANCE:
                car.speed = max(1.5, car.speed - 0.35)

                # If blocked hard, signal and attempt a lane change more urgently.
                try_change_lane(car, all_cars, urgency=0.08)

            elif gap < FOLLOW_DISTANCE:
                car.speed = min(car.speed, car_ahead.speed * 0.95)

                # If this car wants to go faster, signal and attempt to overtake.
                if car.desired_speed > car_ahead.speed + 1.0:
                    try_change_lane(car, all_cars, urgency=0.05)

        else:
            # Occasional natural lane changes, but still signal first and check safety.
            try_change_lane(car, all_cars, urgency=0.006)

        apply_reckless_behavior(car, all_cars)
        update_pending_lane_change(car, all_cars)
        car.update_blinker()
        car.move_forward()
        car.update_lane_change()

def remove_far_cars(traffic, ego_world_y):
    return [
        car for car in traffic
        if ego_world_y - 500 < car.world_y < ego_world_y + 1500
    ]


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

    target_lane = None

    if event.key == pygame.K_LEFT and ego.lane > 0:
        target_lane = ego.lane - 1

    if event.key == pygame.K_RIGHT and ego.lane < LANE_COUNT - 1:
        target_lane = ego.lane + 1

    if target_lane is None:
        return

    # Even with manual control, the ego car should not be allowed to change
    # into an occupied lane. This keeps the environment useful for future AI training.
    if not lane_is_safe(ego, target_lane, traffic):
        return

    ego.lane = target_lane
    ego.target_x = LANES[ego.lane]
    ego.pending_lane = None
    ego.pending_direction = None
    ego.signal_timer = 0
    ego.blinker = None


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
    score = 0
    spawn_timer = 0
    road_offset = 0

    while True:
        CLOCK.tick(FPS)
        score += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            handle_lane_change(event, ego, traffic)

        handle_ego_input(ego)
        ego.update_blinker()

        ego.move_forward()
        ego.update_lane_change()

        update_traffic(traffic, ego)

        spawn_timer += 1
        if spawn_timer > 75:
            new_car = spawn_traffic(ego.world_y, traffic)
            if new_car:
                traffic.append(new_car)
            spawn_timer = 0

        traffic = remove_far_cars(traffic, ego.world_y)

        road_offset = (road_offset + ego.speed) % 120

        if check_collision(ego, traffic):
            print("CRASH! Final score:", score)
            pygame.quit()
            sys.exit()

        draw_road(road_offset, ego.speed, score, len(traffic))

        for car in traffic:
            car.draw(SCREEN, ego.world_y)

        ego.draw(SCREEN, ego.world_y)

        pygame.display.flip()


if __name__ == "__main__":
    main()
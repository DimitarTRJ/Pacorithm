# my_team.py
# Eutopia Pacman Contest agent.
#
# Two weighted-feature reflex agents that can switch between 'offense' and
# 'defense' depending on the score and the time left.
#
#  Offense : dead-end avoidance, ghost-aware route home, food near our border
#            first, power capsules, banks food before it gets lost.
#  Defense : tracks invaders (seen, or inferred from food that disappeared)
#            and patrols near the border when nothing is visible.
#  Strategy: both agents attack early. The more we lead, the more defensive
#            we become (see the LEAD_* constants below).

import random

try:
    # Same import style as baseline_team.py (running from src/contest)
    import util
    from capture_agents import CaptureAgent
    from game import Directions
except ImportError:
    # Package-style imports, in case your repo template uses them
    import contest.util as util
    from contest.capture_agents import CaptureAgent
    from contest.game import Directions


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    """Called by the framework to build your team. Returns two agents."""
    return [eval(first)(first_index), eval(second)(second_index)]


###########
# Helpers #
###########

# Information shared by our two agents: team colour -> last known invader
TRACK = {}

# Set DEBUG = True to print each decision of the listed agents (for testing
# only, and switch it off before submitting). Indices: red 0 and 2, blue 1 and 3.
DEBUG = False
DEBUG_AGENTS = (0, 2)


def compute_dead_ends(walls):
    """
    Returns the set of squares that lie in dead-end corridors.
    Repeatedly peels off squares that have at most one open neighbour.
    """
    cells = set((x, y) for x in range(walls.width) for y in range(walls.height)
                if not walls[x][y])

    def open_neighbours(c, alive):
        x, y = c
        return sum(1 for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                   if n in alive)

    alive = set(cells)
    dead = set()
    changed = True
    while changed:
        changed = False
        for c in list(alive):
            if open_neighbours(c, alive) <= 1:
                alive.discard(c)
                dead.add(c)
                changed = True
    return dead


OFFENSE_WEIGHTS = {
    'food_left': 100,
    'distance_to_food': -1,
    'distance_to_home': -3,
    'distance_to_capsule': -3,
    'banked': 500,
    'ghost_near': -30,
    'ghost_adjacent': -300,
    'dead_end': -60,
    'died': -3000,
    'stop': -100,
    'reverse': -2,
}

DEFENSE_WEIGHTS = {
    'num_invaders': -1000,
    'on_defense': 100,
    'invader_distance': -10,
    'scared_invader_distance': 5,
    'target_distance': -6,
    'patrol_distance': -2,
    'died': -3000,
    'stop': -100,
    'reverse': -2,
}


##########
# Agents #
##########

class BaseAgent(CaptureAgent):
    """Shared logic. Subclasses only decide the default role."""

    # --- offense tuning ---
    CARRY_LIMIT = 5               # head home once carrying this much food
    GHOST_RANGE = 4               # react to ghosts closer than this
    HOME_BIAS = 0.5               # prefer food that is close to our border

    # --- strategy tuning (the score is from OUR point of view) ---
    LEAD_SAFE = 3                 # 2nd agent stays home once we lead by this much
    LEAD_DEFEND_ALL = 6           # BOTH agents defend once we lead by this much...
    LEAD_DEFEND_ALL_LATE = 3      # ...or by this much in the late game
    LATE_MOVES = 120              # "late" = this few moves left per agent
    TRAILING_ATTACK = 2           # 2nd agent goes all-out if trailing by this
                                  # much in the late game

    # --- stalemate breaking ---
    # If the score has not changed for STALL_START moves (and we are not
    # leading), ghost fear is scaled down over the next STALL_RAMP moves,
    # down to MIN_CAUTION of its normal strength.
    STALL_START = 60
    STALL_RAMP = 40
    MIN_CAUTION = 0.0

    # --- spreading out ---
    ZONE_PENALTY = 6              # extra "distance" for food outside our half

    # --- defense tuning ---
    TRACK_FRESH = 12              # how long (in moves) an invader hint stays valid

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.role = 'offense'
        self.protect_lead = False
        self.prev_def_food = None
        self.food_home_cache = {}
        self.last_score = 0
        self.last_change = self.moves_left(game_state)
        self.caution = 1.0
        self.home_goal = None
        self.returning = False
        CaptureAgent.register_initial_state(self, game_state)
        TRACK.pop(self.red, None)

        walls = game_state.get_walls()
        # Squares on our side of the border: reaching one banks the food
        border_x = walls.width // 2 - 1 if self.red else walls.width // 2
        self.home_border = [(border_x, y) for y in range(walls.height)
                            if not walls[border_x][y]]
        self.dead_ends = compute_dead_ends(walls)

        # Spread out: one agent prefers the top half, the other the bottom half
        self.map_height = walls.height
        team = sorted(self.get_team(game_state))
        self.zone_upper = (self.index == team[0])

    # ---------- helpers ----------

    def distance_to_home(self, pos):
        """Maze distance from pos to the nearest square on our side."""
        return min(self.get_maze_distance(pos, b) for b in self.home_border)

    def update_returning(self, game_state):
        """
        Decides, once per turn and from the CURRENT state, whether to run home.
        Latched: once we decide to go home we stay committed until the food is
        banked (or lost), so a ghost drifting in and out of view cannot make
        us flip between 'go home' and 'find food'.
        """
        me = game_state.get_agent_state(self.index)
        if me.num_carrying == 0:
            self.returning = False
            return
        if self.returning:
            return

        my_pos = game_state.get_agent_position(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [util.nearest_point(e.get_position()) for e in enemies
                  if not e.is_pacman and e.get_position() is not None and e.scared_timer <= 1]
        threat = (me.is_pacman and ghosts
                  and min(self.get_maze_distance(my_pos, g) for g in ghosts)
                  <= self.GHOST_RANGE + 2)

        food_left = len(self.get_food(game_state).as_list())
        time_is_short = self.moves_left(game_state) <= self.distance_to_home(my_pos) + 8

        if (me.num_carrying >= self.CARRY_LIMIT   # carrying enough
                or threat                         # a ghost is close
                or food_left <= 2                 # game is nearly won
                or time_is_short                  # not enough time left
                or self.protect_lead):            # we lead: secure the points
            self.returning = True

    def pick_home_goal(self, game_state):
        """
        Chooses ONE border square to run to, once per decision. Squares that a
        ghost can reach before us get a smooth penalty, and the previous goal
        gets a small bonus so the target does not flip back and forth.
        """
        my_pos = game_state.get_agent_position(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [util.nearest_point(e.get_position()) for e in enemies
                  if not e.is_pacman and e.get_position() is not None and e.scared_timer <= 1]

        best, best_cost = None, None
        for b in self.home_border:
            d = self.get_maze_distance(my_pos, b)
            cost = d
            for g in ghosts:
                margin = self.get_maze_distance(g, b) - d   # how much sooner we get there
                cost += 2 * max(0, 3 - margin)
            if b == self.home_goal:
                cost -= 3                                   # stickiness
            if best is None or cost < best_cost:
                best, best_cost = b, cost
        return best

    def food_home_distance(self, food):
        d = self.food_home_cache.get(food)
        if d is None:
            d = self.distance_to_home(food)
            self.food_home_cache[food] = d
        return d

    def enemy_capsules(self, game_state):
        """Power capsules on the enemy side (the ones we can eat)."""
        border_x = self.home_border[0][0]
        if self.red:
            return [c for c in self.get_capsules(game_state) if c[0] > border_x]
        return [c for c in self.get_capsules(game_state) if c[0] < border_x]

    def chased_to_capsule(self, successor, my_pos, ghost_dist):
        """True only if a ghost is very close AND a capsule is within easy reach."""
        if ghost_dist is None or ghost_dist > 3:
            return False
        capsules = self.enemy_capsules(successor)
        if not capsules:
            return False
        return min(self.get_maze_distance(my_pos, c) for c in capsules) <= 8

    def in_my_zone(self, pos):
        """Each attacker prefers one half (top/bottom) of the enemy side."""
        upper = pos[1] >= self.map_height / 2.0
        return upper == self.zone_upper

    def moves_left(self, game_state):
        """Moves this agent still has (timeleft counts moves of all 4 agents)."""
        return getattr(game_state.data, 'timeleft', 1200) // 4

    def visible_invaders(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return [e for e in enemies if e.is_pacman and e.get_position() is not None]

    def would_die(self, game_state, successor):
        """True if this move gets us eaten (we respawn at our start square)."""
        cur = game_state.get_agent_position(self.index)
        new = successor.get_agent_position(self.index)
        return (new == self.start and cur != self.start
                and self.get_maze_distance(cur, new) > 1)

    # ---------- invader tracking (shared by both agents) ----------

    def update_tracking(self, game_state):
        now = self.moves_left(game_state)
        my_pos = game_state.get_agent_position(self.index)

        invaders = self.visible_invaders(game_state)
        if invaders:
            positions = [util.nearest_point(e.get_position()) for e in invaders]
            TRACK[self.red] = {'pos': min(positions,
                                          key=lambda p: self.get_maze_distance(my_pos, p)),
                               'time': now}

        # Food that disappeared since our last turn was eaten by an invader
        food = set(self.get_food_you_are_defending(game_state).as_list())
        if self.prev_def_food is not None and not invaders:
            eaten = self.prev_def_food - food
            if eaten:
                TRACK[self.red] = {'pos': min(eaten,
                                              key=lambda p: self.get_maze_distance(my_pos, p)),
                                   'time': now}
        self.prev_def_food = food

    def tracked_target(self, game_state):
        t = TRACK.get(self.red)
        if t is None or t['time'] - self.moves_left(game_state) > self.TRACK_FRESH:
            return None
        return t['pos']

    def patrol_target(self, game_state):
        """Cycle between the defended food items closest to our border."""
        food = self.get_food_you_are_defending(game_state).as_list()
        if not food:
            return None
        # Each defender covers its own half (top/bottom) so they don't stack
        mine = [f for f in food if self.in_my_zone(f)]
        pool = mine if mine else food
        pool.sort(key=lambda f: self.distance_to_home(f))
        points = pool[:3]
        return points[(self.moves_left(game_state) // 8) % len(points)]

    # ---------- decision ----------

    def choose_action(self, game_state):
        lead = self.get_score(game_state)
        late = self.moves_left(game_state) <= self.LATE_MOVES
        needed = self.LEAD_DEFEND_ALL_LATE if late else self.LEAD_DEFEND_ALL
        self.protect_lead = lead >= needed

        # Stalemate breaker: the longer nothing is scored, the less we fear ghosts
        now = self.moves_left(game_state)
        if lead != self.last_score:
            self.last_score = lead
            self.last_change = now
        stall = self.last_change - now
        self.caution = 1.0
        if lead <= 0 and stall > self.STALL_START:
            self.caution = max(self.MIN_CAUTION,
                               1.0 - (stall - self.STALL_START) / float(self.STALL_RAMP))

        self.update_tracking(game_state)
        self.role = self.choose_role(game_state)

        # Decide once per turn whether to run home (see update_returning)
        self.update_returning(game_state)

        # One fixed home target per decision (forgotten once the food is banked)
        if game_state.get_agent_state(self.index).num_carrying > 0:
            self.home_goal = self.pick_home_goal(game_state)
        else:
            self.home_goal = None

        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]

        if DEBUG and self.index in DEBUG_AGENTS:
            print('agent %d pos=%s role=%s returning=%s goal=%s caution=%.2f %s' % (
                self.index, game_state.get_agent_position(self.index), self.role,
                self.returning, self.home_goal, self.caution,
                [(a, round(v, 1)) for a, v in zip(actions, values)]))
        best = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == best]
        return random.choice(best_actions)

    def choose_role(self, game_state):
        return 'offense'

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != util.nearest_point(pos):
            # Only half a grid step was taken, so apply the action again
            return successor.generate_successor(self.index, action)
        return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return sum(features[f] * weights[f] for f in features)

    def get_features(self, game_state, action):
        if self.role == 'defense':
            return self.defensive_features(game_state, action)
        return self.offensive_features(game_state, action)

    def get_weights(self, game_state, action):
        return DEFENSE_WEIGHTS if self.role == 'defense' else OFFENSE_WEIGHTS

    # ---------- offense ----------

    def offensive_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        died = self.would_die(game_state, successor)
        if died:
            features['died'] = 1

        food_list = self.get_food(successor).as_list()
        features['food_left'] = -len(food_list)

        # Visible enemy ghosts that are dangerous (not scared)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [e for e in enemies
                  if not e.is_pacman and e.get_position() is not None and e.scared_timer <= 1]
        ghost_positions = [util.nearest_point(g.get_position()) for g in ghosts]

        ghost_dist = None
        if my_state.is_pacman and ghost_positions:
            ghost_dist = min(self.get_maze_distance(my_pos, g) for g in ghost_positions)
            # All three are scaled by self.caution (1.0 normally, lower when the
            # game is stuck). Being eaten ('died') is never scaled down.
            if ghost_dist < self.GHOST_RANGE:
                features['ghost_near'] = (self.GHOST_RANGE - ghost_dist) * self.caution
            if ghost_dist <= 1:
                features['ghost_adjacent'] = self.caution
            if my_pos in self.dead_ends and ghost_dist <= self.GHOST_RANGE + 3:
                features['dead_end'] = self.caution   # a CLOSE ghost + dead end = trap

        # Decided ONCE per turn in update_returning(), from the current state.
        # (If it were decided per candidate move, the move that crosses the
        # border would be judged differently from the others and look worse.)
        go_home = self.returning

        if go_home:
            goal = self.home_goal
            features['distance_to_home'] = (self.get_maze_distance(my_pos, goal)
                                            if goal is not None
                                            else self.distance_to_home(my_pos))
            if not my_state.is_pacman and not died:
                features['banked'] = 1  # this move crosses the border and banks the food
        elif self.chased_to_capsule(successor, my_pos, ghost_dist):
            # A ghost is right behind us and a capsule is close: grab it
            features['distance_to_capsule'] = min(
                self.get_maze_distance(my_pos, c) for c in self.enemy_capsules(successor))
        elif food_list:
            # Prefer food that is close to us, close to our border, and in OUR
            # half of the map (the two attackers then use different entrances)
            features['distance_to_food'] = min(
                self.get_maze_distance(my_pos, f)
                + self.HOME_BIAS * self.food_home_distance(f)
                + (0 if self.in_my_zone(f) else self.ZONE_PENALTY)
                for f in food_list)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        return features

    # ---------- defense ----------

    def defensive_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        if self.would_die(game_state, successor):
            features['died'] = 1

        # Stay on defense (be a ghost, not a Pacman)
        features['on_defense'] = 0 if my_state.is_pacman else 1

        invaders = self.visible_invaders(successor)
        features['num_invaders'] = len(invaders)
        scared = game_state.get_agent_state(self.index).scared_timer > 0

        if invaders:
            d = min(self.get_maze_distance(my_pos, util.nearest_point(e.get_position()))
                    for e in invaders)
            if scared:
                features['scared_invader_distance'] = min(d, 4)   # keep away while scared
            else:
                features['invader_distance'] = d
        else:
            target = self.tracked_target(game_state)
            if target is not None:
                features['target_distance'] = self.get_maze_distance(my_pos, target)
            else:
                patrol = self.patrol_target(game_state)
                if patrol is not None:
                    features['patrol_distance'] = self.get_maze_distance(my_pos, patrol)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        return features


class OffensiveAgent(BaseAgent):
    """Attacks. Once we lead enough (and carry nothing), it helps defend."""

    def choose_role(self, game_state):
        carrying = game_state.get_agent_state(self.index).num_carrying
        # Bank any carried food first, then defend if we lead enough
        if carrying == 0 and self.protect_lead:
            return 'defense'
        return 'offense'


class DefensiveAgent(BaseAgent):
    """
    Attacks early, but drops back when it sees (or infers) an invader, and
    stays home for good once we lead by LEAD_SAFE.
    """

    def choose_role(self, game_state):
        lead = self.get_score(game_state)
        late = self.moves_left(game_state) <= self.LATE_MOVES

        if self.visible_invaders(game_state):
            return 'defense'
        if late and lead <= -self.TRAILING_ATTACK:
            return 'offense'                 # all-out attack when trailing late
        if self.tracked_target(game_state) is not None:
            return 'defense'                 # recent sign of an invader
        if lead >= self.LEAD_SAFE:
            return 'defense'
        return 'offense'
# =============================================================================
# MODULE : AAC / PROJET : EXPLORATION MULTI-ROBOTS 3D (DQN + DIJKSTRA)
# =============================================================================
# VERSION 3D (VOXELS)
# =============================================================================

from flask import Flask, render_template, jsonify, request, redirect, url_for
import numpy as np
import heapq
import random
import math

app = Flask(__name__)

# --- CONFIGURATION INITIALE ---
# Note : Pour la 3D, on réduit la taille par défaut car le volume (N^3) grandit vite.
CONFIG = {
    'GRID_SIZE': 15,    # Taille du cube (15x15x15 voxels)
    'NUM_ROBOTS': 4,    # Nombre de robots
    'DENSITY': 0.15,    # Densité d'obstacles (15%)
    'BATTERY': 2000     # Batterie augmentée pour la 3D
}

# La station est au centre du cube (x, y, z)
CHARGING_STATION = np.array([0, 0, 0])

# --- CODES NUMÉRIQUES ---
UNKNOWN = 0   
FREE = 1      
OBSTACLE = 2  

# =============================================================================
# CLASSE : ENVIRONMENT (Le Monde 3D)
# =============================================================================
class Environment:
    def __init__(self, size, density):
        self.size = size
        
        # Grilles 3D (x, y, z)
        self.true_grid = np.zeros((size, size, size), dtype=int)
        self.knowledge_grid = np.zeros((size, size, size), dtype=int)
        
        self.visited_cells = set()
        self.redundancy_score = 0
        self.newly_discovered = [] 
        self.base_distances = {}
        self.unreachable_blacklist = set()
        self.map_changed_this_step = True
        
        # Base au centre du cube
        global CHARGING_STATION
        mid = size // 2
        CHARGING_STATION = np.array([mid, mid, mid])
        self.visited_cells.add(tuple(CHARGING_STATION))
        
        self.generate_obstacles(density)

    def generate_obstacles(self, density):
        """Génère une boite fermée et des obstacles 3D (Voxels)."""
        # 1. Murs d'enceinte (Cube fermé)
        s = self.size
        # Murs X (Gauche/Droite)
        self.true_grid[0, :, :] = OBSTACLE
        self.true_grid[s-1, :, :] = OBSTACLE
        # Murs Y (Devant/Derrière)
        self.true_grid[:, 0, :] = OBSTACLE
        self.true_grid[:, s-1, :] = OBSTACLE
        # Murs Z (Sol/Plafond)
        self.true_grid[:, :, 0] = OBSTACLE
        self.true_grid[:, :, s-1] = OBSTACLE

        # 2. Obstacles aléatoires 3D
        # On calcule le volume intérieur
        volume = (s-2)**3
        num_obstacles = int(volume * density)
        
        for _ in range(num_obstacles):
            x, y, z = np.random.randint(1, s-1, 3)
            self.true_grid[x, y, z] = OBSTACLE
            
        # 3. Zone de sécurité (Sphère 3D autour de la base)
        cx, cy, cz = CHARGING_STATION
        r_safe = 2
        for x in range(cx-r_safe, cx+r_safe+1):
            for y in range(cy-r_safe, cy+r_safe+1):
                for z in range(cz-r_safe, cz+r_safe+1):
                    if 0 <= x < s and 0 <= y < s and 0 <= z < s:
                        self.true_grid[x, y, z] = FREE

    def get_exploration_rate(self):
        floor = self.true_grid
        accessible = np.count_nonzero(floor != OBSTACLE)
        if accessible == 0: return 0
        
        known = 0
        # Comparaison vectorisée pour rapidité
        # Compte cases qui ne sont pas des murs et qui sont connues
        mask = (self.true_grid != OBSTACLE) & (self.knowledge_grid != UNKNOWN)
        known = np.count_nonzero(mask)
        
        return (known / accessible) * 100

    def record_visit(self, pos):
        if np.array_equal(pos, CHARGING_STATION): return
        t_pos = tuple(pos)
        if t_pos in self.visited_cells:
            self.redundancy_score += 1
        else:
            self.visited_cells.add(t_pos)

    def update_base_distances(self):
        from collections import deque
        start = (int(CHARGING_STATION[0]), int(CHARGING_STATION[1]), int(CHARGING_STATION[2]))
        queue = deque([(start[0], start[1], start[2])])
        self.base_distances = {start: 0}
        
        while queue:
            cx, cy, cz = queue.popleft()
            cost = self.base_distances[(cx, cy, cz)]
            neighbors = [
                (cx+1, cy, cz), (cx-1, cy, cz),
                (cx, cy+1, cz), (cx, cy-1, cz),
                (cx, cy, cz+1), (cx, cy, cz-1)
            ]
            for nx, ny, nz in neighbors:
                if 0 <= nx < self.size and 0 <= ny < self.size and 0 <= nz < self.size:
                    if self.knowledge_grid[nx, ny, nz] != OBSTACLE:
                        if (nx, ny, nz) not in self.base_distances:
                            self.base_distances[(nx, ny, nz)] = cost + 1
                            queue.append((nx, ny, nz))

# =============================================================================
# CLASSE : BRAIN (Deep Q-Network 3D)
# =============================================================================
class Brain:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = 32 # Augmenté pour la complexité 3D
        self.learning_rate = 0.01
        
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def predict(self, state):
        state = np.array(state).reshape(1, -1)
        self.z1 = np.dot(state, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2[0]

    def train(self, state, action, target_q):
        state = np.array(state).reshape(1, -1)
        
        z1 = np.dot(state, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        
        current_q = z2
        loss = current_q[0][action] - target_q
        
        d_z2 = np.zeros_like(z2)
        d_z2[0][action] = loss
        
        d_W2 = np.dot(a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * (z1 > 0)
        
        d_W1 = np.dot(state.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        
        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2

# =============================================================================
# CLASSE : ROBOT (Agent 3D)
# =============================================================================
class Robot:
    def __init__(self, id, start_x, start_y, start_z, env):
        self.id = id
        self.pos = np.array([int(start_x), int(start_y), int(start_z)], dtype=int)
        self.env = env
        self.target = None
        self.battery = CONFIG['BATTERY']
        self.state = 'EXPLORING'
        self.next_pos_intention = None
        self.current_mode = "IDLE"
        self.last_collision = False 
        
        # 6 Actions possibles en 3D : Haut, Bas, Gauche, Droite, Avant, Arrière
        # Input state: 6 voisins + 3 coords relatives + 1 batterie = 10
        self.brain = Brain(10, 6)
        
        self.epsilon = 0.1
        self.gamma = 0.9
        self.last_state = None
        self.last_action = None
        
        # Définition des 6 vecteurs de mouvement (x, y, z)
        self.actions = [
            (1, 0, 0), (-1, 0, 0),  # X axis
            (0, 1, 0), (0, -1, 0),  # Y axis
            (0, 0, 1), (0, 0, -1)   # Z axis (Haut/Bas)
        ]

    def get_local_state(self, robots):
        """Perception locale 3D (6 voisins)."""
        x, y, z = self.pos
        peer_positions = set(tuple(r.pos) for r in robots if r.id != self.id and r.state != 'DEAD')
        state = []
        for dx, dy, dz in self.actions:
            nx, ny, nz = int(x + dx), int(y + dy), int(z + dz)
            # 1 si obstacle ou hors map, 0 sinon
            if not (0 <= nx < self.env.size and 0 <= ny < self.env.size and 0 <= nz < self.env.size):
                val = 1
            elif self.env.knowledge_grid[nx, ny, nz] == OBSTACLE or (nx, ny, nz) in peer_positions:
                val = 1
            else:
                val = 0
            state.append(val)
        
        # Position relative 3D et batterie
        rel_x = (self.pos[0] - CHARGING_STATION[0]) / self.env.size
        rel_y = (self.pos[1] - CHARGING_STATION[1]) / self.env.size
        rel_z = (self.pos[2] - CHARGING_STATION[2]) / self.env.size
        batt = self.battery / CONFIG['BATTERY']
        
        return np.array(state + [rel_x, rel_y, rel_z, batt])

    def sense(self):
        """Mise à jour de la carte mentale 3D."""
        x, y, z = self.pos
        r = 2 # Rayon de perception
        # Découpage volumétrique
        x_min, x_max = max(0, int(x-r)), min(self.env.size, int(x+r+1))
        y_min, y_max = max(0, int(y-r)), min(self.env.size, int(y+r+1))
        z_min, z_max = max(0, int(z-r)), min(self.env.size, int(z+r+1))
        
        current_view = self.env.true_grid[x_min:x_max, y_min:y_max, z_min:z_max]
        old_view = self.env.knowledge_grid[x_min:x_max, y_min:y_max, z_min:z_max].copy()
        
        # Mise à jour de la mémoire
        self.env.knowledge_grid[x_min:x_max, y_min:y_max, z_min:z_max] = \
            np.where(self.env.knowledge_grid[x_min:x_max, y_min:y_max, z_min:z_max] == UNKNOWN,
                     np.where(current_view == OBSTACLE, OBSTACLE, FREE),
                     self.env.knowledge_grid[x_min:x_max, y_min:y_max, z_min:z_max])
        
        # Détection des changements pour l'interface
        changes = np.where(self.env.knowledge_grid[x_min:x_max, y_min:y_max, z_min:z_max] != old_view)
        if len(changes[0]) > 0:
            self.env.map_changed_this_step = True
            for i in range(len(changes[0])):
                lx, ly, lz = changes[0][i], changes[1][i], changes[2][i]
                gx, gy, gz = x_min+lx, y_min+ly, z_min+lz
                val = self.env.knowledge_grid[gx, gy, gz]
                # Envoi des coordonnées 3D
                self.env.newly_discovered.append({'x': int(gx), 'y': int(gy), 'z': int(gz), 'type': int(val)})

    def get_next_step_bfs(self, target):
        """A* 3D Search"""
        start = (int(self.pos[0]), int(self.pos[1]), int(self.pos[2]))
        goal = (int(target[0]), int(target[1]), int(target[2]))
        queue = [(0, 0, start[0], start[1], start[2], None)]
        costs = {start: 0}
        
        limit = (self.env.size ** 3) * 1.5
        expanded = 0
        
        while queue:
            f, cost, cx, cy, cz, first_step = heapq.heappop(queue)
            expanded += 1
            if (cx, cy, cz) == goal: return first_step
            if expanded > limit: break 

            neighbors = [
                (cx+1, cy, cz), (cx-1, cy, cz),
                (cx, cy+1, cz), (cx, cy-1, cz),
                (cx, cy, cz+1), (cx, cy, cz-1)
            ]
            random.shuffle(neighbors)

            for nx, ny, nz in neighbors:
                if 0 <= nx < self.env.size and 0 <= ny < self.env.size and 0 <= nz < self.env.size:
                    if self.env.knowledge_grid[nx, ny, nz] != OBSTACLE:
                        step_cost = 1
                        if (nx, ny, nz) in self.env.visited_cells: step_cost = 20
                        new_cost = cost + step_cost
                        if new_cost < costs.get((nx, ny, nz), float('inf')):
                            costs[(nx, ny, nz)] = new_cost
                            nxt = first_step if first_step is not None else (nx, ny, nz)
                            h = abs(nx - goal[0]) + abs(ny - goal[1]) + abs(nz - goal[2])
                            heapq.heappush(queue, (new_cost + h * 2, new_cost, nx, ny, nz, nxt))
                            
        self.env.unreachable_blacklist.add(goal)
        return None

    def plan_next_move(self, robots):
        if self.state == 'DEAD':
            self.next_pos_intention = None
            return

        dist_to_base = abs(self.pos[0] - CHARGING_STATION[0]) + \
                       abs(self.pos[1] - CHARGING_STATION[1]) + \
                       abs(self.pos[2] - CHARGING_STATION[2])
        if dist_to_base <= 1: self.battery = CONFIG['BATTERY']

        dist = self.env.base_distances.get((int(self.pos[0]), int(self.pos[1]), int(self.pos[2])), float('inf'))
        if dist == float('inf'):
            dist = dist_to_base
        
        safety_margin = self.env.size * 2
        
        if self.battery < (dist + safety_margin) and dist_to_base > 1:
            self.state = 'RETURNING'; self.target = CHARGING_STATION
        
        if self.state == 'RETURNING' and dist_to_base <= 1:
            self.state = 'EXPLORING'

        self.next_pos_intention = None
        state_rl = self.get_local_state(robots)

        if self.state == 'RETURNING':
            self.current_mode = "RTB"
            step = self.get_next_step_bfs(CHARGING_STATION)
            if step: self.next_pos_intention = np.array(step)
            return

        if self.target is not None:
            self.current_mode = "BFS"
            step = self.get_next_step_bfs(self.target)
            if step:
                diff = np.array(step) - self.pos
                for idx, action in enumerate(self.actions):
                    if action == tuple(diff): self.last_action = idx; break
            else:
                self.target = None; self.current_mode = "DQN"
                self.last_action = self.choose_action_rl(state_rl)
        else:
            self.current_mode = "DQN"
            self.last_action = self.choose_action_rl(state_rl)

        if self.next_pos_intention is None:
            dx, dy, dz = self.actions[self.last_action]
            self.next_pos_intention = self.pos + np.array([dx, dy, dz])
        
        self.last_state = state_rl

    def choose_action_rl(self, state):
        # 6 actions possibles
        if random.random() < self.epsilon:
            return random.randint(0, 5)
        q_values = self.brain.predict(state)
        return np.argmax(q_values)
    
    def execute_move(self, robots):
        if self.state == 'DEAD': return
        self.last_collision = False
        if self.next_pos_intention is not None:
            nx, ny, nz = self.next_pos_intention
            
            if not (0 <= nx < self.env.size and 0 <= ny < self.env.size and 0 <= nz < self.env.size): return

            # Vérification de collision 3D
            if self.env.true_grid[nx, ny, nz] == OBSTACLE:
                self.brain.train(self.last_state, self.last_action, -100)
                self.target = None 
                self.last_collision = True
            else:
                self.pos = self.next_pos_intention
                reward = -1
                t_pos = tuple(self.pos)
                if t_pos not in self.env.visited_cells: reward = 50 
                elif t_pos in self.env.visited_cells: reward = -30
                
                self.env.record_visit(self.pos)
                self.battery -= 1
                
                if self.battery <= 0:
                    self.state = 'DEAD'
                    self.current_mode = "DEAD"
                    
                if self.state == 'EXPLORING' and self.last_state is not None:
                    new_state = self.get_local_state(robots)
                    target = reward + self.gamma * np.max(self.brain.predict(new_state))
                    self.brain.train(self.last_state, self.last_action, target)

# =============================================================================
# FONCTIONS : ALLOCATION DE CIBLES (Enchères 3D)
# =============================================================================
def get_frontiers(env):
    """Trouve les voxels frontières 3D."""
    frontiers = []
    g = env.knowledge_grid
    # Trouve indices où g == FREE
    frees = np.argwhere(g == FREE)
    np.random.shuffle(frees)
    
    # Echantillonnage
    count = 0
    for idx in frees:
        x, y, z = idx
        # Voisins 6-connexité
        neighbors = [
            (x+1,y,z), (x-1,y,z), 
            (x,y+1,z), (x,y-1,z),
            (x,y,z+1), (x,y,z-1)
        ]
        is_frontier = False
        for nx, ny, nz in neighbors:
            if 0<=nx<env.size and 0<=ny<env.size and 0<=nz<env.size:
                if g[nx,ny,nz] == UNKNOWN:
                    is_frontier = True; break
        if is_frontier:
            frontiers.append(idx)
            count += 1
            if count > 200: break
            
    return np.array(frontiers)

def assign_targets(robots, env):
    frontiers = get_frontiers(env)
    avail = [r for r in robots if r.state not in ('RETURNING', 'DEAD')]
    if len(frontiers) == 0: return

    target_sample = frontiers
    if len(frontiers) > 60:
        indices = np.random.choice(len(frontiers), 60, replace=False)
        target_sample = frontiers[indices]

    assigned_targets = set()
    for r in robots:
        if r.target is not None:
            assigned_targets.add(tuple(r.target))

    for r in avail:
        if r.target is None:
            avail_targets = [t for t in target_sample if tuple(t) not in assigned_targets and tuple(t) not in env.unreachable_blacklist]
            if not avail_targets: break
            
            # Distance Manhattan 3D
            dists = np.sum(np.abs(np.array(avail_targets) - r.pos), axis=1)
            best_idx = np.argmin(dists)
            r.target = avail_targets[best_idx]
            assigned_targets.add(tuple(r.target))

# --- VARIABLES GLOBALES ---
env = None
robots = []
step_cnt = 0
stagnation_cnt = 0
last_cov = 0
history = {'steps': [], 'coverage': [], 'redundancy': []}

@app.route('/', methods=['GET', 'POST'])
def config():
    if request.method == 'POST':
        global CONFIG
        CONFIG['GRID_SIZE'] = int(request.form.get('grid_size'))
        CONFIG['NUM_ROBOTS'] = int(request.form.get('num_robots'))
        CONFIG['DENSITY'] = float(request.form.get('density'))
        CONFIG['BATTERY'] = int(request.form.get('battery'))
        return redirect(url_for('simulation'))
    return render_template('config.html')

@app.route('/simulation')
def simulation():
    global env, robots, step_cnt, history, stagnation_cnt, last_cov
    env = Environment(CONFIG['GRID_SIZE'], CONFIG['DENSITY'])
    robots = []
    
    cx, cy, cz = CHARGING_STATION
    
    # Génération dynamique d'offsets 3D uniques (jusqu'à 124 positions autour de la base)
    offsets = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            for dz in range(-2, 3):
                if dx == 0 and dy == 0 and dz == 0: continue
                offsets.append((dx, dy, dz))
    offsets.sort(key=lambda o: abs(o[0]) + abs(o[1]) + abs(o[2]))
    
    for i in range(CONFIG['NUM_ROBOTS']):
        ox, oy, oz = offsets[i % len(offsets)]
        r = Robot(i, cx + ox, cy + oy, cz + oz, env)
        robots.append(r)
        
    step_cnt = 0
    stagnation_cnt = 0
    last_cov = 0
    history = {'steps': [], 'coverage': [], 'redundancy': []}
    return render_template('simulation_3D.html', config=CONFIG)

@app.route('/step')
def step():
    global env, robots, step_cnt, history, stagnation_cnt, last_cov
    if not env: return jsonify({'status': 'ERROR'})

    cov = env.get_exploration_rate()
    
    if cov == last_cov: stagnation_cnt += 1
    else: stagnation_cnt = 0; last_cov = cov

    if cov >= 99.0 or stagnation_cnt > 300:
        reason = "Coverage Reached" if cov >= 99.0 else "Limit Reached"
        return jsonify({'status': 'COMPLETE', 'reason': reason, 'stats': {'coverage': cov, 'step': step_cnt}})

    if env.map_changed_this_step:
        env.update_base_distances()
        env.map_changed_this_step = False

    for r in robots: 
        if r.state != 'DEAD': r.sense()
    assign_targets(robots, env)
    for r in robots: r.plan_next_move(robots)
    
    # Gestion conflits 3D
    intentions = {}
    for r in robots:
        if r.next_pos_intention is not None:
            k = tuple(r.next_pos_intention)
            if k not in intentions: intentions[k] = []
            intentions[k].append(r)
    
    for k, candidates in intentions.items():
        if len(candidates) > 1:
            winner = random.choice(candidates)
            for r in candidates:
                if r != winner: r.next_pos_intention = None
                
    for i in range(len(robots)):
        for j in range(i+1, len(robots)):
            r1, r2 = robots[i], robots[j]
            if r1.next_pos_intention is not None and r2.next_pos_intention is not None:
                if tuple(r1.next_pos_intention) == tuple(r2.pos) and tuple(r2.next_pos_intention) == tuple(r1.pos):
                    if random.choice([True, False]): r1.next_pos_intention = None
                    else: r2.next_pos_intention = None
    
    stationary_pos = set()
    for r in robots:
        if r.next_pos_intention is None: stationary_pos.add(tuple(r.pos))
    for r in robots:
        if r.next_pos_intention is not None:
            if tuple(r.next_pos_intention) in stationary_pos:
                r.next_pos_intention = None
                stationary_pos.add(tuple(r.pos))

    for r in robots: r.execute_move(robots)
    step_cnt += 1
    
    if step_cnt % 5 == 0:
        history['steps'].append(step_cnt)
        history['coverage'].append(round(cov, 2))
        history['redundancy'].append(env.redundancy_score)

    new_blocks = env.newly_discovered[:]
    env.newly_discovered = []

    return jsonify({
        'status': 'RUNNING',
        'robots': [{
            'id': r.id, 
            'x': int(r.pos[0]), 'y': int(r.pos[1]), 'z': int(r.pos[2]),
            'battery': r.battery, 'max_battery': CONFIG['BATTERY'],
            'mode': r.current_mode,
            'collision': r.last_collision
        } for r in robots],
        'new_blocks': new_blocks,
        'stats': {
            'step': step_cnt, 
            'coverage': round(cov, 2),
            'redundancy': env.redundancy_score
        },
        'graph': history,
        'base': {'x': int(CHARGING_STATION[0]), 'y': int(CHARGING_STATION[1]), 'z': int(CHARGING_STATION[2])}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
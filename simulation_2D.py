# =============================================================================
# MODULE : AAC / PROJET : EXPLORATION MULTI-ROBOTS (DQN + DIJKSTRA)
# =============================================================================
# Ce script est le cerveau du projet. Il gère la logique des robots, la simulation
# de l'environnement, et le serveur Web qui communique avec l'interface graphique.

from flask import Flask, render_template, jsonify, request, redirect, url_for
import numpy as np
import heapq
import random
import math

# Initialisation de l'application Flask pour servir l'interface Web
app = Flask(__name__)

# --- CONFIGURATION INITIALE ---
# Ce dictionnaire stocke les paramètres qui seront modifiés par l'utilisateur
# via le formulaire HTML au début de la simulation.
CONFIG = {
    'GRID_SIZE': 30,    # Taille de la carte carrée (ex: 30x30 cases)
    'NUM_ROBOTS': 4,    # Nombre de robots dans l'essaim
    'DENSITY': 0.2,     # Pourcentage d'obstacles (20% de la carte)
    'BATTERY': 1000     # Capacité de la batterie (nombre de mouvements)
}
# La position de la station de charge sera calculée dynamiquement au centre
CHARGING_STATION = np.array([0, 0, 0])

# --- CODES NUMÉRIQUES POUR LA GRILLE ---
# On utilise des entiers pour représenter l'état des cases car c'est plus rapide
# à traiter pour l'ordinateur que des chaînes de caractères.
UNKNOWN = 0   # Case inconnue (Noir) - Zone non explorée
FREE = 1      # Case libre (Vide) - Zone traversable
OBSTACLE = 2  # Obstacle (Mur Rouge) - Zone infranchissable

# =============================================================================
# CLASSE : ENVIRONMENT (Le Monde)
# =============================================================================
class Environment:
    """
    Cette classe représente le monde physique dans lequel les robots évoluent.
    Elle gère deux grilles distinctes :
    1. true_grid : La réalité (avec tous les murs), inaccessible directement aux robots.
    2. knowledge_grid : La carte mentale partagée que les robots construisent ensemble.
    """
    def __init__(self, size, density):
        self.size = size
        
        # Grilles 3D (x, y, 1) pour compatibilité, mais logique 2D
        self.true_grid = np.zeros((size, size, 1), dtype=int)
        self.knowledge_grid = np.zeros((size, size, 1), dtype=int)
        
        # Suivi des cases visitées pour calculer la redondance
        self.visited_cells = set()
        self.redundancy_score = 0
        
        # Liste pour envoyer seulement les nouveautés à l'interface (Optimisation)
        self.newly_discovered = [] 
        self.base_distances = {}
        self.unreachable_blacklist = set()
        self.map_changed_this_step = True
        
        # Placement de la base au centre exact de la carte
        global CHARGING_STATION
        mid = size // 2
        CHARGING_STATION = np.array([mid, mid, 0])
        self.visited_cells.add(tuple(CHARGING_STATION))
        
        # Génération procédurale du terrain
        self.generate_obstacles(density)

    def generate_obstacles(self, density):
        """
        Crée la carte de départ.
        - Place des murs indestructibles tout autour (enceinte).
        - Place des obstacles aléatoires à l'intérieur.
        - Garantit que la zone de départ est vide pour ne pas bloquer les robots.
        """
        # 1. Murs d'enceinte (Bordures)
        for i in range(self.size):
            self.true_grid[i, 0, 0] = OBSTACLE
            self.true_grid[i, self.size-1, 0] = OBSTACLE
            self.true_grid[0, i, 0] = OBSTACLE
            self.true_grid[self.size-1, i, 0] = OBSTACLE

        # 2. Obstacles aléatoires selon la densité choisie
        num_obstacles = int(self.size * self.size * density)
        for _ in range(num_obstacles):
            x, y = np.random.randint(1, self.size-1, 2)
            self.true_grid[x, y, 0] = OBSTACLE
            
        # 3. Zone de sécurité (Safe Zone) autour de la base
        cx, cy, _ = CHARGING_STATION
        r_safe = 2 # Rayon de sécurité
        for x in range(cx-r_safe, cx+r_safe+1):
            for y in range(cy-r_safe, cy+r_safe+1):
                if 0 <= x < self.size and 0 <= y < self.size:
                    self.true_grid[x, y, 0] = FREE

    def get_exploration_rate(self):
        """
        Calcule le pourcentage de la carte explorée.
        C'est l'indicateur principal de réussite de la mission.
        """
        floor = self.true_grid[:, :, 0]
        # On compte le nombre total de cases qui NE SONT PAS des murs
        accessible = np.count_nonzero(floor != OBSTACLE)
        if accessible == 0: return 0
        
        # On compte combien de ces cases ont été découvertes (ne sont plus UNKNOWN)
        known = 0
        for x in range(self.size):
            for y in range(self.size):
                if self.true_grid[x,y,0] != OBSTACLE and self.knowledge_grid[x,y,0] != UNKNOWN:
                    known += 1
        return (known / accessible) * 100

    def record_visit(self, pos):
        """
        Incrémente le score de redondance si un robot marche sur une case déjà visitée.
        L'objectif est de garder ce score le plus bas possible.
        """
        if np.array_equal(pos, CHARGING_STATION): return
        t_pos = tuple(pos)
        if t_pos in self.visited_cells:
            self.redundancy_score += 1
        else:
            self.visited_cells.add(t_pos)

    def update_base_distances(self):
        """
        Calcule la vraie distance (BFS) de la base vers toutes les cases CONNUES.
        Empêche les robots de se faire piéger dans des labyrinthes en "U".
        """
        from collections import deque
        start = (int(CHARGING_STATION[0]), int(CHARGING_STATION[1]))
        queue = deque([(start[0], start[1])])
        self.base_distances = {start: 0}
        
        while queue:
            cx, cy = queue.popleft()
            cost = self.base_distances[(cx, cy)]
            
            for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]:
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.knowledge_grid[nx, ny, 0] != OBSTACLE:
                        if (nx, ny) not in self.base_distances:
                            self.base_distances[(nx, ny)] = cost + 1
                            queue.append((nx, ny))

# =============================================================================
# CLASSE : BRAIN (Deep Q-Network - L'Intelligence Artificielle)
# =============================================================================
class Brain:
    """
    Réseau de neurones 'Perceptron Multi-Couches' codé à la main (From Scratch).
    Il permet au robot d'apprendre à naviguer par expérience et imitation.
    
    Structure :
    - Entrée : 7 neurones (Environnement local + Batterie + Position relative)
    - Cachée : 16 neurones (Traitement de l'information)
    - Sortie : 4 neurones (Valeur de chaque direction : Haut, Bas, Gauche, Droite)
    """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = 16
        self.learning_rate = 0.01 # Vitesse d'apprentissage
        
        # Initialisation aléatoire des poids (Synapses)
        # W1 connecte l'entrée à la couche cachée
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        # W2 connecte la couche cachée à la sortie
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))

    def relu(self, z):
        """Fonction d'activation ReLU : Élimine les valeurs négatives pour introduire de la non-linéarité."""
        return np.maximum(0, z)

    def predict(self, state):
        """
        Passe avant (Forward Propagation) :
        Le cerveau prend l'état actuel et devine la meilleure action.
        """
        state = np.array(state).reshape(1, -1)
        self.z1 = np.dot(state, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2[0] # Retourne les scores pour les 4 directions

    def train(self, state, action, target_q):
        """
        Rétropropagation (Backpropagation) :
        Le cerveau corrige ses erreurs. Si l'action choisie était mauvaise (mur),
        il ajuste ses poids pour ne plus refaire la même erreur.
        """
        state = np.array(state).reshape(1, -1)
        
        # 1. On recalcule la sortie actuelle
        z1 = np.dot(state, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        
        # 2. Calcul de l'erreur (Différence entre prédiction et réalité)
        current_q = z2
        loss = current_q[0][action] - target_q
        
        # 3. Descente de gradient (Correction des poids)
        d_z2 = np.zeros_like(z2)
        d_z2[0][action] = loss
        
        d_W2 = np.dot(a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * (z1 > 0)
        
        d_W1 = np.dot(state.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        
        # Application de la correction
        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2

# =============================================================================
# CLASSE : ROBOT (L'Agent Mobile)
# =============================================================================
class Robot:
    """
    Représente un robot autonome capable de :
    - Percevoir son environnement (sense)
    - Planifier un chemin (plan_next_move)
    - Apprendre et éviter les obstacles (brain)
    """
    def __init__(self, id, start_x, start_y, env):
        self.id = id
        self.pos = np.array([int(start_x), int(start_y), 0], dtype=int)
        self.env = env
        self.target = None
        self.battery = CONFIG['BATTERY']
        self.state = 'EXPLORING'
        self.next_pos_intention = None
        self.current_mode = "IDLE" # Pour l'affichage (BFS ou DQN)
        self.last_collision = False 
        self.stuck_cnt = 0
        
        # Le robot possède son propre cerveau (Deep Q-Network)
        self.brain = Brain(7, 4)
        
        # Paramètres d'apprentissage
        self.epsilon = 0.1 # 10% de chance d'explorer au hasard
        self.gamma = 0.9   # Importance du futur
        self.last_state = None
        self.last_action = None
        self.actions = [(0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0)]

    def get_local_state(self, robots):
        """Prépare les données pour le réseau de neurones (Normalisation)."""
        x, y, _ = self.pos
        peer_positions = set(tuple(r.pos) for r in robots if r.id != self.id and r.state != 'DEAD')
        state = []
        for dx, dy, _ in self.actions:
            nx, ny = int(x + dx), int(y + dy)
            # 1 si obstacle ou hors map, 0 sinon
            if not (0 <= nx < self.env.size and 0 <= ny < self.env.size):
                val = 1
            elif self.env.knowledge_grid[nx, ny, 0] == OBSTACLE or (nx, ny, 0) in peer_positions:
                val = 1
            else:
                val = 0
            state.append(val)
        
        # Ajout info distance base et batterie
        rel_x = (self.pos[0] - CHARGING_STATION[0]) / self.env.size
        rel_y = (self.pos[1] - CHARGING_STATION[1]) / self.env.size
        batt = self.battery / CONFIG['BATTERY']
        
        return np.array(state + [rel_x, rel_y, batt])

    def sense(self):
        """Mise à jour de la carte mentale (Fog of War)."""
        x, y, _ = self.pos
        r = 2
        x_min, x_max = max(0, int(x-r)), min(self.env.size, int(x+r+1))
        y_min, y_max = max(0, int(y-r)), min(self.env.size, int(y+r+1))
        
        # On compare la vérité terrain avec la mémoire
        current_view = self.env.true_grid[x_min:x_max, y_min:y_max, 0]
        old_view = self.env.knowledge_grid[x_min:x_max, y_min:y_max, 0].copy()
        
        # Si c'était INCONNU, on met la vraie valeur (LIBRE ou OBSTACLE)
        self.env.knowledge_grid[x_min:x_max, y_min:y_max, 0] = \
            np.where(self.env.knowledge_grid[x_min:x_max, y_min:y_max, 0] == UNKNOWN,
                     np.where(current_view == OBSTACLE, OBSTACLE, FREE),
                     self.env.knowledge_grid[x_min:x_max, y_min:y_max, 0])
        
        # 3. On note ce qui a changé pour le renvoyer à l'interface WEB
        changes = np.where(self.env.knowledge_grid[x_min:x_max, y_min:y_max, 0] != old_view)
        if len(changes[0]) > 0:
            self.env.map_changed_this_step = True
            for i in range(len(changes[0])):
                lx, ly = changes[0][i], changes[1][i]
                gx, gy = x_min+lx, y_min+ly
                val = self.env.knowledge_grid[gx, gy, 0]
                self.env.newly_discovered.append({'x': int(gx), 'y': int(gy), 'type': int(val)})

    def get_next_step_bfs(self, target):
        """
        Pathfinding A* Search vers la cible.
        Pénalise les cases déjà visitées pour éviter les zig-zags (step_cost = 20).
        """
        start = (int(self.pos[0]), int(self.pos[1]))
        goal = (int(target[0]), int(target[1]))
        
        queue = [(0, 0, start[0], start[1], None)]
        costs = {start: 0}
        
        # Limite de recherche pour éviter de bloquer le CPU
        limit = (self.env.size ** 2) * 2
        expanded = 0
        
        while queue:
            f, cost, cx, cy, first_step = heapq.heappop(queue)
            expanded += 1
            
            if (cx, cy) == goal:
                return first_step
                
            if expanded > limit:
                break 

            neighbors = [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]
            random.shuffle(neighbors)

            for nx, ny in neighbors:
                if 0 <= nx < self.env.size and 0 <= ny < self.env.size:
                    if self.env.knowledge_grid[nx, ny, 0] != OBSTACLE:
                        step_cost = 1
                        if (nx, ny) in self.env.visited_cells:
                            step_cost = 20
                            
                        new_cost = cost + step_cost
                        if new_cost < costs.get((nx, ny), float('inf')):
                            costs[(nx, ny)] = new_cost
                            nxt = first_step if first_step is not None else (nx, ny)
                            h = abs(nx - goal[0]) + abs(ny - goal[1])
                            heapq.heappush(queue, (new_cost + h * 2, new_cost, nx, ny, nxt))
                            
        self.env.unreachable_blacklist.add(goal)
        return None

    def plan_next_move(self, robots):
        """
        CERVEAU HYBRIDE.
        Décide quel mode utiliser :
        1. Mode RETOUR : Si batterie faible.
        2. Mode BFS (Expert) : Si une cible est définie.
        3. Mode DQN (Apprentissage) : Si bloqué ou pas de cible.
        """
        if self.state == 'DEAD':
            self.next_pos_intention = None
            return

        dist_to_base = abs(self.pos[0] - CHARGING_STATION[0]) + abs(self.pos[1] - CHARGING_STATION[1])
        
        # BUG FIX: Recharge AND immediately clear RETURNING state so robot doesn't waste
        # another step heading to base with a full battery.
        if dist_to_base <= 1:
            self.battery = CONFIG['BATTERY']
            self.state = 'EXPLORING'
            self.target = None  # Pick a new frontier

        # --- CORRECTION 2 : MARGE DE SÉCURITÉ DYNAMIQUE ---
        dist = self.env.base_distances.get((int(self.pos[0]), int(self.pos[1])), float('inf'))
        if dist == float('inf'):
            dist = dist_to_base
            
        # BUG FIX: safety_margin was env.size * 1.5 = 45 steps on a 30x30 grid!
        # That means robots turn back when they still have 45+ steps of battery left,
        # wasting exploration time. Use a flat margin of 10 steps instead.
        safety_margin = 10
        
        if self.battery < (dist + safety_margin) and dist_to_base > 1:
            self.state = 'RETURNING'; self.target = CHARGING_STATION
        
        if self.state == 'RETURNING' and dist_to_base <= 1:
            self.state = 'EXPLORING'

        self.next_pos_intention = None
        state_rl = self.get_local_state(robots)

        if self.state == 'RETURNING':
            self.current_mode = "RTB"
            step = self.get_next_step_bfs(CHARGING_STATION)
            if step: self.next_pos_intention = np.array([step[0], step[1], 0])
            return

        if self.target is not None:
            self.current_mode = "BFS"
            step = self.get_next_step_bfs(self.target)
            if step:
                # On suit l'expert
                diff = np.array([step[0], step[1], 0]) - self.pos
                for idx, action in enumerate(self.actions):
                    if action == tuple(diff): self.last_action = idx; break
            else:
                # Si l'expert échoue (bloqué), le DQN prend le relais
                self.target = None; self.current_mode = "DQN"
                self.last_action = self.choose_action_rl(state_rl)
        else:
            self.current_mode = "DQN"
            self.last_action = self.choose_action_rl(state_rl)

        if self.next_pos_intention is None:
            dx, dy, _ = self.actions[self.last_action]
            self.next_pos_intention = self.pos + np.array([dx, dy, 0])
        
        self.last_state = state_rl

    def choose_action_rl(self, state):
        """Choix action via Réseau de Neurones"""
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        q_values = self.brain.predict(state)
        return np.argmax(q_values)
    
    def execute_move(self, robots):
        """Exécute le mouvement et entraîne le réseau (Imitation Learning)."""
        if self.state == 'DEAD' or self.next_pos_intention is None:
            self.stuck_cnt += 1
            if self.stuck_cnt > 3:
                self.target = None
                self.stuck_cnt = 0
            return
            
        self.last_collision = False
        nx, ny, _ = self.next_pos_intention
        
        # 1. Vérification des limites du monde
        if not (0 <= nx < self.env.size and 0 <= ny < self.env.size): return

        # 2. Vérification des obstacles (Vraie grille)
        if self.env.true_grid[nx, ny, 0] == OBSTACLE:
            self.brain.train(self.last_state, self.last_action, -100) # Punition sévère
            self.target = None 
            self.last_collision = True
        else:
            self.stuck_cnt = 0
            # 3. Le mouvement est valide
            self.pos = self.next_pos_intention
            
            # 4. Calcul de la récompense locale (DQN)
            reward = -1 # Coût de déplacement de base (encourage l'efficacité)
            t_pos = tuple(self.pos)
            if t_pos not in self.env.visited_cells: reward = 50 # Grande récompense (Découverte)
            elif t_pos in self.env.visited_cells: reward = -30  # Punition (Redondance)
            
            self.env.record_visit(self.pos)
            self.battery -= 1
            
            # Gestion de la batterie faible
            if self.battery <= 0:
                self.state = 'DEAD'
                self.current_mode = "DEAD"
                
            # 5. Apprentissage de l'action passée
            if self.state == 'EXPLORING' and self.last_state is not None:
                new_state = self.get_local_state(robots)
                # Formule de Q-Learning : Q = r + gamma * max(Q_next)
                target = reward + self.gamma * np.max(self.brain.predict(new_state))
                self.brain.train(self.last_state, self.last_action, target)

# =============================================================================
# FONCTIONS : ALLOCATION DE CIBLES (Enchères)
# =============================================================================
def get_frontiers(env):
    """Trouve les zones frontières (limite entre exploré et inconnu)."""
    frontiers = []
    g = env.knowledge_grid
    frees = np.argwhere(g[:,:,0] == FREE)
    # BUG FIX: Scan ALL free cells, not just 200.
    # With only 200 samples in a large map, we miss huge unexplored regions.
    for x, y in frees:
        for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
            if 0<=nx<env.size and 0<=ny<env.size:
                if g[nx,ny,0] == UNKNOWN:
                    frontiers.append(np.array([x,y,0]))
                    break
    return np.array(frontiers) if frontiers else np.array([])

def assign_targets(robots, env):
    """
    Assigne intelligemment les cibles.
    Chaque robot reçoit la frontière la plus proche de lui (Greedy Allocation).
    """
    frontiers = get_frontiers(env)
    avail = [r for r in robots if r.state not in ('RETURNING', 'DEAD')]
    if len(frontiers) == 0:
        # BUG FIX: If NO frontiers at all, clear the blacklist - it may have grown
        # too aggressively from temporary pathfinding failures (e.g. robot in the way).
        if len(env.unreachable_blacklist) > 0:
            env.unreachable_blacklist.clear()
        return

    valid_frontiers = [f for f in frontiers if tuple(f) not in env.unreachable_blacklist]
    if len(valid_frontiers) == 0:
        # BUG FIX: All frontiers are blacklisted - this means the blacklist is wrong.
        # Clear it and retry with all frontiers so robots don't just freeze.
        env.unreachable_blacklist.clear()
        valid_frontiers = list(frontiers)
    
    target_sample = valid_frontiers
    if len(valid_frontiers) > 60:
        indices = np.random.choice(len(valid_frontiers), 60, replace=False)
        target_sample = np.array(valid_frontiers)[indices]

    assigned_targets = set()
    for r in robots:
        if r.target is not None:
            assigned_targets.add(tuple(r.target))

    for r in avail:
        if r.target is None:
            avail_targets = [t for t in target_sample if tuple(t) not in assigned_targets and tuple(t) not in env.unreachable_blacklist]
            if not avail_targets: break
            
            dists = np.sum(np.abs(np.array(avail_targets) - r.pos), axis=1)
            best_idx = np.argmin(dists)
            r.target = avail_targets[best_idx]
            assigned_targets.add(tuple(r.target))

# --- VARIABLES GLOBALES FLASK ---
env = None
robots = []
step_cnt = 0
stagnation_cnt = 0
last_cov = 0
history = {'steps': [], 'coverage': [], 'redundancy': []}

# =============================================================================
# ROUTES WEB (API)
# =============================================================================
@app.route('/', methods=['GET', 'POST'])
def config():
    """Page d'accueil pour configurer la simulation."""
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
    """Initialise la simulation et charge la page 3D."""
    global env, robots, step_cnt, history, stagnation_cnt, last_cov
    env = Environment(CONFIG['GRID_SIZE'], CONFIG['DENSITY'])
    robots = []
    
    # Placement des robots autour de la base
    cx, cy, _ = CHARGING_STATION
    
    # Génération dynamique d'offsets uniques (jusqu'à 24 positions autour de la base)
    offsets = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx == 0 and dy == 0: continue
            offsets.append((dx, dy))
    offsets.sort(key=lambda o: abs(o[0]) + abs(o[1]))
    
    for i in range(CONFIG['NUM_ROBOTS']):
        off_x, off_y = offsets[i % len(offsets)]
        r = Robot(i, cx + off_x, cy + off_y, env)
        robots.append(r)
        
    step_cnt = 0
    stagnation_cnt = 0
    last_cov = 0
    history = {'steps': [], 'coverage': [], 'redundancy': []}
    return render_template('simulation_2D.html', config=CONFIG)

@app.route('/step')
def step():
    global env, robots, step_cnt, history, stagnation_cnt, last_cov
    if not env: return jsonify({'status': 'ERROR'})

    n_steps = request.args.get('n', default=1, type=int)
    if n_steps > 30: n_steps = 30 # Cap to prevent server lockup
    
    batch = []
    
    for _ in range(n_steps):
        cov = env.get_exploration_rate()
        
        # Détection de fin (100% ou Stagnation prolongée)
        if abs(cov - last_cov) < 0.000001: stagnation_cnt += 1
        else: stagnation_cnt = 0; last_cov = cov

        # Fin de mission
        if cov >= 99.0 or stagnation_cnt > 1000:
            reason = "Coverage Reached" if cov >= 99.0 else "Limit Reached"
            batch.append({'status': 'COMPLETE', 'reason': reason, 'stats': {'coverage': cov, 'step': step_cnt}})
            return jsonify({'batch': batch})

        if env.map_changed_this_step:
            env.update_base_distances()
            env.map_changed_this_step = False

        # 1. Perception
        for r in robots: 
            if r.state != 'DEAD': r.sense()
        # 2. Assignation
        assign_targets(robots, env)
        # 3. Planification
        for r in robots: r.plan_next_move(robots)
        
        # 4. GESTION DES CONFLITS (Anti-Deadlock)
        intentions = {}
        for r in robots:
            if r.next_pos_intention is not None:
                k = tuple(r.next_pos_intention)
                if k not in intentions: intentions[k] = []
                intentions[k].append(r)
        
        # Résolution Conflit dynamique (2 robots, 1 case)
        for k, candidates in intentions.items():
            if len(candidates) > 1:
                winner = random.choice(candidates)
                for r in candidates:
                    if r != winner: r.next_pos_intention = None # Le perdant attend
        
        # Résolution Conflit dynamique (Swapping / Head-to-Head)
        for i in range(len(robots)):
            for j in range(i+1, len(robots)):
                r1, r2 = robots[i], robots[j]
                if r1.next_pos_intention is not None and r2.next_pos_intention is not None:
                    if tuple(r1.next_pos_intention) == tuple(r2.pos) and tuple(r2.next_pos_intention) == tuple(r1.pos):
                        # Les deux se foncent dessus, un seul passe
                        if random.choice([True, False]): r1.next_pos_intention = None
                        else: r2.next_pos_intention = None
        
        # Résolution Conflit avec un robot stationnaire
        stationary_pos = set()
        for r in robots:
            if r.next_pos_intention is None: stationary_pos.add(tuple(r.pos))
        for r in robots:
            if r.next_pos_intention is not None:
                if tuple(r.next_pos_intention) in stationary_pos:
                    r.next_pos_intention = None # Bloqué par un robot immobile
                    stationary_pos.add(tuple(r.pos)) # Devient immobile à son tour

        # 5. Exécution des mouvements validés
        for r in robots: r.execute_move(robots)
        
        step_cnt += 1
        
        # Mise à jour des graphiques toutes les 5 steps
        if step_cnt % 5 == 0:
            history['steps'].append(step_cnt)
            history['coverage'].append(round(cov, 2))
            history['redundancy'].append(env.redundancy_score)

        new_blocks = env.newly_discovered[:]
        env.newly_discovered = []

        # Construction de l'état actuel de la simulation
        batch.append({
            'status': 'RUNNING',
            'robots': [{
                'id': r.id, 
                'x': int(r.pos[0]), 'y': int(r.pos[1]),
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
            'graph': {'steps': history['steps'][:], 'coverage': history['coverage'][:], 'redundancy': history['redundancy'][:]} if step_cnt % 5 == 0 else None,
            'base': {'x': int(CHARGING_STATION[0]), 'y': int(CHARGING_STATION[1])}
        })

    return jsonify({'batch': batch})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
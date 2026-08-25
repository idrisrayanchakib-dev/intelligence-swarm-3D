
🤖 Exploration Autonome Multi-Robots : Architecture Hybride

![alt text](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)


![alt text](https://img.shields.io/badge/Flask-Web%20Server-green?style=for-the-badge&logo=flask)


![alt text](https://img.shields.io/badge/Frontend-Three.js-orange?style=for-the-badge&logo=three.js)


![alt text](https://img.shields.io/badge/Status-Finished-brightgreen?style=for-the-badge)

Simulation 3D avancée d'un essaim de robots mobiles (Swarm Robotics) explorant un environnement inconnu.

Ce projet repose sur une Architecture d'Intelligence Artificielle Hybride combinant la planification déterministe (Dijkstra) et l'apprentissage profond par renforcement (Deep Q-Network) implémenté "from scratch" avec NumPy.

📸 Aperçu du Projet

(Insérez ici une capture d'écran ou un GIF de votre interface "Swarm Command")

![alt text](https://via.placeholder.com/800x400?text=Capture+Ecran+Swarm+Command)

Le système permet de configurer, visualiser et analyser en temps réel l'exploration d'une carte générée procéduralement, avec une gestion stricte des collisions, de l'énergie et de la redondance.

🧠 Architecture Technique : L'Approche Maître-Élève

La particularité de ce projet est le refus d'utiliser des "boîtes noires" (comme PyTorch ou TensorFlow). Le réseau de neurones est entièrement mathématique, codé avec NumPy. L'architecture suit un modèle Teacher-Student utilisant l'Apprentissage par Imitation (Imitation Learning).

1. L'Expert (The Teacher) : Algorithme de Dijkstra

Le robot utilise une variante de l'algorithme de Dijkstra pour la planification globale stratégique.

Rôle : Calculer le chemin optimal vers les "frontières" (limites entre zone connue et inconnue) ou pour le retour à la base.

Logique de Coûts :

Case inconnue = Coût 1

Case déjà visitée = Coût 20 (Force mathématiquement l'exploration de nouvelles zones).

Fail-Safe : En cas de batterie critique, l'expert prend le contrôle total (Override) pour garantir le retour à la station de charge.

2. L'Élève (The Student) : Deep Q-Network (DQN)

Chaque robot possède son propre cerveau artificiel, un Perceptron Multi-Couches (MLP).

Architecture du Réseau :

🟢 Entrée (7 neurones) : Perception locale (4 cases voisines) + Position relative à la base + Niveau de batterie.

🔵 Cachée (16 neurones) : Traitement de l'information (Activation ReLU).

🔴 Sortie (4 neurones) : Valeur Q pour chaque direction (Haut, Bas, Gauche, Droite).

Apprentissage : À chaque pas guidé par l'expert (Dijkstra), le réseau observe la décision et ajuste ses poids synaptiques (Backpropagation) pour "imiter" ce comportement. Il apprend également par renforcement négatif immédiat lors des collisions.

✨ Fonctionnalités Clés
🎨 Simulation 3D "Premium"

Rendu WebGL fluide via Three.js.

Murs d'enceinte et obstacles volumétriques.

Effets de lumière dynamiques, ombres portées et champ d'étoiles.

🗺️ Système SLAM Centralisé

Knowledge Grid : Les robots partagent une carte commune.

Fog of War : Découverte progressive de l'environnement (Brouillard de guerre).

🔋 Gestion Avancée de l'Énergie

Calcul dynamique de la distance de retour (Manhattan).

Marge de sécurité adaptative : Safety = Map_Size * 1.5 pour garantir le retour.

🛡️ Anti-Collision & Anti-Deadlock

Physique stricte : Les robots ne peuvent pas se traverser.

Gestion des priorités (Enchères) si deux robots visent la même case.

Détection des robots stationnaires pour éviter les embouteillages.

📊 Analyse de Performance

Graphes en temps réel (Chart.js) : Taux de couverture vs Redondance.

Calcul de l'efficacité et détection automatique de la stagnation.

🛠️ Installation et Exécution
Prérequis

Python 3.8+

Navigateur web moderne (Chrome, Firefox, Edge)

Installation

Cloner le projet :

code
Bash
download
content_copy
expand_less
git clone https://github.com/votre-username/votre-projet.git
cd votre-projet

Installer les dépendances :

code
Bash
download
content_copy
expand_less
pip install flask numpy

Lancer le serveur :

code
Bash
download
content_copy
expand_less
python app.py

Accéder à l'interface :
Ouvrez votre navigateur à l'adresse http://127.0.0.1:5000

⚙️ Configuration de la Mission

Une interface de configuration permet de définir les paramètres avant le lancement :

Paramètre	Description	Recommandation
Grid Size	Taille de la carte carrée (ex: 30x30).	30 (Rapide) à 100 (Complexe)
Obstacle Density	% de la carte couvert de murs (0.0 à 1.0).	0.2 (Soit 20%)
Active Robots	Nombre d'agents déployés simultanément.	2 à 8
Battery Capacity	Autonomie (nombre de pas).	1000 (Petite carte), 5000+ (Grande carte)

⚠️ Note Importante : Pour les grandes cartes (ex: 100x100), augmentez significativement la capacité de la batterie. Les détours imposés par les obstacles complexes consomment beaucoup d'énergie pour le retour à la base.

📂 Structure du Code
code
Code
download
content_copy
expand_less
/Projet
│
├── app.py                 # Cerveau du projet (Flask, Logique Robots, DQN, Dijkstra)
│
├── static
│   ├── css
│   │   └── style.css      # Design "Cyber-Interface" (Glassmorphism, Animations)
│   └── assets             # Images et ressources
│
└── templates
    ├── config.html        # Page d'accueil (Formulaire de paramétrage)
    └── simulation.html    # Interface de visualisation 3D (Three.js + Chart.js)
🚀 Perspectives d'Amélioration

Communication Limitée : Remplacer la carte partagée instantanée par un échange de données basé sur un rayon de communication.

Persistance du Modèle : Sauvegarder les poids du réseau neuronal (model.save) pour réutiliser un robot entraîné.

Environnement Dynamique : Ajout d'obstacles mouvants.

Développé dans le cadre du module AAC (Apprentissage et Agents Communicants)

Année Universitaire 2025-2026


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import warnings

# Oculta las advertencias de Gym/Gymnasium para una salida limpia
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sb3_contrib import RecurrentPPO
from chase_env_sb3 import EvadeEnv, PursueEnv

# === Rutas a modelos entrenados ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_lstm")
EVADER_MODEL_PATH = os.path.join(MODELS_DIR, "evader_lstm.zip")
PURSUER_MODEL_PATH = os.path.join(MODELS_DIR, "pursuer_lstm.zip")

# === Configurar visualización ===
# Configura el backend de Matplotlib para asegurar que la ventana se muestre.
import matplotlib
matplotlib.use('TkAgg')

# --- Main Script ---
if __name__ == '__main__':
    try:
        # Cargar modelos entrenados
        evader_model = RecurrentPPO.load(EVADER_MODEL_PATH)
        pursuer_model = RecurrentPPO.load(PURSUER_MODEL_PATH)
        print("✅ Modelos LSTM cargados correctamente.")

    except FileNotFoundError:
        print("❗ ERROR: No se encontraron los archivos de modelo LSTM.")
        print("Por favor, asegúrate de haber ejecutado 'train_lstm.py' primero.")
        exit()

    # === Crear entorno visual ===
    # El agente principal será el perseguidor, y el bot será el evasor
    env = PursueEnv(bot_model=evader_model)
    obs, _ = env.reset()

    # === Inicializar estado LSTM ===
    # Para el agente principal (perseguidor)
    pursuer_lstm_state = None
    pursuer_episode_start = True
    
    # === Configurar visualización ===
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(env.bounds[0], env.bounds[1])
    ax.set_ylim(env.bounds[0], env.bounds[1])
    ax.set_aspect("equal")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title("Visualización de persecución con memoria (LSTM)")

    # Crear elementos visuales
    agent_dot, = ax.plot([], [], 'ro', label="Perseguidor (RL)", markersize=10, zorder=4)
    bot_dot, = ax.plot([], [], 'go', label="Evasor (Bot)", markersize=10, zorder=4)
    ax.legend()
    
    # Rastro de movimiento
    trail_agent, = ax.plot([], [], 'r--', linewidth=1, alpha=0.7)
    trail_bot, = ax.plot([], [], 'g--', linewidth=1, alpha=0.7)
    
    history_agent = []
    history_bot = []

    def init():
        agent_dot.set_data([], [])
        bot_dot.set_data([], [])
        trail_agent.set_data([], [])
        trail_bot.set_data([], [])
        return agent_dot, bot_dot, trail_agent, trail_bot

    def update(frame):
        global obs, pursuer_lstm_state, pursuer_episode_start, history_agent, history_bot

        # Predecir acción con memoria LSTM
        # Se necesita expandir la dimensión para el predict de sb3-contrib
        action, pursuer_lstm_state = pursuer_model.predict(
            obs,
            state=pursuer_lstm_state,
            episode_start=np.array([pursuer_episode_start]),
            deterministic=True
        )
        pursuer_episode_start = False

        # Realizar paso en el entorno
        obs, reward, terminated, truncated, info = env.step(action)

        # Reiniciar episodio si terminó
        if terminated or truncated:
            obs, _ = env.reset()
            pursuer_lstm_state = None
            pursuer_episode_start = True
            history_agent.clear()
            history_bot.clear()

        # Actualizar posiciones en visual
        # Usamos las coordenadas correctas (x, y) sin invertir
        agent_pos = env.agent_rl.pos
        bot_pos = env.bot.pos
        
        agent_dot.set_data(agent_pos[0], agent_pos[1])
        bot_dot.set_data(bot_pos[0], bot_pos[1])

        # Actualizar rastro de movimiento
        history_agent.append(agent_pos.copy())
        history_bot.append(bot_pos.copy())

        if len(history_agent) > 50:
            history_agent.pop(0)
            history_bot.pop(0)

        trail_agent.set_data(*zip(*history_agent))
        trail_bot.set_data(*zip(*history_bot))

        return agent_dot, bot_dot, trail_agent, trail_bot

    # Iniciar animación
    ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=100)
    plt.tight_layout()
    plt.show()
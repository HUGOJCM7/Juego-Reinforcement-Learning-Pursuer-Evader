import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import TD3
import warnings
import os

# Oculta las advertencias de Gym/Gymnasium para una salida limpia
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from chase_env_sb3 import EvadeEnv, PursueEnv

# 1. Configura el backend de Matplotlib para asegurar que la ventana se muestre.
import matplotlib
matplotlib.use('TkAgg')

# --- 2. Configuración de Modelos y Visualización ---
# Elige qué modelo quieres visualizar.
# Opciones: "evader", "pursuer", "evader_vs_pursuer", "none"
VISUALIZE_MODEL = "evader_vs_pursuer"  

# Ruta dinámica a la carpeta de modelos
LOCAL_MODELS_DIR = './models'
DRIVE_MODELS_DIR = '/content/drive/MyDrive/saved_models'

if os.path.exists(LOCAL_MODELS_DIR):
    MODELS_DIR = LOCAL_MODELS_DIR
    print(f"📁 Usando modelos desde: {MODELS_DIR}")
elif os.path.exists(DRIVE_MODELS_DIR):
    MODELS_DIR = DRIVE_MODELS_DIR
    print(f"📁 Usando modelos desde Google Drive: {MODELS_DIR}")
else:
    raise FileNotFoundError("❌ No se encontraron modelos ni en './models' ni en Google Drive.")

# Rutas completas a los modelos entrenados
EVADER_BEST_MODEL_PATH = os.path.join(MODELS_DIR, "evader_model_best", "best_model.zip")
PURSUER_BEST_MODEL_PATH = os.path.join(MODELS_DIR, "pursuer_model_best", "best_model.zip")

class ChaseVisualizer:
    """Clase para manejar la visualización de la simulación de persecución con Matplotlib."""
    def __init__(self, env: EvadeEnv | PursueEnv, rl_model: TD3 = None):
        self.env = env
        self.rl_model = rl_model
        self.bounds = self.env.bounds
        self.history_rl_agent = []
        self.history_bot = []
        
        self.rl_agent_color = self.env.agent_rl.color
        self.bot_color = self.env.bot.color

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self._setup_plot()
        self._setup_artists()
    
    def _setup_plot(self):
        """Configura los límites y el estilo del gráfico."""
        self.ax.set_xlim(self.bounds)
        self.ax.set_ylim(self.bounds)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("Simulación de Persecución")
        self.ax.set_xlabel("Posición X")
        self.ax.set_ylabel("Posición Y")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.set_facecolor('#f0f0f0')

    def _setup_artists(self):
        """Inicializa todos los objetos gráficos para la animación."""
        self.rl_agent_circle = plt.Circle((0, 0), self.env.agent_rl.radius, color=self.rl_agent_color, zorder=4)
        self.bot_circle = plt.Circle((0, 0), self.env.bot.radius, color=self.bot_color, zorder=4)
        
        self.ax.add_patch(self.rl_agent_circle)
        self.ax.add_patch(self.bot_circle)
        
        self.vel_rl_agent_arrow = self.ax.quiver(0, 0, 0, 0, color=self.rl_agent_color, angles='xy', scale_units='xy', scale=1.0, zorder=5, width=0.005, headwidth=5)
        self.vel_bot_arrow = self.ax.quiver(0, 0, 0, 0, color=self.bot_color, angles='xy', scale_units='xy', scale=1.0, zorder=5, width=0.005, headwidth=5)

        self.trail_rl_agent, = self.ax.plot([], [], color=self.rl_agent_color, linestyle='--', linewidth=1, alpha=0.7)
        self.trail_bot, = self.ax.plot([], [], color=self.bot_color, linestyle='--', linewidth=1, alpha=0.7)
        
        self.info_text = self.ax.text(-1.05, 1.05, "", fontsize=10, ha='left', va='top')
        
        self.ax.legend([self.rl_agent_circle, self.bot_circle], 
                       [f"Agente (RL): {self.env.role.capitalize()}", f"Bot: {('pursuer' if self.env.role == 'evader' else 'evader').capitalize()}"])
        
    def reset(self):
        """Reinicia la visualización para un nuevo episodio."""
        self.env.reset()
        self.history_rl_agent.clear()
        self.history_bot.clear()

    def update(self, frame):
        """Función de actualización principal para la animación."""
        if self.rl_model:
            obs_rl_agent = self.env.unwrapped._get_obs() # El entorno Vectorizado necesita .unwrapped
            action, _ = self.rl_model.predict(obs_rl_agent, deterministic=True)
        else:
            action = self.env.action_space.sample()

        obs, reward, terminated, truncated, _ = self.env.step(action)
        
        # El entorno Vectorizado devuelve un array de 1 elemento, lo desempaquetamos
        rl_pos = self.env.unwrapped.agent_rl.pos
        bot_pos = self.env.unwrapped.bot.pos
        
        self.rl_agent_circle.set_center(rl_pos)
        self.bot_circle.set_center(bot_pos)
        
        self.vel_rl_agent_arrow.set_offsets(rl_pos)
        self.vel_rl_agent_arrow.set_UVC(self.env.unwrapped.agent_rl.vel[0], self.env.unwrapped.agent_rl.vel[1])
        
        self.vel_bot_arrow.set_offsets(bot_pos)
        self.vel_bot_arrow.set_UVC(self.env.unwrapped.bot.vel[0], self.env.unwrapped.bot.vel[1])
        
        self.history_rl_agent.append(rl_pos.copy())
        self.history_bot.append(bot_pos.copy())
        
        if len(self.history_rl_agent) > 50:
            self.history_rl_agent.pop(0)
            self.history_bot.pop(0)
        
        self.trail_rl_agent.set_data(*zip(*self.history_rl_agent))
        self.trail_bot.set_data(*zip(*self.history_bot))
        
        dist = np.linalg.norm(rl_pos - bot_pos)
        status_text = "Capturado!" if terminated and dist < (self.env.unwrapped.agent_rl.radius + self.env.unwrapped.bot.radius) else "Tiempo límite" if terminated else "Corriendo"
        
        self.info_text.set_text(
            f"Paso: {self.env.unwrapped.step_count}\n"
            f"Distancia: {dist:.3f}\n"
            f"Recompensa: {reward:.2f}\n"
            f"Estado: {status_text}"
        )

        if terminated or truncated:
            print(f"Episodio terminado en el paso {self.env.unwrapped.step_count}. Recompensa total: {reward:.2f}")
            self.reset()

        return (self.rl_agent_circle, self.bot_circle, self.vel_rl_agent_arrow, 
                self.vel_bot_arrow, self.trail_rl_agent, self.trail_bot, self.info_text)

# --- 5. Creación y Visualización de la Animación ---
if __name__ == '__main__':
    rl_model = None
    env = None
    
    try:
        if VISUALIZE_MODEL == "evader":
            env = EvadeEnv(render_mode="human")
            rl_model = TD3.load(EVADER_BEST_MODEL_PATH, env=env)
            print("Cargado modelo del evasor.")
        elif VISUALIZE_MODEL == "pursuer":
            env = PursueEnv(render_mode="human")
            rl_model = TD3.load(PURSUER_BEST_MODEL_PATH, env=env)
            print("Cargado modelo del perseguidor.")
        elif VISUALIZE_MODEL == "evader_vs_pursuer":
            evader_model = TD3.load(EVADER_BEST_MODEL_PATH)
            env = PursueEnv(bot_model=evader_model, render_mode="human")
            rl_model = TD3.load(PURSUER_BEST_MODEL_PATH, env=env)
            print("Visualizando perseguidor entrenado contra evasor entrenado.")
        else: # Por defecto o "none"
            env = EvadeEnv(render_mode="human")
            print("No se cargará ningún modelo entrenado, visualizando agente aleatorio.")
    
    except FileNotFoundError as e:
        print(f"\n¡ERROR! No se pudo cargar el modelo. Asegúrate de que existe el archivo: {e}")
        print("Visualizando agente aleatorio en su lugar.")
        env = EvadeEnv(render_mode="human")
        rl_model = None
        
    # Asegurarse de que el entorno se ha inicializado
    if env is None:
        print("Error al inicializar el entorno. Saliendo.")
    else:
        visualizer = ChaseVisualizer(env, rl_model)
        ani = FuncAnimation(visualizer.fig, visualizer.update, interval=50, blit=False)
        plt.tight_layout()
        plt.show()
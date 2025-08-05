import numpy as np
import os
import warnings

# Oculta las advertencias de Gym/Gymnasium para una salida limpia
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from stable_baselines3 import TD3
from stable_baselines3.td3 import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Importamos tu entorno desde el archivo
from chase_env_sb3 import EvadeEnv, PursueEnv

# --- Configuración del entrenamiento ---
NUM_ROUNDS = 15*9
TIMESTEPS_PER_ROUND = 50_000 

# Ruta base automática relativa al archivo actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

EVADER_SAVE_PATH = os.path.join(MODELS_DIR, "evader_model.zip")
PURSUER_SAVE_PATH = os.path.join(MODELS_DIR, "pursuer_model.zip")
EVADER_BEST_MODEL_PATH = os.path.join(MODELS_DIR, "evader_model_best")
PURSUER_BEST_MODEL_PATH = os.path.join(MODELS_DIR, "pursuer_model_best")

os.makedirs(MODELS_DIR, exist_ok=True)

def load_or_create_model(env, main_path, best_model_path, params):
    """Carga el mejor modelo si existe, si no, crea uno nuevo."""
    best_model_file = os.path.join(best_model_path, 'best_model.zip')
    
    if os.path.exists(best_model_file):
        try:
            model = TD3.load(best_model_file, env=env)
            print(f"✅ Modelo 'best_model' cargado desde {best_model_file}")
            return model
        except Exception as e:
            print(f"Error al cargar el mejor modelo: {e}. Creando uno nuevo...")
    
    # Si el mejor modelo no existe o falló la carga, intentar cargar el último modelo
    if os.path.exists(main_path):
        try:
            model = TD3.load(main_path, env=env)
            print(f"✅ Modelo principal cargado desde {main_path}")
            return model
        except Exception as e:
            print(f"Error al cargar el modelo principal: {e}. Creando uno nuevo...")
    
    # Si nada funciona, crear uno nuevo
    model = TD3(env=env, **params)
    print(f"➕ Creando nuevo modelo.")
    return model

# --- Inicialización del entorno y los modelos ---
evader_env_base = DummyVecEnv([lambda: EvadeEnv()])
pursuer_env_base = DummyVecEnv([lambda: PursueEnv()])

# Hiperparámetros de TD3
td3_params = {
    "policy": MlpPolicy,
    "verbose": 1,
    "learning_rate": 1e-4, 
    "buffer_size": 100_000,
    "batch_size": 128,
    "learning_starts": 1000,
    "train_freq": (1, "step"), 
    "policy_kwargs": dict(net_arch=[128, 128]),
}

# La función ahora recibe las dos rutas para ser más robusta
evader_model = load_or_create_model(evader_env_base, EVADER_SAVE_PATH, EVADER_BEST_MODEL_PATH, td3_params)
pursuer_model = load_or_create_model(pursuer_env_base, PURSUER_SAVE_PATH, PURSUER_BEST_MODEL_PATH, td3_params)

# --- Bucle de Entrenamiento por Turnos ---
print("\n--- COMENZANDO ENTRENAMIENTO ---")
for round_num in range(1, NUM_ROUNDS + 1):
    
    # 🧠 Paso 1: Entrenar al EVASOR contra el PERSEGUIDOR
    print(f"\nRound {round_num}/{NUM_ROUNDS} - Entrenando al evasor 🧠")
    
    evader_env_vs_pursuer = DummyVecEnv([lambda: EvadeEnv(bot_model=pursuer_model)])
    evader_model.set_env(evader_env_vs_pursuer)

    eval_callback_evader = EvalCallback(
        evader_env_vs_pursuer,
        best_model_save_path=EVADER_BEST_MODEL_PATH,
        log_path=EVADER_BEST_MODEL_PATH,
        eval_freq=5000, 
        deterministic=True,
        render=False
    )

    evader_model.learn(total_timesteps=TIMESTEPS_PER_ROUND, callback=eval_callback_evader, reset_num_timesteps=False)
    evader_model.save(EVADER_SAVE_PATH) # Guardar el último modelo en la ruta principal


    # 🚓 Paso 2: Entrenar al PERSEGUIDOR contra el EVASOR
    print(f"\nRound {round_num}/{NUM_ROUNDS} - Entrenando al perseguidor 🚓")
    
    pursuer_env_vs_evader = DummyVecEnv([lambda: PursueEnv(bot_model=evader_model)])
    pursuer_model.set_env(pursuer_env_vs_evader)

    eval_callback_pursuer = EvalCallback(
        pursuer_env_vs_evader,
        best_model_save_path=PURSUER_BEST_MODEL_PATH,
        log_path=PURSUER_BEST_MODEL_PATH,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    pursuer_model.learn(total_timesteps=TIMESTEPS_PER_ROUND, callback=eval_callback_pursuer, reset_num_timesteps=False)
    pursuer_model.save(PURSUER_SAVE_PATH) # Guardar el último modelo en la ruta principal

print("\n--- ENTRENAMIENTO COMPLETADO ---")
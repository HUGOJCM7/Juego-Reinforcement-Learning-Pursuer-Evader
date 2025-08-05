import os
import warnings
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy

# Ocultar advertencias molestas
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Importar tus entornos
from chase_env_sb3 import EvadeEnv, PursueEnv

# === Configuración ===
NUM_ROUNDS = 3 # Aumentamos para un mejor entrenamiento
TIMESTEPS_PER_ROUND = 50_000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_lstm")
# Usamos el formato .zip para la carga y guardado consistente
EVADER_SAVE_PATH = os.path.join(MODELS_DIR, "evader_lstm.zip")
PURSUER_SAVE_PATH = os.path.join(MODELS_DIR, "pursuer_lstm.zip")
EVADER_BEST_PATH = os.path.join(MODELS_DIR, "evader_best")
PURSUER_BEST_PATH = os.path.join(MODELS_DIR, "pursuer_best")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_or_create_lstm_model(env, main_path, best_path, params):
    """Carga modelo recurrente o crea uno nuevo."""
    best_model_file = os.path.join(best_path, "best_model.zip")

    if os.path.exists(best_model_file):
        try:
            model = RecurrentPPO.load(best_model_file, env=env)
            print(f"✅ Modelo 'best' cargado desde {best_model_file}")
            return model
        except Exception as e:
            print(f"Error al cargar el mejor modelo: {e}")

    if os.path.exists(main_path):
        try:
            model = RecurrentPPO.load(main_path, env=env)
            print(f"✅ Modelo principal cargado desde {main_path}")
            return model
        except Exception as e:
            print(f"Error al cargar modelo: {e}")

    print("➕ Creando nuevo modelo LSTM")
    return RecurrentPPO(env=env, **params)

# === Crear entornos base ===
evader_env_base = DummyVecEnv([lambda: EvadeEnv()])
pursuer_env_base = DummyVecEnv([lambda: PursueEnv()])

# === Hiperparámetros para RecurrentPPO (LSTM) ===
lstm_params = {
    "policy": MlpLstmPolicy,
    "verbose": 1,
    "learning_rate": 1e-4,
    "n_steps": 256, # Aumentado para capturar secuencias más largas
    "batch_size": 256, # CORRECCIÓN: ahora es igual a n_steps
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "tensorboard_log": os.path.join(MODELS_DIR, "tensorboard"),
}

# === Crear o cargar modelos ===
evader_model = load_or_create_lstm_model(evader_env_base, EVADER_SAVE_PATH, EVADER_BEST_PATH, lstm_params)
pursuer_model = load_or_create_lstm_model(pursuer_env_base, PURSUER_SAVE_PATH, PURSUER_BEST_PATH, lstm_params)

# === Entrenamiento en rondas alternadas ===
print("\n--- INICIO ENTRENAMIENTO CON MEMORIA ---")
for round_num in range(1, NUM_ROUNDS + 1):
    
    # Evader contra perseguidor
    print(f"\n🔁 Ronda {round_num} - Entrenando EVASOR")
    evader_env_vs_pursuer = DummyVecEnv([lambda: EvadeEnv(bot_model=pursuer_model)])
    evader_model.set_env(evader_env_vs_pursuer)

    eval_evader = EvalCallback(
        evader_env_vs_pursuer,
        best_model_save_path=EVADER_BEST_PATH,
        log_path=EVADER_BEST_PATH,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    evader_model.learn(total_timesteps=TIMESTEPS_PER_ROUND, callback=eval_evader, reset_num_timesteps=False)
    evader_model.save(EVADER_SAVE_PATH)

    # Perseguidor contra evasor
    print(f"🔁 Ronda {round_num} - Entrenando PERSEGUIDOR")
    pursuer_env_vs_evader = DummyVecEnv([lambda: PursueEnv(bot_model=evader_model)])
    pursuer_model.set_env(pursuer_env_vs_evader)

    eval_pursuer = EvalCallback(
        pursuer_env_vs_evader,
        best_model_save_path=PURSUER_BEST_PATH,
        log_path=PURSUER_BEST_PATH,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    pursuer_model.learn(total_timesteps=TIMESTEPS_PER_ROUND, callback=eval_pursuer, reset_num_timesteps=False)
    pursuer_model.save(PURSUER_SAVE_PATH)

print("\n✅ ENTRENAMIENTO CON LSTM COMPLETADO")
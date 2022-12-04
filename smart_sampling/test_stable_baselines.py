from env import FrameEnvironment
from NExTQA import FrameLoader
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from models import StateConfig, StateModel, PredictionConfig, PredictionModel

model_config = StateConfig()
prediction_config = PredictionConfig()

state_model = StateModel(model_config)
prediction_model = PredictionModel(prediction_config)

dataset = FrameLoader()

env = FrameEnvironment(
    embedding_dim=768,
    dataset=dataset,
    state_model=state_model,
    prediction_model=prediction_model,
    normalization_factor=0.5,
    train=True,
    dense_reward=True,
)

check_env(env)

make_env = lambda :  FrameEnvironment(
    embedding_dim=768,
    dataset=dataset,
    state_model=state_model,
    prediction_model=prediction_model,
    normalization_factor=0.5,
    train=True,
    dense_reward=True,
)

env = make_vec_env(make_env, n_envs=4)
model = A2C("MlpPolicy", env).learn(total_timesteps=10)

print("\n\nSuccess!\n\n")
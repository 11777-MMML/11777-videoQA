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
    normalization_factor=0.5
)

check_env(env)
model = A2C("MlpPolicy", env).learn(total_timesteps=10)

print("\n\nSuccess!\n\n")
from env import FrameEnvironment
from NExTQA import FrameLoader
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

start_state = env.reset()
print("start state:", start_state.shape)
next_state, reward, done, _ = env.step(0)
print("next state:", next_state.shape)

count = 0
while not done:
    action = count % 2
    next_state, reward, done, _ = env.step(action)
    print(f"trajectory {count}: {next_state.shape}, {reward}, {done}")
    count += 1


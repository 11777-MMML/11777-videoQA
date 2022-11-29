from env import FrameEnvironment
from NExTQA import FrameLoader
from models import ModelConfig, TransformerModel, PredictionConfig, PredictionModel

model_config = ModelConfig()
prediction_config = PredictionConfig()

state_model = TransformerModel(model_config)
prediction_model = PredictionModel(prediction_config)

dataset = FrameLoader()

env = FrameEnvironment(
    embedding_dim=768,
    dataset=dataset,
    state_model=state_model,
    prediction_model=prediction_model,
    normalization_factor=0.5
)

start_state = env.reset()
print(start_state)
next_state, reward, done, _ = env.step(0)
print(next_state)

count = 0
while not done:
    action = count % 2
    count += 1
    next_state, reward, done, _ = env.step(action)


import torch
from env import FrameEnvironment
from NExTQA import FrameLoader
from models import StateConfig, StateModel, PredictionConfig, PredictionModel, AgentConfig, Agent

torch.manual_seed(0)

model_config = StateConfig()
prediction_config = PredictionConfig()
agent_config = AgentConfig()

state_model = StateModel(model_config)
prediction_model = PredictionModel(prediction_config)
agent = Agent(agent_config)

dataset = FrameLoader()

env = FrameEnvironment(
    embedding_dim=768,
    dataset=dataset,
    state_model=state_model,
    prediction_model=prediction_model,
    normalization_factor=0.5
)

state = env.reset()
print(f"start_state:{state}")

done = False

while not done:
    action_logits = agent(state)
    action = torch.argmax(action_logits).item()
    print(f"action: {action} input_state: {state.shape}")
    state, reward, done, _ = env.step(action)

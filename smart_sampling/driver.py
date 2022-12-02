from env import FrameEnvironment
from NExTQA import FrameLoader
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from models import StateConfig, StateModel, PredictionConfig, PredictionModel

STEPS = 10000
EPOCHS = 10
EVAL_EPISODES=1000

class TensorBoardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorBoardCallback, self).__init__(verbose)
        self.total_examples = 0
        self.total_correct = 0
        
    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        base_env = self.training_env.envs[0].env
        is_correct = base_env._is_correct
        prediction_error = base_env._prediction_error
        buffer_size = len(base_env._buffer)
        num_frames = base_env._max_step

        if is_correct == 0:
            self.total_correct += 1
        self.total_examples += 1

        self.logger.record("prediction_error", prediction_error)
        self.logger.record("frame_ratio", buffer_size/num_frames)
        self.logger.record("buffer_size", buffer_size)
        self.logger.record("num_frames", num_frames)

    def _on_training_end(self):
        self.logger.record("total_examples", self.total_examples)
        self.logger.record("total_correct", self.total_correct)
        self.logger.record("Accuracy", self.total_correct/self.total_examples)

        self.total_examples = 0
        self.total_correct = 0

model_config = StateConfig()
prediction_config = PredictionConfig()

state_model = StateModel(model_config)
prediction_model = PredictionModel(prediction_config)

train_dataset = FrameLoader(mode="training")
val_dataset = FrameLoader(mode="validation")

train_env = FrameEnvironment(
    embedding_dim=768,
    dataset=train_dataset,
    state_model=state_model,
    prediction_model=prediction_model,
    normalization_factor=0.5
)

train_env = Monitor(train_env)


val_env = FrameEnvironment(
    embedding_dim=768,
    dataset=val_dataset,
    state_model=state_model,
    prediction_model=prediction_model,
    normalization_factor=0.5
)

val_env = Monitor(val_env)

log_path = "./logs/"
logger = configure(
    log_path,
    ["stdout", "csv", "tensorboard", "log"]
)

model = A2C("MlpPolicy", train_env, verbose=1)
model.set_logger(logger)

for epoch in range(EPOCHS):
    print("TRAINING!")
    model.learn(total_timesteps=STEPS, tb_log_name="training_run", callback=TensorBoardCallback())
    
    # print("EVALUATING!")
    # mean_reward, std_reward = evaluate_policy(model, val_env, n_eval_episodes=EVAL_EPISODES)
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
print("\n\nSuccess!\n\n")
from stable_baselines3 import A2C
from env import FrameEnvironment
from NExTQA import FrameLoader
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.base_class import BaseAlgorithm
from models import StateConfig, StateModel, PredictionConfig, PredictionModel
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
def eval(num_iters: int, model: BaseAlgorithm, eval_env: DummyVecEnv):
    old_env = model.get_env()
    model.set_env(eval_env)
    total_correct = 0
    total_examples = 0

    for _ in range(num_iters):
        obs = model.env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = model.env.step(action)

            if done:
                base_env = model.env.envs[0].env
                total_examples += 1
                
                if base_env._is_correct:
                    total_correct += 1
    
    print("Accuracy:", total_correct / total_examples)
    model.logger.record("Validation/Accuracy", total_correct/total_examples)
    model.set_env(old_env)

if __name__ == '__main__':

    model_config = StateConfig()
    prediction_config = PredictionConfig()

    state_model = StateModel(model_config)
    prediction_model = PredictionModel(prediction_config)

    val_dataset = FrameLoader(mode="validation")

    val_env = FrameEnvironment(
    embedding_dim=768,
    dataset=val_dataset,
    state_model=state_model,
    prediction_model=prediction_model,
    normalization_factor=0.5
    )

    val_env = Monitor(val_env)

    make_env = lambda : val_env
    val_env = DummyVecEnv([make_env])
    model = A2C("MlpPolicy", val_env, verbose=1)

    eval(1, model, val_env)
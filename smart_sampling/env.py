from gym import Env, spaces
from torch.utils.data import Dataset
from torch import randint, nn, concat, square, sum, Tensor, float32

class FrameEnvironment(Env):
    def __init__(self, embedding_dim: int, dataset: Dataset, state_model: nn.Module, prediction_model: nn.Module, normalization_factor: float):
        self(FrameEnvironment, self).__init__()

        # Add state to buffer
        self.action_space = spaces.Discrete(2)
        
        # Overall representation of the frames accumulated so far
        self.embedding_dim = embedding_dim

        # Give aggregation of previous frames back
        self.observation_space = spaces.Box(embedding_dim)
    
        # Source of Frames
        self.dataset = dataset

        # Representation Model
        self.state_model = state_model

        # Prediction Model
        self.prediction_model = prediction_model

        # Normalization factor
        self.normalization_factor = normalization_factor

        # Internal book keeping
        self._frames = None
        self._buffer = []
        self._curr_step = None
        self._max_step = None
        self._curr_obs = None
    
    def reset(self):
        low = 0
        high = len(self.dataset)
        index = randint(low=low, high=high)

        self._frames = self.dataset[index]
        self._curr_step = 0
        self._max_step = len(self._frames)

        self._curr_obs = self._frames[self._curr_step]
        rep = self.state_model(self._curr_obs)
        return rep

    def _reward(self, state_input: Tensor, curr_step: int) -> float32:
        next_obs_pred = self.prediction_model(state_input)
        next_frame = curr_step + 1
        prediction_error = sum(square(next_obs_pred - next_frame))

        buffer_size = len(self._buffer)
        frame_penalty = (buffer_size/self._max_step)

        reward = self.normalization_factor * prediction_error + (1 - self.normalization_factor) * frame_penalty

        # The higher this value, less is the reward
        return -reward
    
    def step(self, action):
        # If the frame is selected
        if action > 0:
            self._buffer.append(self._curr_obs.unsqueeze())

        # Find the latest representation
        state_input = concat(tensors=self._buffer, dim=0)
        rep = self.model(state_input)

        # calculate the reward
        reward = self._reward(state_input, self._curr_step)

        # Go to the next frame
        self._curr_step += 1

        done = False
        if self._curr_step == self._max_step:
            done = True
        else:
            self._curr_obs = self._frame[self._curr_step]

        # TODO: Add infors for debugging
        return rep, reward, done, {}
            


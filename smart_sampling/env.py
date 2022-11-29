from gym import Env, spaces
from torch.utils.data import Dataset
from transformers import DistilBertModel, DistilBertTokenizer
from torch import randint, nn, concat, square, sum, Tensor, float32, no_grad, device, cuda, argmax

BERT_CHECKPOINT = "distilbert-base-uncased"

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

        # Question Answering
        self._question = None
        self._candidates = []
        self._answer = None

        # Can be removed if embeddings are saved
        # Language Model
        self._tokenizer = DistilBertTokenizer.from_pretrained(BERT_CHECKPOINT)
        self._language_model = DistilBertModel.from_pretrained(BERT_CHECKPOINT)
        self._language_model.eval()

        # Cuda
        self._device = device("cuda" if cuda.is_available() else "cpu")

        self.state_model.to(self._device)
        self.prediction_model.to(self._device)
        self._language_model.to(self._device)

    def reset(self):
        low = 0
        high = len(self.dataset)
        index = randint(low=low, high=high)

        example = self.dataset[index]

        question = example["question"]
        
        a0 = example["a0"]
        a1 = example["a1"]
        a2 = example["a2"]
        a3 = example["a3"]
        a4 = example["a4"]

        answer = example["answer"]

        self._question = question
        self._candidates = [a0, a1, a2, a3, a4]
        self._answer = answer 

        video_feature = example["video_feature"]

        # N, D
        self._frames = video_feature
        self._curr_step = 0
        self._max_step = len(self._frames)

        with no_grad():
            inputs = self._tokenizer(question, return_tensors="pt").to(self._device)
            question_rep = self._language_model(**inputs)
            question_rep = question_rep.squeeze()
            question_rep = question_rep.detach().cpu()

            # Take CLS Token
            self._buffer = [question_rep[0]]

        self._curr_obs = self._frames[self._curr_step]
        rep = self.state_model(self._buffer[0])
        return rep

    # def _reward(self, state_input: Tensor, curr_step: int) -> float32:
    #     next_obs_pred = self.prediction_model(state_input)
    #     next_frame = curr_step + 1
    #     prediction_error = sum(square(next_obs_pred - next_frame))

    #     buffer_size = len(self._buffer)
    #     frame_penalty = (buffer_size/self._max_step)

    #     reward = self.normalization_factor * prediction_error + (1 - self.normalization_factor) * frame_penalty

    #     # The higher this value, less is the reward
    #     return -reward
    
    def _sparse_reward(self, state_input: Tensor) -> float32:

        with no_grad():
            inputs = self._tokenizer(self._question, return_tensors="pt").to(self._device)
            question_rep = self._language_model(**inputs)
            question_rep = question_rep.squeeze()
            question_rep = question_rep.detach().cpu()
            question_rep = question_rep[0]

            candidates = []

            for candidate in self._candidates:
                inputs = self._tokenizer(candidate, return_tensors="pt").to(self._device)
                candidate_rep = self._language_model(**inputs)
                candidate_rep = candidate_rep.squeeze()
                candidate_rep = candidate_rep[0]
                candidates.append = candidate_rep.detach().cpu()
        
        logits = self.prediction_model(self._buffer, [question_rep], candidates)

        # Penalty if we get it wrong
        prediction_error = argmax(logits) != self._answer
        buffer_size = len(self._buffer)
        frame_penalty = (buffer_size/self._max_step)

        penalty = self.normalization_factor * prediction_error + (1 - self.normalization_factor) * frame_penalty

        return -penalty
    
    def _dense_reward(self, state_input:Tensor, curr_step: int) -> float32:
        return 0
    
    def step(self, action):
        # If the frame is selected
        if action > 0:
            self._buffer.append(self._curr_obs.unsqueeze())

        # Find the latest representation
        temp_buffer = self._buffer + [self._curr_obs]
        state_input = concat(tensors=temp_buffer, dim=0)
        rep = self.model(state_input)

        # calculate the reward
        reward = self._dense_reward(state_input, self._curr_step)

        # Go to the next frame
        self._curr_step += 1

        done = False
        if self._curr_step == self._max_step:
            done = True
            reward += self._sparse_reward(state_input)
        else:
            self._curr_obs = self._frame[self._curr_step]

        # TODO: Add infos for debugging
        return rep, reward, done, {}
            


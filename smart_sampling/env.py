import numpy as np
from gym import Env, spaces
from torch.utils.data import Dataset
from transformers import DistilBertModel, DistilBertTokenizer
from torch import randint, nn, float32, no_grad, device, cuda, argmax, optim, zeros_like, Tensor, hstack

BERT_CHECKPOINT = "distilbert-base-uncased"

class FrameEnvironment(Env):
    def __init__(self, embedding_dim: int, dataset: Dataset, state_model: nn.Module, prediction_model: nn.Module, normalization_factor: float, train: bool, dense_reward: bool):
        super().__init__()

        # Add state to buffer
        self.action_space = spaces.Discrete(2)
        
        # Overall representation of the frames accumulated so far
        self.embedding_dim = embedding_dim

        # Give aggregation of previous frames back
        self.observation_space = spaces.Box(shape=(embedding_dim,), low=-np.inf, high=np.inf)
    
        # Source of Frames
        self.dataset = dataset

        # Representation Model
        self.state_model = state_model

        # Prediction Model
        self.prediction_model = prediction_model
        self.prediction_optimizer = optim.AdamW(self.prediction_model.parameters(), lr=0.0003)
        self.prediction_cross_entropy_loss = nn.CrossEntropyLoss()

        # Normalization factor
        self.normalization_factor = normalization_factor

        # Internal book keeping
        self._frames = None
        self._buffer = []
        self._curr_step = None
        self._max_step = None
        self._curr_obs = None

        # Performance
        self._prediction_error = 0
        self._next_frame_prediction_error = 0

        # Question Answering
        self._question = None
        self._candidates = []
        self._answer = None
        self._is_correct = False

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

        # Training
        self._train = train

        # Dense Reward
        self._has_dense_reward = dense_reward
        self.classifier = nn.Linear(
            in_features=2 * self.embedding_dim, 
            out_features=1,
            bias=True,
            device=self._device,
        )

        self.state_optimizer = optim.AdamW(self.state_model.parameters(), lr=0.0003)
        self.state_cross_entropy_loss = nn.BCEWithLogitsLoss()
    
    def get_representation_from_lm(self, text_input):
        with no_grad():
            inputs = self._tokenizer(text_input, return_tensors="pt").to(self._device)
            outputs = self._language_model(**inputs)
            rep = outputs.last_hidden_state
            rep = rep.squeeze()
            rep = rep.detach().cpu()

        return rep[0]

    def reset(self):
        low = 0
        high = len(self.dataset)
        success = False

        while not success:
            try:
                index = randint(low=low, high=high, size=(1,)).item()
                example = self.dataset[index]
                success = True
            except:
                success = False
        
        question = example["question"]
        
        a0 = example["a0"]
        a1 = example["a1"]
        a2 = example["a2"]
        a3 = example["a3"]
        a4 = example["a4"]

        answer = example["answer"]

        self._question = question
        self._candidates = [a0, a1, a2, a3, a4]
        self._answer = int(answer)

        video_feature = example["video_feature"]

        # N, D
        self._frames = video_feature
        self._curr_step = 0
        self._max_step = len(self._frames)

        question_rep = self.get_representation_from_lm(question)
        question_rep = question_rep.unsqueeze(0)

        # Take CLS Token
        self._buffer = [question_rep]

        # First decision should look at the first frame
        self._curr_obs = self._frames[self._curr_step]
        temp_buffer = self._buffer + [self._curr_obs.unsqueeze(0)]
        rep = self.state_model(temp_buffer)
        return rep[-1].cpu().detach().numpy()
    
    def _calculate_final_reward(self, question_rep, candidates):
        logits = self.prediction_model(self._buffer, [question_rep], candidates)

        y_gt = zeros_like(logits)
        y_gt[self._answer] = 1
        prediction_error = self.prediction_cross_entropy_loss(logits, y_gt)
        self._is_correct = self._answer == argmax(logits)

        return prediction_error

    def _sparse_reward(self) -> float32:

        question_rep = self.get_representation_from_lm(self._question)
        question_rep = question_rep.unsqueeze(0)

        candidates = []

        for candidate in self._candidates:
            candidate_rep = self.get_representation_from_lm(candidate)
            candidate_rep = candidate_rep.unsqueeze(0)
            candidates.append(candidate_rep)
        
        if not self._train:
            with no_grad():
                self.prediction_model.eval()
                prediction_error = self._calculate_final_reward(question_rep, candidates)
        else:
            self.prediction_model.train()
            prediction_error = self._calculate_final_reward(question_rep, candidates)
            self.prediction_optimizer.zero_grad()
            prediction_error.backward()
            self.prediction_optimizer.step()

        prediction_error = prediction_error.detach().cpu().numpy()
        self._prediction_error = prediction_error
        buffer_size = len(self._buffer)
        frame_penalty = (buffer_size/self._max_step)

        penalty = self.normalization_factor * prediction_error + (1 - self.normalization_factor) * frame_penalty

        return -penalty
    
    def _calculate_step_reward(self, input_tensor, is_next_frame):
        logits = self.classifier(input_tensor)
        next_frame_prediction_error = self.state_cross_entropy_loss(logits, is_next_frame)

        return next_frame_prediction_error

    def _dense_reward(self, curr_step: int, curr_rep: Tensor) -> float32:
        if curr_step + 1 < self._max_step and self._has_dense_reward:
            frame = None
            is_next_frame = randint(low=0, high=2, size=(1,))
            
            if is_next_frame.item() == 1:
                frame = self._frames[curr_step + 1]
            else:
                index = randint(low=curr_step+1, high=self._max_step, size=(1,)).item()
                frame = self._frames[index]
            
            frame = frame.to(self._device)

            is_next_frame = is_next_frame.to(self._device)
            is_next_frame = is_next_frame.unsqueeze(0)
            is_next_frame = is_next_frame.to(float32)
            
            curr_rep = curr_rep.unsqueeze(0)
            frame = frame.unsqueeze(0)

            input_tensor = hstack(tensors=(curr_rep, frame))

            if not self._train:
                with no_grad():
                    self.classifier.eval()
                    next_frame_prediction_error = self._calculate_step_reward(input_tensor, is_next_frame)
            else:
                next_frame_prediction_error = self._calculate_step_reward(input_tensor, is_next_frame)
                self.state_optimizer.zero_grad()
                next_frame_prediction_error.backward()
                self.state_optimizer.step()
            
            next_frame_prediction_error = next_frame_prediction_error.detach().cpu().numpy().item()
            self._next_frame_prediction_error = next_frame_prediction_error

            return -self._next_frame_prediction_error
                
        return 0
    
    def step(self, action):
        # If the frame is selected
        if action > 0:
            self._buffer.append(self._curr_obs.unsqueeze(0))

        # Go to the next frame
        self._curr_step += 1

        done = False
        reward = 0

        if self._curr_step == self._max_step:
            done = True
            reward = self._sparse_reward()
            rep = self.state_model(self._buffer)
        else:
            # For the nth step look at n + 1 frames
            self._curr_obs = self._frames[self._curr_step]
            temp_buffer = self._buffer + [self._curr_obs.unsqueeze(0)]

            if not self._train or not self._has_dense_reward:
                with no_grad():
                    self.state_model.eval()
                    rep = self.state_model(temp_buffer)
            else:
                rep = self.state_model(temp_buffer)

            # calculate the reward
            reward = self._dense_reward(self._curr_step, rep[-1])

        # TODO: Add infos for debugging
        return rep[-1].cpu().detach().numpy(), reward, done, {}
            


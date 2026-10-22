import numpy as np
import json
from pathlib import Path


SIGNATURE_BEAT_COUNTS = {
    "2/4": 2,
    "2/8": 2,
    "3/4": 3,
    "3/8": 3,
    "4/4": 4,
    "4/8": 4,
    "5/4": 5,
    "5/8": 5,
    "6/4": 6,
    "6/8": 6,
    "7/4": 7,
    "7/8": 7,
    "8/4": 8,
    "8/8": 8,
    "9/4": 9,
    "9/8": 9,
    "13/16": 13,
}

BEAT_COUNTS = []
for signature in SIGNATURE_BEAT_COUNTS:
    beat_count = SIGNATURE_BEAT_COUNTS[signature]
    if beat_count not in BEAT_COUNTS:
        BEAT_COUNTS.append(beat_count)
BEAT_COUNTS.sort()
STATES = []
DOWNBEAT_STATES = []

for beat_count in BEAT_COUNTS:
    for position in range(1, beat_count + 1):
        if position == 1:
            DOWNBEAT_STATES.append(len(STATES))
        STATES.append((beat_count, position))

MODEL_PATH = Path(__file__).resolve().parent / "uncommon_model.json"


def get_beat_family(beat_count):
    family = beat_count
    while family % 2 == 0:
        family = family // 2
    return family


def build_transition_matrix(beat_transition_probs):
    transition_matrix = np.zeros((len(STATES), len(STATES)))

    for state_index in range(len(STATES)):
        beat_count, position = STATES[state_index]

        if position < beat_count:
            transition_matrix[state_index, state_index + 1] = 1
        else:
            source = BEAT_COUNTS.index(beat_count)
            for destination in range(len(BEAT_COUNTS)):
                next_state = DOWNBEAT_STATES[destination]
                transition_matrix[state_index, next_state] = beat_transition_probs[source, destination]

    return transition_matrix


def load_model(model_path=MODEL_PATH):
    with open(model_path, encoding="utf-8") as file:
        model = json.load(file)

    transition_matrix = np.array(model["transition_matrix"])
    initial_probs = np.array(model["initial_probs"])
    return transition_matrix, initial_probs


def build_observation_scores(downbeat_probs):
    n_beats = len(downbeat_probs)
    n_states = len(STATES)
    observation_scores = np.zeros((n_beats, n_states))

    for beat_index in range(n_beats):
        p = downbeat_probs[beat_index]

        for state_index in range(n_states):
            if state_index in DOWNBEAT_STATES:
                observation_scores[beat_index, state_index] = p
            else:
                observation_scores[beat_index, state_index] = 1 - p

    return observation_scores


def initialize_viterbi(observation_scores, initial_probs):
    n_beats = len(observation_scores)
    n_states = len(initial_probs)

    scores = np.full((n_beats, n_states), -np.inf)
    previous_states = np.full((n_beats, n_states), -1, dtype=int)

    if n_beats > 0:
        for state_index in range(n_states):
            p_initial = initial_probs[state_index]
            p_observation = observation_scores[0, state_index]

            if p_initial > 0 and p_observation > 0:
                scores[0, state_index] = np.log(p_initial) + np.log(p_observation)

    return scores, previous_states


def viterbi(observation_scores, transition_matrix, initial_probs):
    n_beats = len(observation_scores)
    n_states = len(initial_probs)
    scores, previous_states = initialize_viterbi(observation_scores, initial_probs)

    if n_beats == 0:
        return np.array([], dtype=int)

    log_transitions = np.full((n_states, n_states), -np.inf)

    for previous_state in range(n_states):
        for state_index in range(n_states):
            p_transition = transition_matrix[previous_state, state_index]
            if p_transition > 0:
                log_transitions[previous_state, state_index] = np.log(p_transition)

    for beat_index in range(1, n_beats):
        for state_index in range(n_states):
            p_observation = observation_scores[beat_index, state_index]

            if p_observation > 0:
                best_score = -np.inf
                best_previous_state = -1

                for previous_state in range(n_states):
                    candidate_score = (
                        scores[beat_index - 1, previous_state]
                        + log_transitions[previous_state, state_index]
                    )

                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_previous_state = previous_state

                scores[beat_index, state_index] = best_score + np.log(p_observation)
                previous_states[beat_index, state_index] = best_previous_state

    last_state = int(np.argmax(scores[n_beats - 1]))

    if scores[n_beats - 1, last_state] == -np.inf:
        return None

    state_path = np.zeros(n_beats, dtype=int)
    state_path[n_beats - 1] = last_state

    for beat_index in range(n_beats - 1, 0, -1):
        state_index = state_path[beat_index]
        state_path[beat_index - 1] = previous_states[beat_index, state_index]

    return state_path


A = None
pi = None

if MODEL_PATH.is_file():
    A, pi = load_model()

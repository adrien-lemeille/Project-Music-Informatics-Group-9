from hmm import STATES, load_model, build_observation_scores, viterbi

A, pi = load_model()

probas = (
    [0.99, 0.01, 0.01, 0.01] * 4
    + [0.99, 0.01, 0.01] * 5
    + [0.99]
)

observations = build_observation_scores(probas)
chemin = viterbi(observations, A, pi)

for indice in chemin:
    print(STATES[indice])
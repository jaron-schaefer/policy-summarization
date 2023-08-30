# policy-summarization

Code for "Linking Human Reconstruction Models for Policy Summarizations to Task Structure" (2023).
The write-up is in `docs/report.pdf`.

The idea: you explain an agent's behaviour by showing a few example state-action pairs (a policy summary), and the
question is how people generalise from those. I compared two models of that, IRL (machine teaching + max-ent IRL)
and imitation learning (active learning + Gaussian random fields), in a gridworld task and looked at how the task
structure changes which one fits. There's a computational experiment and a small study with participants.

- `models/`: summary extraction and reconstruction models, value iteration
- `trials/`: generating the gridworlds and running the pipeline (see `trials/trial_generation/how_to_generate_new_trials.txt`)
- `experiment_code/`: the participant experiment (pygame / PyGaze, runs with the mouse by default)

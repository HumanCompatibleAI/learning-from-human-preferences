# Deep Reinforcement Learning from Human Preferences


This code is built on a reproduction of OpenAI and DeepMind's [Deep Reinforcement Learning from Human Preferences](https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/), 
done by Matthew Rahtz, and has been significantly refactored for easier plug and play use in research applications for the Center for Human Compatible Artificial Intelligence
at UC Berkeley. The code as written by Matthew Rahtz prior to refactor can be found [here](https://github.com/mrahtz/learning-from-human-preferences). 
Refactoring work was primarily completed by Cody Wild in Spring 2020. 


## Refactor

### Goals 

The goals of the original version of this codebase were to reproduce successful training of DRLHP with both synthetic and human preferences on Pong, Enduro, and a novel moving dot environment, and its API was 
primarily designed for use in that small set of reproduction cases. The goals of the refactor were: 

* To change the API from a single `run.py` file with fixed options and environments into a more modular design that can be plugged into other experimental code

* To add support for arbitrary pixel-based environments of varying sizes and numbers of channels

* To allow for easier integration with a wider range of up-to-date RL implementations, rather than having a specific version of A2C implemented within the codebase 

* To make it easy to load pre-trained policies (either from RL or imitation) to warm-start the training process, in addition to starting a policy from scratch


### Design
To these ends, the essential elements of DRLHP were refactored to exist on a gym-style Env Wrapper. This wrapper: 

* Manages creating subprocesses for both asynchronous reward predictor training and asynchronous collection of preferences over pairs of segments

* Stores observations that are passed back from the underlying environment, concatenating them into segments. When segments
reach a specified length, they are passed in a pipe to the subprocess responsible for constructing pairs of segments and requesting preferences 

* Can either return the underlying env reward or the reward given by the reward predictor, which is loaded and updated throughout the 
course of reward predictor training 

* Can be used with any gym-compatible RL algorithm, which can be passed the reward-predictor reward without needing to be specially modified



## Setup

This repo is structured as a Python package, and can be installed by running 

`python setup.py install`

If you wish to develop the package, you may wish to install it in editable mode using `-e` and
install some optional packages with the `dev` extras:

`pip install -e .[dev]`

Tests can be run with

`pytest tests/`

## Human Preferences GUI
When training with human rather than synthetic preferences, you'll see two windows: a larger one showing a pair of examples of agent
behaviour, and another smaller window showing the last full episode that the
agent played (so you can see how qualitative behaviour is changing). Enter 'L'
in the terminal to indicate that you prefer the left example; 'R' to indicate
you prefer the right example; 'E' to indicate you prefer them both equally; and
 press enter if the two clips are incomparable.



## Workflows 
Once the `drlhp` package is installed, you can import the env wrapper (the main entry point to the algorithm) using: 
`from drlhp import HumanPreferencesEnvWrapper`

To construct an DRLHP environment, call the wrapper with your chosen underlying environment as an argument. This implementation
of DRLHP is designed to work with pixel observations, so the environment you choose to wrap should either (1) return 
pixels as observations, or (2) return an observation that can be reduced to pixels (e.g. a dict space that can be indexed into). If 
your environment falls into the latter bucket, you can use `obs_transform_func` to specify a function to map between your observation
and pixels. 

The EnvWrapper in a number of different arguments, and has been designed to support a number of different workflows. 
Here is a (non-exhaustive!) set of examples of those workflows. 


### Default - Collecting human preferences & training reward
```
wrapped_env = HumanPreferencesEnvWrapper(env, 
                                         segment_length=100
                                         synthetic_preferences=False, 
                                         n_initial_training_steps=10)
```
The above code will create an environment that will spin off both preference acquisition and reward training processes, and 
will switch to returning the predicted reward as environment reward after 10 training steps of the reward model. Synthetic 
preferences is set to True by default, but here we're setting it to False to indicate a desire to run the PrefInterface GUI. 

### Collecting preferences, but not training reward
```
wrapped_env = HumanPreferencesEnvWrapper(env, 
                                         synthetic_preferences=False,
                                         collect_prefs=True, 
                                         train_reward=False, 
                                         log_dir=<log_dir>)
```
This workflow can be useful if you want to batch-collect preferences to be saved out and used in training later. When you want to 
save out preferences to a file, you can call `wrapped_env.save_prefs()`. By default, preferences will be saved to 
`train.pkl.gz` and `val.pkl.gz` within `log_dir`


### Training reward from pre-collected preferences 
```
wrapped_env = HumanPreferencesEnvWrapper(env, 
                                         prefs_dir=<prefs_dir>
                                         collect_prefs=False, 
                                         train_reward=True, 
                                         reward_predictor_ckpt_interval=10)
```

This will load a preferences database out of `prefs_dir` and use it to train a reward model, which will save a checkpoint 
every 10 training steps to `<log_dir>/reward_predictor_checkpoints/reward_predictor.ckpt`. By default, `log_dir` is set to 
`drlhp_logs`. 

### Using a pretrained reward model without additional training
```
wrapped_env = HumanPreferencesEnvWrapper(env, 
                                         pretrained_reward_predictor_dir='my_log_dir/reward_predictor_checkpoints/
                                         collect_prefs=False, 
                                         train_reward=False)
```
To confirm that the env is using the trained reward model rather than the underlying env, check that the
`using_reward_from_predictor` flag is set to True. If at some point you want to switch back to underlying environment 
reward (for example, for evaluation purposes), you can call `wrapped_env.switch_to_true_reward()` to make the switch, 
and `wrapped_env.switch_to_predicted_reward()` 


## Architecture

Other than the environment wrapper itself, there are three main components:
* The preference interface ([`pref_interface.py`](drlhp/pref_interface.py))
* The PrefBuffer and PrefDB ([`pref_db.py`](drlhp/pref_db.py))
* The reward predictor ([`reward_predictor.py`](drlhp/reward_predictor.py))

### Data Flow

Segments are built up through accumulation of environment observation frames returned from 
the underlying environment, which are concatenated into segments once they reach `segment_length`.
When a segment is finalized, it is sent to the PrefInterface (via a multiprocessing queue). 

The PrefInterface combines pairs of segments together, and solicits a preference ranking 
between them, either by using synthetic preferences, or by rendering each segment (a set of image 
frames) as a video clip to be shown to the user. After it shows the clip to the user, it 
asks through a command-line interface which clip of each pair shows more of
the kind of behaviour the user wants, looping the video until it gets a response. 

Preference labels are sent to a PrefDB (either train or validation) by means of the
 PrefBuffer, which manages directing each preference to one DB Or the other. 
The preferences within the PrefDBs are then used to train a neural network reward
 predictor to assign a high scalar reward to preferred segments, 


That network can then be used to predict rewards for future video clips by
feeding the clip in, running a forward pass to calculate the "how much the user
likes this clip" value, then normalising the result to have zero mean and
constant variance across time.

This normalised value is then used returned as a reward signal, which is passed back 
by the environment 

### Processes

This code spawns two different subprocesses, other than the master process:
* One for running the preference interface to query the user for preference.
* One for training the reward predictor 

These processes communicate via a set of queues 

* `seg_pipe`, which sends segments to the PrefInterface
* `pref_pipe`, which sends (segment, preference label) pairs from the PrefInterface to be added to the PrefDB

Some tricky aspects to this:
* Video clips must be sent from the environment wrapper to the process asking for
  preferences using a small queue; clips are dropped if the queue is full. 
  The preference interface then just gets as many clips as it can from the queue 
  in 0.5 seconds, in between asking about each pair of clips. (Pairs to show the 
  user are selected from the clip database internal to the preference interface 
  into which clips from the queue are stored.)
* Preferences must be sent from the preference interface to the PrefDB which feeds the reward
  predictor using a queue. Preferences should never be dropped, though, so the
  preference interface blocks until the preference can be added to the queue,
  and the reward predictor training process runs a background thread which
  constantly receives from the queue, storing preference in the reward
  predictor process's internal database.


## Changes to the paper's setup

For the environments originally tested in the reproduction, it turned out to be possible 
to reach the milestones in the results section above even without implementing a number 
of features described in the original paper.

* For regularisation of the reward predictor network, the paper uses dropout,
  batchnorm and an adaptive L2 regularisation scheme. Here, we only use
  dropout. (Batchnorm is also supported. L2 regularisation is not implemented.)
* In the paper's setup, the rate at which preferences are requested is
  gradually reduced over time. We just ask for preferences at a constant rate.
* The paper selects video clips to show the user based on predicted reward
  uncertainty among an ensemble of reward predictors. Early experiments
  suggested a higher chance of successful training by just selecting video
  clips randomly (also noted by the paper in some situations), so we don't do
  any ensembling. (Ensembling code *is* implemented in
  [`reward_predictor.py`](drlhp/reward_predictor.py), but we always operate with only
  a single-member ensemble, and [`pref_interface.py`](drlhp/pref_interface.py) just
  chooses segments randomly.)
* The preference for each pair of video clips is calculated based on a softmax
  over the predicted latent reward values for each clip. In the paper,
  "Rather than applying a softmax directly...we assume there is a 10% chance
  that the human responds uniformly at random. Conceptually this adjustment is
  needed because human raters have a constant probability of making an error,
  which doesn’t decay to 0 as the difference in reward difference becomes
  extreme." I wasn't sure how to implement this - at least, I couldn't see a
  way to implement it that would actually affect the gradients - so we just do
  the softmax directly.



## Code Credits
All files except HumanPreferencesEnvWrapper were predominantly written by Matthew Rahtz, with some minor changes as part of this refactor. 

# OpenWakeWord Evaluation and Adversarial Data Generation

This repository evaluates the OpenWakeWord engine's performance in creating custom wake words . This repository is the fork of synthetic speech dataset generation repository by OpenWakeWord, available at `https://github.com/dscripka/synthetic_speech_dataset_generation` and users some scripts from `https://github.com/dscripka/openWakeWord`

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Generating Adversarial Text](#generating-adversarial-text)
- [Generating Synthetic Speech](#generating-synthetic-speech)
- [Evaluating OpenWakeWord Engine](#evaluating-openwakeword-engine)
- [References](#references)

## Introduction

We focus on generating adversarial synthetic data to test the engine's robustness. Adversarial data includes words that sound phonetically similar to the wake word, which helps assess false accept and reject rates.

## Setup

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/openwakeword-evaluation.git
    cd openwakeword-evaluation
    ```

2. **Set Up Virtual Environment**:

    ```bash
    pip3 install virtualenv
    virtualenv venv
    source venv/bin/activate
    ```

3. **Install Dependencies**:

    ```bash
    pip3 install -r requirements.txt
    apt-get install espeak
    ```

4. ## Generating Adversarial Text

### Command to Run:

```bash
python generate_adversarial_text.py "hey mycroft" 1 adversarial_texts.txt
```
5. ## Generating Synthetic Speech

Command to Run:
```
python3 generate_speech.py --model VITS --input_file adversarial_texts.txt --n_speakers 100 --output_dir aout --max_per_speaker 5
```

6. ## Evaluating OpenWakeWord Engine

Command to Run:
```
python3 wakeword_test.py about
```

Results :
```
False accepts are 3.0 %
Details of false accepts:
{'aout/1d4f5a333c4c49279b59e350054da39c.wav': 5}
{'aout/bdb669a443d249408eeee25ed5005c95.wav': 2}
{'aout/8f1f817fb995479ab8389aa499159a8e.wav': 4}
{'aout/972975ef4ace498aa2d122a523c05487.wav': 5}
{'aout/e3e9868f9996451fb495194bb944d63b.wav': 3}
{'aout/8af19950c9904b8cb98b91ccdcee2785.wav': 4}
```

References

	•	OpenWakeWord GitHub Repository
	•	Synthetic Speech Dataset Generation Repository

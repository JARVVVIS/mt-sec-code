## MT-Sec Eval. Kit:

This repository contains the code for the evaluation of MT-Sec, a multi-turn secure code generation benchmark.

## Setup:

```python
pip install -r requirements.txt
```

## Run Evals:
Sample command to run the evaluation for the `editing` interaction type using the `gpt-4o` model. Check the config.py file for more options.

```python
python multi_turn_eval.py --interaction editing --model_name gpt-4o
```

Results will be saved in the `results` directory as .pkl files having the additional columns "result_capability" and "result_safety", storing result for each correctness and safety unit-tests respectively.
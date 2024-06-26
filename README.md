# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

# How to run
🦎 🦫

## Challenge 1 - 2
### New Models
If you want to add a new model to the pipeline, you will need to specify:

- model name
- model parameters
- custom datapreparation function (in \data\custom_dataprep.py)
- whether to use log transform
- model hyper-parameters for tuning with optuna

All of this info can be found in config.py.

### New evaluation metrics
If you want to add new evaluation metrics, just drop them in the 'METRICS' dictionary in config.py, after importing them from the sklearn library.

### Fitting and Testing a model
You can train/fit a model and test it using train_test.py from the terminal with the following command:
```console
python train_test.py input_file output_dir {linear,ridge,lasso,xgboost} [--debug] [-h]
```

where:
- input_file is the location of diamonds.csv (yes this only works for that specific file)
- output_dir is the directory used to saved the models (create the dir beforehand or a nasty error will appear!)
- {linear,ridge,lasso,xgboost} are the regresion model choices (new ones will be added automatically)
- [--debug] is an optional parameter, to enable debug mode and get some extra prints
- [-h] to get this help section in the terminal

for example, we can train a linear model with 
```console
python .\challenges\train_test.py .\data\diamonds.csv .\data\saved_models\ linear --debug
```
and it will be saved in the directory '\\saved_models\\'. The saved file is a dictionary containing:
- 'name':           the model name (str)
- 'model':          the trained model
- 'hyperparams':    the hyperparameters used to train the model
- 'scores':         dictionary containing performance metrics

### Loading a model
You can also load a saved model to test it further or see the performance it obtained. This is done using pickle.

There is an example of this in the file 'load_example.py', which can be called from the terminal using the following command:
```console
python load_example.py model_to_load [--debug] [-h]
```
where:
- model_to_load is the .pkl file to load
- [--debug] is an optional parameter, to enable debug mode and get some extra prints
- [-h] to get this help section in the terminal

for example, we can load the linear model with
```console
python .\challenge\load_example.py .\data\saved_models\LinearRegressionModel.pkl --debug
```
This will load the trained model in-code (it can be modified to actually use the model further), and it will print the saved testing results.

### Hyper-parameter optimization
If you want to find the best hyper-parameters for a specific model, you need to define the dictionary of parameters in config.py (if not already there), and then run hyper_optim.py from the terminal using:
```console
python hyper_optim.py input_file output_dir {linear,ridge,lasso,xgboost} [--n_trials N_TRIALS] [--debug] [-h]
```

where:
- input_file is the location of diamonds.csv (yes this only works for that specific file)
- output_dir is the directory used to saved the models (create the dir beforehand or a nasty error will appear!)
- {linear,ridge,lasso,xgboost} are the regresion model choices (new ones will be added automatically)
- [--n_trials N_TRIALS] is an optional parameter that sets the max number of optuna trials to run (higher values means more time!)
- [--debug] is an optional parameter, to enable debug mode and get some extra prints
- [-h] to get this help section in the terminal

for example, we can optimize the hyper-parameters of the xgboost model with 
```console
python .\challenges\hyper_optim.py .\data\diamonds.csv .\data\saved_models\ xgboost --n_trials 1000 --debug
```
and it will be saved in the directory '\\saved_models\\'. Performance metrics will also be printed and saved alongside the model.
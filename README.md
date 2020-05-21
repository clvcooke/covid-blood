# COVID-19 Detection from Blood Cell Morphology

## Getting started
### Installing Python + Dependencies
For this project we will be using the *Anaconda Package Manager* for python. If you haven't used Anaconda before, this is a great time to learn!

To install anaconda on the server you'll need to download and run the installer:

```bash
cd /tmp
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
```

After anaconda is installed you can install the project dependencies by creating a new `covid` environment.

If you cloned this repo to your home directory this would look like this:

```bash
cd ~/covid-blood
conda env create -f environment.yml
```

You can check if it worked by a) looking at the logs, and b) by running `conda activate covid`

### Experiment Tracking
We will be using `wandb` to track experiments. You can sign up for a free account here: [Weights and Biases](wandb.ai). You'll want to use your `.edu` email to get all the free stuff :).

When you first run an experiment you'll be prompted to authorize your account and be told where to look to track your experiments.

### Remote Access
Due to IRB guidelines all our data will remain on Duke controlled computers. For now that will mean the COL GPU Server.

To work with this data you will need to remotely access the server either through SSH or RDP.

### Running code using PyCharm
PyCharm has a great *remote debugging* feature built into it's professional edition. Remote debugging allows you to run your code on a remote server, but debug it using your local IDE.
 You can find instructions for setting this up here:
 [Remote Debugging over SSH in PyCharm](https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html#remote-interpreter)
 
## Contribution guidelines

Don't push to master!

### Code style
Functions should use named arguments by default
## Py-taxis
Py-taxis is a Python package for analysis of movies of swimming bacteria.
It consists of three modules:
* image - image processing module, it uses [**OpenCV**](https://opencv.org/) to detect bacteria in each frame of the movie and [**trackpy**](https://github.com/soft-matter/trackpy) to connect coordinates into trajectorie.
* proc - trajectory analysis module, allows to calculate instantaneous parameters of the trajectories, filter spurious trajectories and detect distinct motility states, e.g. runs and tumbles.
* plot - contains various handy plotting functions.
[**Read the tutorial**](/examples/Full_walkthrough.ipynb) for details on how to use it.

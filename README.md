## Py-taxis
Py-taxis is a Python package for analysis of movies of swimming bacteria.
It consists of three modules:
* [**image**](/py-taxis/image.py) - image processing module, uses [**OpenCV**](https://opencv.org/) to detect bacteria in each frame of the movie and [**trackpy**](https://github.com/soft-matter/trackpy) to connect coordinates into trajectories. Please see [**this video**](/examples/video_with_detected_trajectories.avi) for an example of trajectories detection.
* [**proc**](/py-taxis/proc.py) - trajectory analysis module, allows to calculate instantaneous parameters of the trajectories, filter spurious trajectories and detect distinct motility states, e.g. runs and tumbles.
* [**plot**](/py-taxis/plot.py) - contains various handy plotting functions.

[**Read the tutorial**](/examples/Full_walkthrough.ipynb) for details on how to use the library. If you use py-taxis for published research, please cite [**this paper**](https://www.biorxiv.org/content/early/2017/10/30/211474) to credit the authors. The workflow below illustrates different stages of data analysis from detecting bacteria to assigning motility states.

![Alt text](/examples/analysis_workflow.png?raw=true "Title")

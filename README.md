## Py-taxis
Py-taxis is a Python package for analysis of movies of swimming bacteria. [**Read the tutorial**](/examples/Full_walkthrough.ipynb) for details on how to use the library. If you use py-taxis for published research, please cite [**this paper**](https://www.biorxiv.org/content/early/2017/10/30/211474) to credit the authors. 

### Library structure
Py-taxis consists of three modules:
* [**image**](/py-taxis/image.py) - image processing module, uses [**OpenCV**](https://opencv.org/) to detect bacteria in each frame of the movie and [**trackpy**](https://github.com/soft-matter/trackpy) to connect coordinates into trajectories.
* [**proc**](/py-taxis/proc.py) - trajectory analysis module, allows to calculate instantaneous parameters of the trajectories, filter spurious trajectories and detect distinct motility states, e.g. runs and tumbles.
* [**plot**](/py-taxis/plot.py) - contains various handy plotting functions.


### Data analysis workflow
The workflow below illustrates different stages of data analysis from detecting bacteria to assigning motility states.


![Alt text](/examples/analysis_workflow.png?raw=true "Title")

### Example of the detected trajectories

Trajectory in blue belongs to simming bacteria, while trajectory in red - to the bacteria stuck to the glass surface. Such circular trajectories are removed at the filtering stage of data analsys in order to minimize noise due to errouneous assignment of motility states.

 ![**this video**](/examples/detected_trajectories.gif) 
# Algorithm Distillation

I used a course project as a good excuse to replicate Algorithm Distillation ([Laskin et al., 2022](https://arxiv.org/abs/2210.14215)), and came quite close to the official results depiste some compute-saving shortcuts. An overly basic explainer written for a non-RL class can be found in `AD-writeup.pdf`. This is a minimialist from-scratch version in the Dark-Key-To-Door environment, with extra experiments to investigate the way that AD's N independent RL agent dataset format creates a meta-learner out of a training loop that would otherwise be doing regular behavior cloning.

![AD Diagram](writeup_and_readme/ad_figure_png_version.png)

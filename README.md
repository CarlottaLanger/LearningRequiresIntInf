# Learning to Predict Requires Integrated Information
This is the code and data corresponding to the article "Learning to Predict requires Integrated Information" (arXiv-preprint: https://arxiv.org/abs/2209.01418). 

There we analyze the behavior of the Integrated Information measure, which can be seen as a measure for the complexity of the agent's controller, while the agents learn to predict and navigate their environment. This simple setup allows us to additionally calculate measures for every information flow among the agent's brain, body and environment, including a measure for Morphological Computation. The movement
of the agents is demonstrated in the video below. If their body touches a wall they are stuck, as long as at least one of the sensors is still detecting a wall.



https://user-images.githubusercontent.com/21078779/187882582-525eede3-27f9-410d-9f28-d64c6f97a3b8.mp4



In the main.py document one can change the agents from fully connected to limited (no integrated information possible) by setting the integrated information variable in line 57 to 0. 
The sensor length can be changed in line 64.

# Requirements
The code requires a python version 3 and it was tested on python 3.8 with the following packages: 
1. matplotlib 3.7.1 
2. shapely 1.8.5 
3. scipy 1.10.1 
4. descartes 1.1.0. 

# Results

The program saves for every 100th step the measures in the file with the path given in line 67 of the main.py. 
There we have in that order:
0. Integrated Information,
1. Morphological Computation
2. Reactive Cotnrol
3. Sensory Information
4. Total Information Flow
5. Command
6. P(goal)
7. Sensory Prediction
8. Actuator Prediction
9. Synergistic Prediction
10. Full Prediction
11. Goal Prediciton
12. Action Effect
13. Success Rate and World Difference.

The animation displays the results for the Integrated Information, Morphological Computation, Action Effect, Sensory Information, Command and Total Information Flow, as depicted in the picture below. 

![Figure_1a](https://user-images.githubusercontent.com/21078779/227466420-36e0cc59-412c-421b-807f-dcb4578e8871.png)

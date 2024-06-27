import random

import shapely as sh
import scipy.spatial.distance
import shapely.geometry as sg
import shapely.affinity as sa
import numpy as np
import AuxiliaryFunctions as af
import EmAlgorithm2 as em
import Measures as me
import matplotlib.animation as animation
from scipy.stats import entropy
import time
from math import log2
import copy

global world_diff
global sample
sample = False
global iteration
global integratedinformation
iteration = 0

# define the environment
arena = sg.box(-10,-10,10,10)
hole = sg.box(-7,-7,-5,7)
hole = hole.union(sg.box(-7,5,6,7))
hole = hole.union(sg.box(-7,-5,6,-7))
hole = hole.union(sg.box(0,-1,12,1))
hole = hole.union(sg.box(10,-11,11,11))
hole = hole.union(sg.box(-10,-11,-11,11))
hole = hole.union(sg.box(-11,-10,11,-11))
hole = hole.union(sg.box(11,10,-11, 11))
arena = arena.difference(hole)

walls = arena.boundary
iters = 0

colorGoal = '#E6C200'
colorRe = '#93091F'
colorBlu = '#003C97'
ForestGreen = '#228b22'
colorT = '#2DA0A1'



import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
#np.random.seed(19680801)
np.random.seed(80801)
tau = 2*np.pi


# set to False to deactivate the animation, increases the performance
anim = True
#0: no integrated information, 1: no integrated information in the beginning, 2: fully connected agents
integratedinformation = 2

# set ideal to "True" if the agents should have access to the empirical world model. These enambles the PWM agents
global ideal
ideal = True

# change the accuracy of the world model of the PWM agents with the learninglimit
global learninglimit
learninglimit = 20000

sample = False
global path

#sensor length and body size
sens_length= 1.25
bodySize = 0.55 #0.3


def rotate_origin(obj, theta):
    return sa.rotate(obj,theta,use_radians=True,origin=(0,0))


path = 'results'+ str(integratedinformation)+'measadures'+ str(sens_length).replace('.','')+'.py'
global sensors
sensors = [
    rotate_origin(sg.LineString([(0,bodySize),(0, bodySize+sens_length)]),-0.1*tau),
    rotate_origin(sg.LineString([(0,bodySize),(0,bodySize+sens_length)]),0.1*tau),
]

# for plotting only
bodyPoly = sg.Point(0,0).buffer(bodySize, resolution=2)
bodyPoly = bodyPoly.union(sg.box(-0.01,-1,0.01,0))

def increaseSensorSize(incr):
    global sensors
    sensors = [
        rotate_origin(sg.LineString([(0, bodySize), (0, bodySize+incr)]), -0.1 * tau),
        rotate_origin(sg.LineString([(0, bodySize), (0, bodySize+incr)]), 0.1 * tau),
    ]

class AgentBody:
    # note: this class should only store what needs to be copied - everything
    # else (geometry etc.) should be generated on the fly
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.pos = sg.Point(x, y)
        self.theta = theta
    def randomValidPosition(self):
        while True:
            x, y = np.random.random(2)*24-12
            self.pos = sg.Point(x, y)
            self.theta = np.random.random()*tau
            if self.touchingWall():
                return self
    def touchingWall(self):
        return arena.contains(self.pos) and (self.pos.distance(walls) > bodySize )

    def sensorValues(self,plot=True,plot_sensors=True):
        mySensors = [
            sa.translate(rotate_origin(s,self.theta),self.pos.x,self.pos.y)
            for s in sensors
        ]
        result = np.zeros(len(mySensors) + 1)
        for i in range(len(mySensors)):
            if hole.intersects(mySensors[i]):
                result[i] = 1
        if anim==True:
            if plot_sensors:
                for i in range(len(mySensors)):
                    if result[i]:
                        col = '#00ff00'
                    else:
                        col = '#000000'
                    plot_line(ax,mySensors[i],col)
            if plot:
                if not self.touchingWall():
                    col = '#ff0000'
                else:
                    col = '#0000ff'
                body = sa.translate(rotate_origin(bodyPoly,self.theta),self.pos.x,self.pos.y)
                plot_poly(body,col)
        result[len(mySensors)] =  self.touchingWall()
        return result
    def update(self, controllerValues, dt=1.0):
        turnLeft, turnRight = controllerValues
        speed = (np.sum(controllerValues)+1)*0.2
        turning = 0.04*tau*(turnRight-turnLeft)
        self.theta += turning
        self.pos = sa.translate(self.pos,
            -speed*np.sin(self.theta)*dt,speed*np.cos(self.theta))

    # here the agent is stuck at a wall
    def stick(self, controllerValues, dt=1.0):
        turnLeft, turnRight = controllerValues
        left, right, alive = self.sensorValues()
        if left == 0 and right == 0:
            turning = 0.04 * tau * (turnRight - turnLeft)
            self.theta += turning
            speed = (np.sum(controllerValues) + 1) * 0.2
            self.pos = sa.translate(self.pos, -speed * np.sin(self.theta) * dt, speed * np.cos(self.theta))
        else:
            turning = 0.04 * tau * (turnRight - turnLeft)
            self.theta += turning


class Controller:
    # Inner workings of the agent
    def __init__(self, n_inputs=3, n_outputs=2, n_hidden=2):
        self.last_c = np.array([0,0])
        self.n = 0
        self.n_pred = 0
        self.n_sa = np.zeros(pow(2, n_inputs + n_outputs))
        self.last_s = np.array([0, 0, 1])
        self.last_a = np.array([0, 0])
        self.goal = np.array([0.5, 0.5])
        self.p_sca = np.zeros(2**(n_inputs+n_outputs+n_hidden)) + (1/ 2**(n_inputs+n_outputs+n_hidden))
        self.p_a = af.rand_cond_distr(2, n_hidden + n_inputs)
        self.p_s= np.ones(pow(2,n_inputs + n_inputs + n_outputs) ) / pow(2,n_inputs)
        self.p_s_pred= np.ones(pow(2,n_inputs + n_hidden + n_outputs) ) / pow(2,n_inputs)
        if integratedinformation == 1:
            self.p_c = af.rand_cond_distr2(2, n_hidden + n_inputs)
        if integratedinformation == 2:
            self.p_c = af.rand_cond_distr(2, n_hidden + n_inputs)
        if integratedinformation == 0:
            self.p_c_i = af.rand_cond_distr2(2, 1 + n_inputs)


    # em-algorithm to find the optimal policies and update the world model
    # here one could insert a different learning algorithm that updates the actuator, controller and prediction nodes
    def learn_policies(self, integratedinformation):
        #number of learning steps
        l_steps = 5

        global ideal
        p_c_red = np.zeros((2, pow(2, 4)))
        p_s_pred_red = np.zeros(pow(2, 3))
        p_s_red = np.zeros(pow(2,3))

        if(ideal == False):
            for i in range(pow(2, 3)):
                p_s_pred_red[i] = self.p_s_pred[
                    self.last_c[0] * pow(2, 6) + self.last_c[1] * pow(2, 5) + self.last_a[0] * pow(2, 4) + self.last_a[
                        1] * pow(2, 3) + i]
        else:
            for i in range(pow(2,3)):
                p_s_red[i] = self.p_s[int(
                self.last_s[0] * pow(2, 7) + self.last_s[1] * pow(2, 6) + self.last_s[2] * pow(2, 5) + self.last_a[
                    0] * pow(2, 4) + self.last_a[1] * pow(2, 3) + i)]

        if integratedinformation == 0:
            for i in range(pow(2, 3)):
                p_c_red[0][i] = self.p_c_i[0][self.last_c[1] * pow(2, 4) + i]
                p_c_red[1][i] = self.p_c_i[1][self.last_c[0] * pow(2, 4) + i]

        else:
            for i in range(pow(2, 4)):
                for j in range(2):
                    p_c_red[j][i] = self.p_c[j][self.last_c[0] * pow(2, 5) + self.last_c[1] * pow(2, 4) + i]

        if(ideal ==False):
            for i in range(l_steps):
                # policies
                p1 = em.conditioning_pred_reduced(self.p_s_pred, p_c_red, self.p_a, p_s_pred_red)
                p_c_red, self.p_a = em.factorizing_reduced(p1)

                # world model
                p2 = em.conditioning_on_sensors_red(p_s_pred_red, p_c_red, self.p_a, self.p_s, self.p_s_pred)
                self.p_s_pred = em.factorizing_sensors(p2)
                for i in range(pow(2, 3)):
                    p_s_pred_red[i] = self.p_s_pred[
                        self.last_c[0] * pow(2, 6) + self.last_c[1] * pow(2, 5) + self.last_a[0] * pow(2, 4) + self.last_a[
                            1] * pow(2, 3) + i]
        else:
            for i in range(l_steps):
                p1 = em.conditioning_pred_reduced_world(self.p_s, p_c_red, self.p_a, p_s_red)
                p_c_red, self.p_a = em.factorizing_reduced(p1)

        if integratedinformation == 0:
            for i in range(pow(2, 3)):
                self.p_c_i[0][int(self.last_c[0] * pow(2, 4) + i)] = p_c_red[0][i]
                self.p_c_i[1][int(self.last_c[1] * pow(2, 4) + i)] = p_c_red[1][i]

        else:
            for i in range(pow(2, 4)):
                for j in range(2):
                    self.p_c[j][self.last_c[0] * pow(2, 5) + self.last_c[1] * pow(2, 4) + i] = p_c_red[j][i]
        if(ideal == False):
            for i in range(pow(2, 3)):
                self.p_s_pred[
                    self.last_c[0] * pow(2, 6) + self.last_c[1] * pow(2, 5) + self.last_a[0] * pow(2, 4) + self.last_a[
                        1] * pow(2, 3) + i] = p_s_pred_red[i]
        else:
            for i in range(pow(2, 3)):
                self.p_s[int(self.last_s[0] * pow(2, 7) + self.last_s[1] * pow(2, 6) + self.last_s[2] * pow(2, 5) +
                                 self.last_a[0] * pow(2, 4) + self.last_a[1] * pow(2, 3) + i)] = p_s_red[i]

        return self.p_s

    # calculate the next step and update the sampled distributions
    def update(self, integratedinformation, inputValues):

        #sampling the last values
        self.n = self.n + 1
        index_sca = int(af.getIndex(self.last_s, self.last_c, self.last_a))
        for i in range(len(self.p_sca)):
            self.p_sca[i] = self.p_sca[i] * (self.n / (self.n + 1))
        self.p_sca[index_sca] = self.p_sca[index_sca] + (1/(self.n +1))

        if inputValues[2] == 0:
            self.goal[0] = self.goal[0] *(self.n / (self.n + 1)) +  (1/(self.n +1))
            self.goal[1] = self.goal[1] *(self.n / (self.n + 1))
        else:
            self.goal[1] = self.goal[1] * (self.n / (self.n + 1)) + (1 / (self.n + 1))
            self.goal[0] = self.goal[0] * (self.n / (self.n + 1))

        index_ssa = int(af.getIndex( self.last_s, self.last_a, inputValues))

        index_sa = int(af.getIndex(self.last_s, self.last_a, []))
        self.n_sa[index_sa] = self.n_sa[index_sa] +1
        global ideal
        if ideal:
            global iteration
            global learninglimit
            if iteration < learninglimit:
                for i in range(len(self.p_s)):
                    if int((i // (pow(2, len(inputValues))))) == index_sa:
                        if i == index_ssa:
                            self.p_s[i] = ((self.n_sa[index_sa]) / (self.n_sa[index_sa] + 1)) * self.p_s[i] + (
                                        1 / (self.n_sa[index_sa] + 1))
                        else:
                            self.p_s[i] = ((self.n_sa[index_sa]) / (self.n_sa[index_sa] + 1)) * self.p_s[i]
                    else:
                        self.p_s[i] = self.p_s[i]
        else:
            for i in range(len(self.p_s)):
                if int((i// (pow(2,len(inputValues))))) == index_sa:
                    if i == index_ssa:
                        self.p_s[i] = ( (self.n_sa[index_sa])/(self.n_sa[index_sa] +1) )* self.p_s[i] + (1/(self.n_sa[index_sa] +1))
                    else:
                        self.p_s[i] = ((self.n_sa[index_sa]) / (self.n_sa[index_sa] + 1)) * self.p_s[i]
                else:
                    self.p_s[i] = self.p_s[i]

        self.learn_policies(integratedinformation)

        #values for the movement
        # for the next c
        if integratedinformation == 0:
            p_next_c_i = np.zeros((len(self.p_c_i), 2))
            index_sc0 = np.array([int(af.getIndex(inputValues, self.last_c[0], [0])),
                                  int(af.getIndex(inputValues, self.last_c[0], [1]))])
            index_sc1 = np.array([int(af.getIndex(inputValues, self.last_c[1], [0])),
                                  int(af.getIndex(inputValues, self.last_c[1], [1]))])
            for i in range(2):
                p_next_c_i[0][i] = self.p_c_i[0][index_sc0[i]]
                p_next_c_i[1][i] = self.p_c_i[1][index_sc1[i]]
            c = np.random.choice([0, 1], 1, p=p_next_c_i[0]), np.random.choice([0, 1], 1, p=p_next_c_i[1])


        else:
            p_next_c = np.zeros((len(self.p_c),2) )
            index_sc=  np.array( [int(af.getIndex( inputValues, self.last_c, [0])), int(af.getIndex(inputValues, self.last_c, [1]))])
            for j in range(len(self.p_c)):
                for i in range(2):
                    p_next_c[j][i] = self.p_c[j][index_sc[i]]
            c = np.random.choice([0,1],1,p=p_next_c[0]), np.random.choice([0,1],1,p=p_next_c[1])

        #for the next actuator states
        p_next_a = np.zeros((len(self.p_a),2) )
        index_sc = np.array([int(af.getIndex(inputValues, c, [0])), int(af.getIndex(inputValues, c, [1]))])
        for j in range(len(self.p_a)):
            for i in range(2):
                p_next_a[j][i] = self.p_a[j][index_sc[i]]
        #here we need to add noise to avoid the probability of an actuator state being stuck at 0
        noise = np.random.normal(0, .1, p_next_a.shape)
        p_next_a = np.abs(p_next_a + noise)
        sumpa0 = np.sum(p_next_a[0])
        sumpa1 = np.sum(p_next_a[1])
        outputValues = np.random.choice([0,1],1,p=p_next_a[0]/sumpa0), np.random.choice([0,1],1,p=p_next_a[1]/sumpa1)

        #after calculating the movement
        self.last_s = inputValues
        self.last_a = outputValues
        self.last_c = c
        return outputValues



class Agent:
    def __init__(self,it, x=0.0, y=0.0, theta=0.0):
        self.body = AgentBody(x,y,theta)
        self.controller = Controller(len(sensors) +1)
        self.it = it
    def alive(self):
        return self.body.touchingWall()
    def set(self, it):
        self.it = it
    def reset(self, it):
        self.body.randomValidPosition()
        self.it = it
        return self
    def update(self,dt=1.0,plot=True,plot_sensors=True):
        sensorValues = self.body.sensorValues(plot=plot,plot_sensors=plot_sensors)
        global integratedinformation
        controllerValues = self.controller.update( integratedinformation, sensorValues)
        if sensorValues[2]:
            self.body.update(controllerValues)
        else:
            self.body.stick(controllerValues)
        return True



def updateagents(t, output):
    global iteration
    iteration = iteration +1
    if anim==True:
        fig.show()
        fig.canvas.draw()
    global world
    global path
    n_agents = 1
    output.write('np.array([')
    agents = [Agent(0).reset(0) for i in range(n_agents)]
    iters = 0
    c= [0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0]
    while(iters < 20000):
        if anim==True:
            ax.clear()
            plot_arena()
        alive = [agent.update(plot=True) for agent in agents]
        if iters%100==0:
            for agent in agents:
                output.write("([")
                if integratedinformation==0:
                   d = me.calc_meas_noint(agent.controller.p_sca, agent.controller.p_s, agent.controller.p_s_pred, agent.controller.p_c_i, agent.controller.p_a)
                else:
                   d = me.calc_meas(agent.controller.p_sca, agent.controller.p_s, agent.controller.p_s_pred, agent.controller.p_c, agent.controller.p_a)
                   #d1 = me.calc_meas_morph(agent.controller.p_sca, agent.controller.p_s)
                d = np.append(d, agent.controller.goal[1])
                print("goal", agent.controller.goal[1], d,)
                d_write = ','.join(map(str, d))
                output.write(d_write)
                if agent == agents[0]:
                    for i in range(len(c)):
                        c[i] = np.append(c[i], d[i])
                    if anim==True:
                        ax1.clear()
                        ax3.clear()
                        ax2.clear()
                        ax4.clear()
                        ax5.clear()
                        plotmeasures(c, iters//100)
                output.write("]),")
        iters += 1
        if anim==True:
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.show()
    output.write("]),")
    if iteration == 10:
        exit(0)
    return 0



def plot_arena():
    plot_poly(arena)
   # #print(walls.bounds)
    x_range = walls.bounds[0]-2,walls.bounds[2]+2
    y_range = walls.bounds[1]-2,walls.bounds[3]+2
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect(1)

if anim == True:
    import matplotlib
    # the next line is needed on my machine to make animation work - if you have
    # problems on your machine, try changing it to a different MatPlotLib backend
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from descartes.patch import PolygonPatch
    plt.rcParams["figure.figsize"] = (12,8)

    output = open(path , 'a+')
    output.write('\n')
    output.write('np.array([')
    print("test1")
    fig = plt.figure()
    ax = fig.add_subplot(321)
    ax1 = fig.add_subplot(322)
    ax2 = fig.add_subplot(323)
    ax3 = fig.add_subplot(324)
    ax4 = fig.add_subplot(325)
    ax5 = fig.add_subplot(326)


    COLOR = {
        True:  '#6699cc',
        False: '#ff3333'
        }
    def v_color(ob):
        return COLOR[ob.is_valid]
    def plot_poly(polygon, col=None):
        if not col:
            col = v_color(polygon)
    #	patch = PolygonPatch(polygon, facecolor=col, edgecolor=col, alpha=1, zorder=2)
        patch = PolygonPatch(polygon, fill=False, edgecolor=col, alpha=1, zorder=2)
        ax.add_patch(patch)
    def plot_line(ax, ob, col='#000000'):
        x, y = ob.xy
        ax.plot(x, y, color=col, alpha=1, linewidth=1, solid_capstyle='round', zorder=2)
    def plotmeasures(c, n):
        k = 0
        ax1.plot(np.arange(n + 2)[k:], c[16][0:], color=colorGoal, label="Goal")
        global ideal
        ax2.plot(np.arange(n + 2)[k:], c[0][0:], color=colorT, label = "Integrated Information")
        ax4.plot(np.arange(n + 2)[k:], c[1][0:], color=ForestGreen, label = "Morphological Computation")
        ax4.plot(np.arange(n + 2)[k:], c[12][0:], color=colorRe, label = "Action Effect")
        ax3.plot(np.arange(n + 2)[k:], c[3][0:], color=colorRe , label = "Sensory Information")
        ax5.plot(np.arange(n + 2)[k:], c[4][0:], color='black', label="Total Information Flow")
        ax3.plot(np.arange(n + 2)[k:], c[5][0:], color=colorBlu, label="Command")
        if not ideal:
            ax3.plot(np.arange(n + 2)[k:], c[10][0:], color=ForestGreen, label="Full Prediction")

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper left")
        ax3.legend(loc="upper left")
        ax4.legend(loc="upper left")
        ax5.legend(loc="upper left")

    ani = animation.FuncAnimation(fig, updateagents, fargs=(output,))
    plt.show()
    output.write('])')
else:
    start_time = time.time()
   # writing the solutions to a file
    output = open(path, 'a')
    for i in range(10):
        updateagents(1, output)
    output.write('])')
    output.close()


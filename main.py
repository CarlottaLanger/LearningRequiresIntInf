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

tau = 2*np.pi

def rotate_origin(obj, theta):
    return sa.rotate(obj,theta,use_radians=True,origin=(0,0))

anim = True
#0: no integrated information, 1: no integrated information in the beginning, 2: fully connected agents
integratedinformation = 2
sample = False
#global iteration
#iteration = 0
global path

#sensor length
sens_length=1.25
bodySize = 0.55 #0.3



path = 's'+ str(integratedinformation)+'measures'+ str(sens_length).replace('.','')+'.py'
global sensors
sensors = [
    rotate_origin(sg.LineString([(0,bodySize),(0, bodySize+sens_length)]),-0.1*tau),
   # sg.LineString([(0,bodySize),(0,3)]),
    rotate_origin(sg.LineString([(0,bodySize),(0,bodySize+sens_length)]),0.1*tau),
]

# for plotting only
bodyPoly = sg.Point(0,0).buffer(bodySize, resolution=2)
bodyPoly = bodyPoly.union(sg.box(-0.01,-1,0.01,0))

def increaseSensorSize(incr):
    global sensors
    sensors = [
        rotate_origin(sg.LineString([(0, bodySize), (0, bodySize+incr)]), -0.1 * tau),
        # sg.LineString([(0,bodySizey),(0,3)]),
        rotate_origin(sg.LineString([(0, bodySize), (0, bodySize+incr)]), 0.1 * tau),
    ]
  #  #print("length", sensors[0].length)
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
            if self.touchingWall(): # and sum(self.sensorValues(False))==0:
                return self
    def touchingWall(self):
      #  #print(arena.contains(self.pos) and (self.pos.distance(walls) > bodySize ))
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
        #print(controllerValues)
        turnLeft, turnRight = controllerValues
        speed = (np.sum(controllerValues)+1)*0.2
        turning = 0.04*tau*(turnRight-turnLeft)
        self.theta += turning
       # #print(self.pos)
        self.pos = sa.translate(self.pos,
            -speed*np.sin(self.theta)*dt,speed*np.cos(self.theta))

    # here the agent is stuck at a wall
    def stick(self, controllerValues, dt=1.0):
        turnLeft, turnRight = controllerValues
        left, right, alive = self.sensorValues()
       # #print("Left", left, right)
        lr = left + right
        counter = 0
        if left == 0 and right == 0:
            turning = 0.04 * tau * (turnRight - turnLeft)
            self.theta += turning
            speed = (np.sum(controllerValues) + 1) * 0.2
            ##print(self.pos)
            self.pos = sa.translate(self.pos, -speed * np.sin(self.theta) * dt, speed * np.cos(self.theta))
           # #print(self.pos)
        else:
            turning = 0.04 * tau * (turnRight - turnLeft)
            self.theta += turning

lrate = 0.05

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
        #self.p_c = af.rand_cond_distr2(2, n_hidden + n_inputs)
        self.p_s= np.ones(pow(2,n_inputs + n_inputs + n_outputs) ) / pow(2,n_inputs)
        self.p_s_pred= np.ones(pow(2,n_inputs + n_hidden + n_outputs) ) / pow(2,n_inputs)
        if integratedinformation == 1:
            self.p_c = af.rand_cond_distr2(2, n_hidden + n_inputs)
        if integratedinformation == 2:
            self.p_c = af.rand_cond_distr(2, n_hidden + n_inputs)
        if integratedinformation == 0:
            self.p_c_i = af.rand_cond_distr2(2, 1 + n_inputs)
       # #print(self.p_s, np.sum(self.p_s))

    # Em-algorithm to find the optimal policies and update the world model
    def learn_policies(self, integratedinformation):
        #number of learning steps
        l_steps = 15

        p_c_red = np.zeros((2, pow(2,4)))
        p_s_pred_joint = np.zeros(pow(2,9))
        p_s_pred_joint_old = np.zeros(pow(2,9))
        p_s_pred_red = np.zeros(pow(2,3))
        old_pred = np.copy(self.p_s_pred)

        for i in range(pow(2,3)):
            p_s_pred_red[i] = self.p_s_pred[self.last_c[0]*pow(2,6) + self.last_c[1]*pow(2,5)+ self.last_a[0]*pow(2,4) + self.last_a[1]*pow(2,3) + i]

        if integratedinformation ==0:
            for i in range(pow(2,3)):
                p_c_red[0][i] = self.p_c_i[0][self.last_c[1]*pow(2,4) + i]
                p_c_red[1][i] = self.p_c_i[1][self.last_c[0]*pow(2,4)+ i]

        else:
            for i in range(pow(2,4)):
                for j in range(2):
                    p_c_red[j][i] = self.p_c[j][self.last_c[0]*pow(2,5) + self.last_c[1]*pow(2,4) + i]

        ##print(p_c_red, np.sum(p_c_red))
        for i in range(l_steps):
            # policies
            p1 = em.conditioning_pred_reduced(self.p_s_pred, p_c_red, self.p_a, p_s_pred_red )
            p_c_red, self.p_a = em.factorizing_reduced(p1)

            # world model
            p2 = em.conditioning_on_sensors_red(p_s_pred_red,p_c_red,self.p_a, self.p_s, self.p_s_pred)
            self.p_s_pred = em.factorizing_sensors(p2)
            for i in range(pow(2, 3)):
                p_s_pred_red[i] = self.p_s_pred[self.last_c[0] * pow(2, 6) + self.last_c[1] * pow(2, 5) + self.last_a[0] * pow(2, 4) + self.last_a[1] * pow(2, 3) + i]

        if integratedinformation == 0:
            for i in range(pow(2,3)):
                self.p_c_i[0][int(self.last_c[0]*pow(2,4) +  i)] = p_c_red[0][i]
                self.p_c_i[1][ int(self.last_c[1] * pow(2, 4) + i)] = p_c_red[1][i]

        else:
            for i in range(pow(2,4)):
                for j in range(2):
                    self.p_c[j][self.last_c[0]*pow(2,5) + self.last_c[1]*pow(2,4) + i] = p_c_red[j][i]
        for i in range(pow(2,3)):
            self.p_s_pred[ self.last_c[0] * pow(2, 6) + self.last_c[1] * pow(2, 5) + self.last_a[0] * pow(2, 4) + self.last_a[ 1] * pow(2, 3) + i] = p_s_pred_red[i]
        global world_diff
        global world_entropy

        for i in range(pow(2,9)):
            p_s_pred_joint[i] = self.p_s_pred[i%pow(2,5) + pow(2,5)*(i//pow(2,7))]* self.p_sca[i//pow(2,3)]
            p_s_pred_joint_old[i] = old_pred[i%pow(2,5) + pow(2,5)*(i//pow(2,7))]* self.p_sca[i//pow(2,3)]
        world_entropy= 0 #entropy(p_s_pred_joint)
        wdiff = 0
        for i in range(pow(2,9)):
            if old_pred[i%pow(2,5) + pow(2,5)*(i//pow(2,7))]*self.p_s_pred[i%pow(2,5) + pow(2,5)*(i//pow(2,7))] !=0:
                wdiff = wdiff + p_s_pred_joint_old[i] * (log2(old_pred[i%pow(2,5) + pow(2,5)*(i//pow(2,7))]) - log2(self.p_s_pred[i%pow(2,5) + pow(2,5)*(i//pow(2,7))]))
                world_entropy = world_entropy - p_s_pred_joint[i]  * log2(self.p_s_pred[i%pow(2,5) + pow(2,5)*(i//pow(2,7))])
        world_diff = wdiff
        return np.linalg.norm(old_pred-self.p_s_pred)

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

        #for the next a
        p_next_a = np.zeros((len(self.p_a),2) )
        index_sc = np.array([int(af.getIndex(inputValues, c, [0])), int(af.getIndex(inputValues, c, [1]))])
        for j in range(len(self.p_a)):
            for i in range(2):
                p_next_a[j][i] = self.p_a[j][index_sc[i]]
        noise = np.random.normal(0, .1, p_next_a.shape)
       # #print(p_next_a)
        p_next_a = np.abs(p_next_a + noise)
        ##print(p_next_a)
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



def updateagents(t):
    #print("t", t)
   # plt.ion()
    global iteration
    iteration = iteration +1
    if anim==True:
        fig.show()
        fig.canvas.draw()
    global world
   # #print(world)
    global path
    output = open(path , 'a')
    output.write('\n')
    output.write('np.array([')
    n_agents = 1

    agents = [Agent(0).reset(0) for i in range(n_agents)]
    iters = 0
    c= [0,0,0, 0,0,0,0,0,0,0,0,0,0,0]
    goal_var = False
    while(iters < 20000): #((goal_var == False or iters < 1000) and (iters < 5000) ):
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
                d = np.append(d, agent.controller.goal[1])
                global world_diff
                d = np.append(d, world_diff)
                d_write = ','.join(map(str, d))
                output.write(d_write)
                if agent == agents[0]:
                    for i in range(len(c)):
                        c[i] = np.append(c[i], d[i])
                        #print(i, c[i], len(c), len(d))
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
    if iteration == 5:
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
        #ax3.plot(np.arange(0, pow(2, 6)), agent.controller.p_a[0], color="red")
       # ax2.plot(np.arange(0, pow(2, 7)), agent.controller.p_s_pred, color="green")
        k = 0
       # #print("n ",np.arange(n+2)[k:], c[7][1:])
        ax1.plot(np.arange(n + 2)[k:], c[12][0:], color=colorGoal, label="Goal")

      #  ax2.plot(np.arange(n+2)[1:], c[0][1:], color = 'violet', label = "IntInf internal")
        ax2.plot(np.arange(n + 2)[k:], c[0][0:], color=colorT, label = "IntInf")
        ax4.plot(np.arange(n + 2)[k:], c[1][0:], color=ForestGreen, label = "MorphComp")
        ax2.plot(np.arange(n + 2)[k:], c[9][0:], color=colorRe , label = "Synergistic")
        ax3.plot(np.arange(n + 2)[k:], c[3][0:], color=colorRe , label = "SensInf")
        ax4.plot(np.arange(n + 2)[k:], c[4][0:], color='black', label="MutualInf")
        ax3.plot(np.arange(n + 2)[k:], c[5][0:], color=colorBlu, label="Command")
        ax5.plot(np.arange(n + 2)[k:], c[13][0:], color=ForestGreen, label="WorldDiff")

        ax1.legend(loc="upper left")
        ax2.legend(loc="upper left")
        ax3.legend(loc="upper left")
        ax4.legend(loc="upper left")
        ax5.legend(loc="upper left")
    output = open(path, 'a')
    #repeat = false
    #for i in range(2):
    ani = animation.FuncAnimation(fig, updateagents, repeat=False)

    plt.show()
    #main()
else:
    start_time = time.time()
   # writing the solutions to a file
    output = open(path, 'a')
    output.write("import numpy as np")
    output.write('\n')
    output.write('meas = np.array([')
    for i in range(10):
        updateagents(1, output)
    output.write('])')
    output.close()
    #print("--- %s seconds ---" % (time.time() - start_time))


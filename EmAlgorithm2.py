import numpy as np


def factorizing_reduced(p):
    p_c = np.zeros((2, pow(2,4)))
    p_a = np.zeros((2, 2*pow(2,5)))
    p_cs_a = np.zeros(pow(2, 5))
    p_s_c = np.zeros(pow(2,3))

    for i in range(len(p)):
        p_c[0][(i // pow(2, 3)) % pow(2, 4)] = p_c[0][(i // pow(2, 3)) % pow(2,4)] + p[i]
        p_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 4) % pow(2, 3)))] = p_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 4) % pow(2, 3)))] +  p[i]
        p_s_c[(i // pow(2, 4)) % pow(2, 3)] = p_s_c[(i // pow(2, 4)) % pow(2,3)] + p[i]

        p_a[0][(i % 2) + 2 * ((i // 4) % pow(2, 5))] = p_a[0][(i % 2) + 2 * ((i // 4) % pow(2, 5))] + p[i]
        p_a[1][((i // 2) % 2) + 2 * ((i // 4) % pow(2, 5))] = p_a[1][((i // 2) % 2) + 2 * ((i // 4) % pow(2, 5))]  + p[i]
        p_cs_a[(i//pow(2,2))%pow(2,5)] = p_cs_a[(i//pow(2,2))%pow(2,5)] + p[i]

    for i in range(len(p_c[0])):
        for j in range(2):
            if p_s_c[i//2] !=0:
                p_c[j][i] = p_c[j][i] / p_s_c[i//2]
            else:
                p_c[j][i] = 0.5
    for i in range(len(p_a[0])):
        for j in range(2):
            if p_cs_a[i//2] !=0:
                p_a[j][i] = p_a[j][i] / p_cs_a[i // 2]
            else:
                p_a[j][i] = 0.5
    return  p_c, p_a

def conditioning_pred_reduced(p_s_pred, p_c, p_a, p_s_pred_red):
    p_ext = np.zeros(pow(2,10))
    p_next = np.zeros(pow(2, 7))
    p_old = np.zeros(pow(2,10))
    p_s_goal =np.array([0,1])
    next_p = 0.0
    for i in range(len(p_old)):
        p_old[i] = p_s_pred_red[(i // pow(2, 7)) % pow(2, 3)] \
            * p_c[0][(i // pow(2, 6)) % pow(2, 4)] * p_c[1][(i // pow(2, 5)) % 2 + 2 * ((i // pow(2, 7) % pow(2, 3)))] \
            * p_a[0][((i//pow(2,3)) % 2) + 2 * ((i // pow(2,5)) % pow(2, 5))] * p_a[1][((i // pow(2,4)) % 2) + 2 * ((i // pow(2,5)) % pow(2, 5))] \
            * p_s_pred[i%pow(2,7)]


    next_p = np.array([0.0,0.0])
    for i in range(len(p_old)):
        next_p[i%2] = next_p[i%2] + p_old[i]

    for i in range(len(p_old)):
        if next_p[i % 2] > 0:
            p_ext[i] = (p_old[i] / next_p[i % 2]) * p_s_goal[i % 2]
        else:
            # #print("else")
            p_ext[i] = (1 / 64) * p_s_goal[i % 2]
    for i in range(len(p_ext)):
        p_next[i//pow(2,3)] = p_next[i//pow(2,3)] + p_ext[i]
    return p_next


#world model
def conditioning_on_sensors_red(p_s_pred_red,p_c_red, p_a,  p_s_goal, p_s_pred):
    p_next = np.zeros(pow(2,10))
    p_ssa =np.zeros(pow(2,8))
    p_sa = np.zeros(pow(2, 5))
    p_sca_full = np.zeros(pow(2,10))

    for i in range(pow(2,10)):
        p_sca_full[i] = p_s_pred_red[(i // pow(2, 7))] \
            * p_c_red[0][(i // pow(2, 6)) % pow(2, 4)] * p_c_red[1][(i // pow(2, 5)) % 2 + 2 * ((i // pow(2, 6) % pow(2, 3)))] \
            * p_a[1][(i//pow(2,3)) % 2 + 2 * ((i // pow(2,5)) % pow(2, 5))] * p_a[0][((i // pow(2,4))  % pow(2, 6))] * p_s_pred[i%pow(2,7)]
    for i in range(pow(2,10)):
        p_ssa[i%pow(2,5) + pow(2,5)*(i//pow(2,7))] = p_ssa[i%pow(2,5) + pow(2,5)*(i//pow(2,7))] + p_sca_full[i]
        p_sa[(i//pow(2,3))%pow(2,2) + pow(2,2)*((i//pow(2,7))%pow(2,3))] = p_sa[(i//pow(2,3))%pow(2,2) + pow(2,2)*((i//pow(2,7))%pow(2,3))] + p_sca_full[i]
#    #print("ssa_ext", np.sum(p_ssa))
    for i in range(pow(2,8)):
        if p_sa[i//pow(2,3)]> 0.000000001:
            p_ssa[i] = p_ssa[i] / p_sa[i//pow(2,3)]
        else:
            p_ssa[i] = 1/8
    for i in range(len(p_next)):
        if p_ssa[i%pow(2,5) + pow(2,5)*(i//pow(2,7))]>0.00000000001:
            p_next[i] = p_sca_full[i] * ( p_s_goal[i%pow(2,5) + pow(2,5)*(i//pow(2,7))] / p_ssa[i%pow(2,5) + pow(2,5)*(i//pow(2,7))]   )
        #    #print(p_ssa[i%pow(2,5) + pow(2,5)*(i//pow(2,7))])
        else:
            p_next[i] = p_sca_full[i] * (p_s_goal[i % pow(2, 5) + pow(2, 5) * (i // pow(2, 7))] / (0.00000000001))

    return p_next

def factorizing_sensors(p):
    p_s_pred = np.zeros(pow(2,7))
    p_ca = np.zeros(pow(2,4))

    for i in range(len(p)):
        p_s_pred[(i//(pow(2,4)))%pow(2,7)] = p_s_pred[(i//(pow(2,4)))%pow(2,7)] + p[i]
        p_ca[(i//(pow(2,7)))%pow(2,4)] = p_ca[(i//(pow(2,7)))%pow(2,4)] + p[i]
    for i in range(len(p_s_pred)):
        if p_ca[i // (pow(2, 3))]!=0:
            p_s_pred[i] = p_s_pred[i] / p_ca[i // (pow(2, 3))]
        else:
            p_s_pred[i] = 1 / 8
    return  p_s_pred
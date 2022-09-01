import numpy as np
import AuxiliaryFunctions as af
from math import log2

def IntegratedInformation(p_sca, p_s, p_s_pred, p_c, p_a_f):
    p_c_split = np.zeros((2,pow(2,5)))
    p_c_j = np.zeros((2,pow(2,6)))
    p_c_tj = np.zeros(4)
    p_c_c = np.zeros((2, pow(2,3)))
   # p_c_c = np.zeros(pow(2,4))
    p_c_t = np.zeros(pow(2,2))

    p_s_j = np.zeros(pow(2,8))
    p_sc = np.zeros(pow(2,5))
    p_sc_j = np.zeros((2,pow(2,4)))

    p_scacs_p = np.zeros(pow(2,12))
    p_scacs = np.zeros(pow(2, 12))
    p_full = np.zeros(pow(2,14))
    p_mutful = np.zeros(pow(2,14))

    p_sa = np.zeros(pow(2,5))
    p_s_a = np.zeros(pow(2, 5))

    p_a = np.zeros(pow(2,2))
    p_a_c = np.zeros((2,pow(2,3)))
    p_a_s_n = np.zeros(pow(2,5))

    p_a_n = np.zeros(pow(2,2))
    p_c_n = np.zeros(pow(2,2))
    p_s_n = np.zeros(pow(2,3))

    int_int = 0.0
    int_act = 0.0
    morph = 0.0
    react = 0.0
    sensi = 0.0
    muti = 0.0
    comm = 0.0
    p_goal = [0.0,0.0]
    for i in range(pow(2,7)):
        p_sc[i//pow(2,2)] = p_sc[i//pow(2,2)]  + p_sca[i]
        p_sc_j[0][(i//pow(2,2))%pow(2,3) + pow(2,3)*( i//pow(2,6)) ] = p_sc_j[0][(i//pow(2,2))%pow(2,3) + pow(2,3)*( i//pow(2,6))] + p_sca[i]
        p_sc_j[1][(i//pow(2,2))%pow(2,4)  ] = p_sc_j[1][(i//pow(2,2))%pow(2,4) ] + p_sca[i]

        p_c_t[(i//pow(2,2))%4] = p_c_t[(i//pow(2,2))%4] + p_sca[i]
        p_a[i%4] = p_a[i%4] + p_sca[i]
        p_sa[i%4 + 4*((i//pow(2,4))%pow(2,3))] = p_sa[i%4 + 4*((i//pow(2,4))%pow(2,3))] + p_sca[i]
    for i in range(pow(2,8)):
        p_s_j[i] = p_s[i] * p_sa[i//pow(2,3)]
    for i in range(pow(2,8)):
        p_s_a[i%pow(2,5)] = p_s_a[i%pow(2,5)] + p_s_j[i]
    for i in range(pow(2,5)):
        p_s_a[i] = p_s_a[i] / p_a[i//pow(2,3)]
    for i in range(pow(2,6)):
        for j in range(2):
            p_c_j[j][i] = p_c[j][i]*p_sc[i//2]
    for i in range(pow(2,6)):
        p_c_split[0][i%pow(2,4) +pow(2,4) *(i//pow(2,5))  ] = p_c_split[0][i%pow(2,4) +pow(2,4) *(i//pow(2,5))  ] + p_c_j[0][i]
        p_c_split[1][i % pow(2,5)] = p_c_split[1][i % pow(2,5)] + p_c_j[1][i]
    for i in range(pow(2,5)):
        for j in range(2):
            if p_c_split[j][i] !=0:
                p_c_split[j][i] = p_c_split[j][i] / p_sc_j[j][i//2]
            else:
                p_c_split[j][i] = 0.5
   # ###print("Integrated Infromation", np.sum(p_c_split[0]), np.sum(p_c_split[1]), len(p_c[0]), np.sum(p_c), np.sum(p_c_j), np.sum(p_sc_j[0]), np.sum(p_sc_j[1]), np.sum(p_sc), p_c_split[1])
    for i in range(pow(2,12)):
        p_scacs_p[i] = p_sca[i//pow(2,5)]*p_c[0][(i//2)%pow(2,4) + pow(2,4)*((i//pow(2,7))%pow(2,2))]*p_c[1][i%2 + 2*((i//4)%pow(2,3)) + pow(2,4)*((i//pow(2,7))%pow(2,2))] * \
                       p_s_pred[(i//pow(2,2))%pow(2,7)]
        p_scacs[i] =   p_sca[i//pow(2,5)]*p_c[0][(i//2)%pow(2,4) + pow(2,4)*((i//pow(2,7))%pow(2,2))]*p_c[1][i%2 + 2*((i//4)%pow(2,3)) + pow(2,4)*((i//pow(2,7))%pow(2,2))] * \
                       p_s[(i//pow(2,2))%pow(2,5) + pow(2,5) *(i//pow(2,9))]
    for i in range(pow(2,14)):
        p_full[i] = p_scacs[i//pow(2,2)]*p_a_f[0][(i//2)%pow(2,6)]*p_a_f[1][i%2 + 2*((i//pow(2,2))%pow(2,5))]

    for i in range(pow(2,14)):
        p_c_n[(i//pow(2,2))%pow(2,2)] = p_c_n[(i//pow(2,2))%pow(2,2)] + p_full[i]
        p_s_n[(i//pow(2,4))%pow(2,3)] = p_s_n[(i//pow(2,4))%pow(2,3)] + p_full[i]
        p_a_n[i%pow(2,2)] = p_a_n[i%pow(2,2)] + p_full[i]
        p_a_s_n[(i//pow(2,2))%pow(2,5)] = p_a_s_n[(i//pow(2,2))%pow(2,5)] + p_full[i]
        p_goal[i//pow(2,4)%2] = p_goal[i//pow(2,4)%2] + p_full[i]
    for i in range(pow(2,14)):
        p_a_c[0][(i//2)%pow(2,3)] = p_a_c[0][(i//2)%pow(2,3)] +p_full[i]
        p_a_c[1][(i %2) + 2*((i//pow(2,2))%pow(2,2))] = p_a_c[1][(i %2) + 2*((i//pow(2,2))%pow(2,2))] + p_full[i]
        p_c_tj[(i//pow(2,2))%pow(2,2)] = p_c_tj[(i//pow(2,2))%pow(2,2)] +p_full[i]
      #  p_c_c[(i//pow(2,2))%pow(2,2) + pow(2,2)*((i//pow(2,9))%pow(2,2))] = p_c_c[(i//pow(2,2))%pow(2,2) + pow(2,2)*((i//pow(2,9))%pow(2,2))] +p_full[i]
        p_c_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,9)) % 4)] = p_c_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,9)) % 4)] + p_full[i]
        p_c_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % 4)] = p_c_c[1][(i // pow(2, 2)) % 2 + 2 * ( (i // pow(2, 9)) % 4)] + p_full[i]
        p_mutful[i] = p_sca[i // pow(2,7)] * p_c_n[(i//pow(2,2))%pow(2,2)]* p_s_n[(i//pow(2,4))%pow(2,3)] *p_a_n[i%pow(2,2)]

    for i in range(pow(2,5)):
        p_a_s_n[i] = p_a_s_n[i]/p_s_n[i//pow(2,2)]
 #   ###print("full", np.sum(p_full), np.sum(p_c_c), np.sum(p_c_t), np.sum(p_a_s_n), p_goal)
    for i in range(pow(2,3)):
        for j in range(2):
            p_a_c[j][i] = p_a_c[j][i] / p_c_tj[i//2]
            p_c_c[j][i] = p_c_c[j][i] / p_c_t[i//2]

  #  ###print("full", np.sum(p_full), np.sum(p_c_c[0]), np.sum(p_c_c[1]))
    for i in range(pow(2,12)):
        if p_c[0][(i//2)%pow(2,4) + pow(2,4)*((i//pow(2,7))%pow(2,2))]*p_c[1][i%2 + 2*(i//4)%pow(2,3) + pow(2,4)*((i//pow(2,7))%pow(2,2))] >0 :
          #  ###print("int_inf", p_c[0][(i//2)%pow(2,4) + pow(2,4)*((i//pow(2,7))%pow(2,2))]*p_c[1][i%2 + 2*(i//4)%pow(2,3) + pow(2,4)*((i//pow(2,7))%pow(2,2))],p_c_split[0][(i//2)%pow(2,4) + pow(2,4)*((i//pow(2,8))%pow(2,1))]* p_c_split[1][i%2 + 2*(i//4)%pow(2,3) + pow(2,4)*((i//pow(2,7))%pow(2,1))] )
            diff = (log2(p_c[0][(i//2)%pow(2,4) + pow(2,4)*((i//pow(2,7))%pow(2,2))]*p_c[1][i%2 + 2*(i//4)%pow(2,3) + pow(2,4)*((i//pow(2,7))%pow(2,2))])\
            - log2(p_c_split[0][(i//2)%pow(2,4) + pow(2,4)*((i//pow(2,8))%pow(2,1))]*  p_c_split[1][i%2 + 2*(i//4)%pow(2,3) + pow(2,4)*((i//pow(2,7))%pow(2,1))]))
        else:
            diff = 0
            ###print("else")
        int_int = int_int + p_scacs_p[i] *  diff
        int_act = int_act + p_scacs[i] * diff
        if p_c[0][(i//2)%pow(2,4) + pow(2,4)*((i//pow(2,7))%pow(2,2))]*p_c[1][i%2 + 2*(i//4)%pow(2,3) + pow(2,4)*((i//pow(2,7))%pow(2,2))] > 0 :
            sensi = sensi + p_scacs[i] * (log2(p_c[0][(i//2)%pow(2,4) + pow(2,4)*((i//pow(2,7))%pow(2,2))]*p_c[1][i%2 + 2*(i//4)%pow(2,3) + pow(2,4)*((i//pow(2,7))%pow(2,2))]) \
                                        - log2(p_c_c[0][(i//pow(2,1))%2 + 2*((i//pow(2,7)) % 4)]* p_c_c[1][i % 2 + 2 * ((i // pow(2, 7)) % 4)]))
           # ###print(sensi, p_c[0][(i//2)%pow(2,4) + pow(2,4)*((i//pow(2,7))%pow(2,2))]*p_c[1][i%2 + 2*(i//4)%pow(2,3) + pow(2,4)*((i//pow(2,7))%pow(2,2))],p_c_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,9)) % 4)]* p_c_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % 4)])
    for i in range(pow(2,10)):
        if p_s[i%pow(2,5) + pow(2,5)*((i//pow(2,7))%pow(2,3))] > 0:
            morph = morph + p_sca[i//pow(2,3)] * p_s[i%pow(2,5) + pow(2,5)*((i//pow(2,7))%pow(2,3))] * (log2(p_s[i%pow(2,5) + pow(2,5)*((i//pow(2,7))%pow(2,3))]) -  log2(p_s_a[i%pow(2,5)]) )

    for i in range(pow(2,14)):
      #  ###print(p_a_f[0][(i//2)%pow(2,6)] )
     #   p_test[i] = p_full[i]* p_a_c[0][(i//2)%pow(2,3)] * p_a_c[1][i%2 + 2*((i//pow(2,2))%pow(2,2))]
        if  p_full[i] > 0:
            muti = muti + p_full[i]*(log2(p_full[i])- log2(p_mutful[i]))
        if  p_a_f[0][(i//2)%pow(2,6)] * p_a_f[1][i%2 + 2*((i//pow(2,2))%pow(2,5))]* p_a_c[0][(i//2)%pow(2,3)] * p_a_c[1][i%2 + 2*((i//pow(2,2))%pow(2,2))]> 0:
            react = react + p_full[i] *(log2(p_a_f[0][(i//2)%pow(2,6)] * p_a_f[1][i%2 + 2*((i//pow(2,2))%pow(2,5))]) - log2(p_a_c[0][(i//2)%pow(2,3)] * p_a_c[1][i%2 + 2*((i//pow(2,2))%pow(2,2))]))
           # ###print("react", react)
       # ###print(p_a_s_n[(i//pow(2,2))%pow(2,5)])
        if p_a_f[0][(i//2)%pow(2,6)] * p_a_f[1][i%2 + 2*((i//pow(2,2))%pow(2,5))] > 0:
            comm = comm + p_full[i] * (log2(p_a_f[0][(i//2)%pow(2,6)] * p_a_f[1][i%2 + 2*((i//pow(2,2))%pow(2,5))]) - log2(p_a_s_n[(i//pow(2,2))%pow(2,5)]))

        else:
            print("else")
  #  ###print("Integrated Infromation", int_int,  morph, react, sensi, muti, comm, p_goal[1])
    return int_int, int_act, morph, react, sensi, muti, comm, p_goal[1]


def calc_meas(p_sca, p_s, p_s_pred, p_c, p_a):
    p_ct = np.zeros(pow(2,2))
    p_cj_scj = np.zeros((2, pow(2,5)))
    p_sncj = np.zeros((2, pow(2, 4)))

    p_cn_c = np.zeros((2, pow(2,3)))
    p_sn_a = np.zeros(pow(2,5))

    p_full = np.zeros(pow(2,14))
    p_mutful = np.zeros(pow(2,14))

    p_full_predict = np.zeros(pow(2,10))
    p_marg_predict = np.zeros(pow(2,7))
    p_s_c_a_pred = np.zeros(pow(2,5))
    p_s_c_c_pred = np.zeros(pow(2,5))
    p_sa = np.zeros(pow(2,5))
    p_ca = np.zeros(pow(2,5))

    p_at = np.zeros(pow(2,2))
    p_an_c = np.zeros((2, pow(2,3)))
    p_an_s = np.zeros((2, pow(2,4)))

    p_a_n = np.zeros(pow(2,2))
    p_c_n = np.zeros(pow(2,2))
    p_s_n = np.zeros(pow(2,3))
    p_s_pred2 = np.zeros(pow(2,7))
    p_s_pred2_sgoal = np.zeros(pow(2,6))


    p_goal = [0.0,0.0]
    p_goal_pred = [0.0,0.0]


    for i in range(pow(2,14)):
        p_full[i] = p_sca[i//pow(2,7)]*p_c[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,9))%pow(2,2))]*\
                    p_c[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2,4)*((i//pow(2,9))%pow(2,2))] * \
                    p_s[(i//pow(2,4))%pow(2,5) + pow(2,5) *(i//pow(2,11))]*\
                    p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))]
    #distr on var only in time point t
    for i in range(pow(2,10)):
        p_full_predict[i] = p_sca[i//pow(2,3)] * p_s_pred[i%pow(2,7)]
    for i in range(pow(2,10)):
        p_s_c_a_pred[i%pow(2,5)] = p_s_c_a_pred[i%pow(2,5)] + p_full_predict[i]
        p_s_c_c_pred[i%pow(2,3) + pow(2,3)*((i//pow(2,5))%pow(2,2))] = p_s_c_c_pred[i%pow(2,3) + pow(2,3)*((i//pow(2,5))%pow(2,2))] + p_full_predict[i]
        p_s_pred2[i%pow(2,7)] = p_s_pred2[i%pow(2,7)] + p_full_predict[i]
        p_s_pred2_sgoal[(i//2)%pow(2,6)] = p_s_pred2_sgoal[(i//2)%pow(2,6)] + p_full_predict[i]
        p_goal_pred[i%2] = p_goal_pred[i%2] + p_full_predict[i]

    p_marg_predict = np.copy(p_s_pred2)

    for i in range(pow(2,7)):
        p_at[i%pow(2,2)] = p_at[i%pow(2,2)] + p_sca[i]
        p_ct[(i//pow(2,2))%pow(2,2)] = p_ct[(i//pow(2,2))%pow(2,2)] + p_sca[i]
        p_sa[i%pow(2,2) + pow(2,2) * ((i//pow(2,4))%pow(2,3))] =  p_sa[i%pow(2,2) + pow(2,2) * ((i//pow(2,4))%pow(2,3))] + p_sca[i]
        p_ca[i%pow(2,4)] = p_ca[i%pow(2,4)] + p_sca[i]
    #distr on both var
    ####print("pred", np.sum(p_s_c_a_pred), np.sum(p_s_c_c_pred))
    for i in range(pow(2,5)):
        p_s_c_a_pred[i] = p_s_c_a_pred[i]/ p_at[i//pow(2,3)]
        p_s_c_c_pred[i] = p_s_c_c_pred[i]/ p_ct[i//pow(2,3)]
    for i in range(pow(2,7)):
        p_s_pred2[i] = p_s_pred2[i] /p_ca[i//pow(2,3)]

    for i in range(pow(2,6)):
        p_s_pred2_sgoal[i] = p_s_pred2_sgoal[i] /p_ca[i//pow(2,2)]

    for i in range(pow(2,14)):
        p_cj_scj[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%2)] =p_cj_scj[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%2)] + p_full[i]
        p_cj_scj[1][(i // pow(2, 2)) % 2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)] = p_cj_scj[1][(i // pow(2, 2)) % 2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)]+ p_full[i]
        p_sncj[0][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,10))%2)] = p_sncj[0][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,10))%2)]+ p_full[i]
        p_sncj[1][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,9))%2)] = p_sncj[1][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,9))%2)] + p_full[i]

        p_sn_a[(i//pow(2,4))%pow(2,5)] = p_sn_a[(i//pow(2,4))%pow(2,5)] +p_full[i]

        p_a_n[i%pow(2,2)] = p_a_n[i%pow(2,2)] +p_full[i]
        p_s_n[(i//pow(2,4))%pow(2,3)] = p_s_n[(i//pow(2,4))%pow(2,3)] + p_full[i]
        p_c_n[(i//pow(2,2))%pow(2,2)] = p_c_n[(i//pow(2,2))%pow(2,2)] + p_full[i]

        p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,9))%pow(2,2))] = p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,9))%pow(2,2))] +p_full[i]
        p_cn_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 2))] = p_cn_c[1][(i//pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 2))] + p_full[i]

        p_an_c[0][(i//2)%pow(2,3)] = p_an_c[0][(i//2)%pow(2,3)] +p_full[i]
        p_an_c[1][ i%2 + 2*((i // 4) % pow(2, 2))] = p_an_c[1][ i%2 + 2*((i // 4) % pow(2, 2))]+ p_full[i]

        p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))] = p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))] + p_full[i]
        p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))] = p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))] + p_full[i]
        p_goal[(i // pow(2, 4)) % 2] = p_goal[(i // pow(2, 4) )% 2] + p_full[i]
    for i in range(pow(2,14)):
        p_mutful[i] = p_sca[i // pow(2, 7)] * p_c_n[(i // pow(2, 2)) % pow(2, 2)] * p_s_n[(i // pow(2, 4)) % pow(2, 3)] * p_a_n[i % pow(2, 2)]
 #   ###print("full2", np.sum(p_full), np.sum(p_cj_scj), np.sum(p_sncj), np.sum(p_a_n), np.sum(p_s_n), np.sum(p_c_n), p_goal)
    #conditioning
    for i in range(pow(2,5)):
        for j in range(2):
            if p_sncj[j][i//2] > 0:
                p_cj_scj[j][i] = p_cj_scj[j][i] / p_sncj[j][i//2]
            else:
                p_cj_scj[j][i] = 0.5
        if p_at[i//pow(2,3)] > 0:
            p_sn_a[i] = p_sn_a[i] / p_at[i//pow(2,3)]
        else:
            p_sn_a[i] = 0.5

    for i in range(pow(2,4)):
        for j in range(2):
            if p_s_n[i//2] > 0:
                p_an_s[j][i] = p_an_s[j][i] / p_s_n[i//2]
 #   ###print("t", np.sum(p_an_s), np.sum(p_s_n))
    for i in range(pow(2,3)):
        for j in range(2):
            if p_c_n[i//2]>0:
                p_an_c[j][i] = p_an_c[j][i] / p_c_n[i//2]
            else:
                p_an_c[j][i] = 0.5
            if p_ct[i//2]>0:
                p_cn_c[j][i] =  p_cn_c[j][i] / p_ct[i//2]
            else:
                p_cn_c[j][i] = 0.5
   # ###print("t", np.sum(p_cn_c))
    int_int = 0.0
    morph = 0.0
    react = 0.0
    sensi = 0.0
    muti = 0.0
    comm = 0.0
    pred1 = 0.0
    pred2 = 0.0
    syn = 0.0
    predfull = 0.0
    predgoal =  0.0

    for i in range(pow(2,7)):
        if p_s_pred2[ i]*p_s_c_a_pred[i%pow(2,5)] > 0:
            pred1 = pred1+ p_marg_predict[i] * (log2(p_s_pred2[ i]/p_s_c_a_pred[i%pow(2,5)]))
        if p_s_pred[ i%pow(2,7)]* p_s_c_c_pred[i%pow(2,3) + pow(2,3)*((i//pow(2,5))%pow(2,2))] > 0:
            pred2 = pred2 + p_marg_predict[i] * (log2(p_s_pred2[ i] /p_s_c_c_pred[i%pow(2,3) + pow(2,3)*(i//pow(2,5))]))
        if p_s_pred2[i % pow(2, 7)] * p_s_n[i % pow(2, 3)] > 0:
            predfull = predfull + p_marg_predict[i] * (log2(p_s_pred2[i]/p_s_n[i % pow(2, 3)]))
        if p_s_pred2[i % pow(2, 7)] * p_s_n[i % pow(2, 3)] > 0:
            predgoal = predgoal + p_marg_predict[i] * (log2(p_s_pred2[i]/(p_s_pred2_sgoal[(i//2)%pow(2,6)]* p_goal_pred[i%2])))

    ###print(pred2, np.sum(p_full_predict), np.sum(p_s_pred), np.sum(p_s_c_c_pred))
    for i in range(pow(2,14)):
  #      ###print(p_cj_scj[0][(i//2)%pow(2,4) + pow(2,4)*((i//pow(2,8))%pow(2,1))]*  p_cj_scj[1][i%2 + 2*(i//4)%pow(2,3) + pow(2,4)*((i//pow(2,7))%pow(2,1))])

        if (p_c[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,9))%pow(2,2))]*p_c[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3) )+ pow(2,4)*((i//pow(2,9))%pow(2,2))]*p_cj_scj[0][(i // pow(2, 3)) % pow(2, 4) + pow(2, 4) * ((i // pow(2, 10)) % 2)]*  p_cj_scj[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 4)) % pow(2, 3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)]) >0 :
                int_int = int_int + p_full[i]* (log2(p_c[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,9))%pow(2,2))]*p_c[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2,4)*((i//pow(2,9))%pow(2,2))]) -
                                                log2(p_cj_scj[0][(i // pow(2, 3)) % pow(2, 4) + pow(2, 4) * ((i // pow(2, 10)) % 2)]*  p_cj_scj[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 4)) % pow(2, 3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)]))

        if (p_c[0][(i // pow(2, 3)) % pow(2, 4) + pow(2, 4) * ((i // pow(2, 9)) % pow(2, 2))] * p_c[1][ (i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 4)) % pow(2, 3)) + pow(2, 4) * ( (i // pow(2, 9)) % pow(2, 2))]*p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,9))%pow(2,2))]*  p_cn_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 2))]) > 0:
            sensi = sensi + p_full[i]*  (log2(p_c[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,9))%pow(2,2))]*p_c[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2,4)*((i//pow(2,9))%pow(2,2))]) -log2(p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,9))%pow(2,2))]*  p_cn_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 2))]))

        if  p_s[(i//pow(2,4))%pow(2,5) + pow(2,5)*(i//pow(2,11))] > 0:
         #   ###print(p_sn_a[(i//pow(2,4))%pow(2,5)] )
            morph = morph + p_full[i] * (log2(p_s[(i//pow(2,4))%pow(2,5) + pow(2,5)*(i//pow(2,11))])-log2(p_sn_a[(i//pow(2,4))%pow(2,5)] ) )
        if p_full[i]*p_mutful[i] > 0:
            muti = muti + p_full[i]*(log2(p_full[i])- log2(p_mutful[i]))
        if p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))]* p_an_c[0][(i//2)%pow(2,3)] *p_an_c[1][i%2 + 2*((i//pow(2,2))%pow(2,2))] >0:
            react = react + p_full[i]*(log2(p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))])-
                                       log2(p_an_c[0][(i//2)%pow(2,3)] *p_an_c[1][i%2 + 2*((i//pow(2,2))%pow(2,2))] ) )
        if p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))]   * p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))]*p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))]  > 0:
            comm = comm + p_full[i] * (log2(p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))])-
                                       log2(p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))]   * p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))] ))
       # else:
   #         ###print("else")
   # ###print("Integrated Infromation2", int_int, morph, react, sensi, muti, comm, p_goal[1])


    #iterative scaling
    p_it = np.zeros(pow(2,10)) + (1/pow(2,10))
 #   ###print(p_it)
    iterations = 20
    diff = 1
    while diff > 0.000001:
        ####print("0", p_it)
        oldp = np.copy(p_it)
        pit_st_at = np.zeros(pow(2,5))
        for j in range(pow(2,10)):
            pit_st_at[(j//pow(2,3))%pow(2,2) + pow(2,2)*((j//pow(2,7)))] = pit_st_at[(j//pow(2,3))%pow(2,2) + pow(2,2)*((j//pow(2,7)))] + p_it[j]
        for j in range(pow(2,10)):
            p_it[j] = p_it[j]*(p_sa[(j//pow(2,3))%pow(2,2) + pow(2,2)*((j//pow(2,7)))]/ pit_st_at[(j//pow(2,3))%pow(2,2) + pow(2,2)*((j//pow(2,7)))])
     #       ###print(p_sa[(j//pow(2,3))%pow(2,2) + pow(2,2)*((j//pow(2,7)))]/ pit_st_at[(j//pow(2,3))%pow(2,2) + pow(2,2)*((j//pow(2,7)))])
        ###print("first proj", np.sum(p_it))
        ###print(af.kl(p_sa, pit_st_at))
        ###print("1", p_it)
        pit_s_a = np.zeros(pow(2,5))
        pit_a = np.zeros(pow(2,2))
        for j in range(pow(2,10)):
            pit_s_a[j%pow(2,5)] = pit_s_a[j%pow(2,5)] + p_it[j]
            pit_a[(j//pow(2,3))%pow(2,2)] = pit_a[(j//pow(2,3))%pow(2,2)] + p_it[j]
        for j in range(pow(2,5)):
            pit_s_a[j] = pit_s_a[j] / pit_a[j//pow(2,3)]
        ###print("second proj0", np.sum(p_s_c_a_pred), np.sum(pit_s_a))
        ###print(af.kl(p_full_predict, p_it))
        for j in range(pow(2,10)):
            p_it[j] = p_it[j]*(p_s_c_a_pred[j%pow(2,5)]/ pit_s_a[j%pow(2,5)])
        ###print("second proj", np.sum(p_it))
     #   ###print("2",)
        ###print(af.kl(p_s_c_a_pred, pit_s_a))

        pit_s_c = np.zeros(pow(2,5))
        pit_c = np.zeros(pow(2,2))
        for j in range(pow(2,10)):
            pit_s_c[j%pow(2,3) + pow(2,3)*((j//pow(2,5))%pow(2,2))] = pit_s_c[j%pow(2,3) + pow(2,3)*((j//pow(2,5))%pow(2,2))] + p_it[j]
            pit_c[((j//pow(2,5))%pow(2,2))] = pit_c[((j//pow(2,5))%pow(2,2))] + p_it[j]
        for j in range(pow(2,5)):
            pit_s_c[j] = pit_s_c[j] / pit_c[j//pow(2,3)]
        for j in range(pow(2,10)):
            p_it[j] = p_it[j]*(p_s_c_c_pred[j%pow(2,3) + pow(2,3)*((j//pow(2,5))%pow(2,2))]/ pit_s_c[j%pow(2,3) + pow(2,3)*((j//pow(2,5))%pow(2,2))])
        ###print("3", af.kl(p_s_c_c_pred, pit_s_c))
        diff = af.kl(oldp, p_it)
        ###print("third proj", diff)
       # ###print("3", p_it)
    ###print("Synergisic", af.kl(p_full_predict, p_it))
    syn = af.kl(p_full_predict, p_it)
    ###print(syn)
   # exit(0)
  #  ###print( int_int, morph, react, sensi, muti, comm, p_goal[1], pred1, pred2, syn, predfull, predgoal)
    return int_int, morph, react, sensi, muti, comm, p_goal[1], pred1, pred2, syn,  predfull, predgoal

def calc_meas_noint(p_sca, p_s, p_s_pred, p_c_i, p_a):
  #  ###print("calculate measures", np.sum(p_sca), np.sum(p_s), np.sum(p_c_i), np.sum(p_a))
    p_ct = np.zeros((2,2))
    p_ct2 = np.zeros(pow(2, 2))
    p_cj_scj = np.zeros((2, pow(2,5)))
    p_sncj = np.zeros((2, pow(2, 4)))

    p_cn_c = np.zeros((2, pow(2,2)))
    p_sn_a = np.zeros(pow(2,5))

    p_full = np.zeros(pow(2,14))
    p_mutful = np.zeros(pow(2,14))

    p_at = np.zeros(pow(2,2))
    p_an_c = np.zeros((2, pow(2,3)))
    p_an_s = np.zeros((2, pow(2,4)))

    p_a_n = np.zeros(pow(2,2))
    p_c_n = np.zeros(pow(2,2))
    p_s_n = np.zeros(pow(2,3))
    p_sa = np.zeros(pow(2, 5))


    p_ca = np.zeros(pow(2, 5))


    p_s_pred2_sgoal = np.zeros(pow(2, 6))

    p_full_predict = np.zeros(pow(2,10))
    p_s_c_a_pred = np.zeros(pow(2,5))
    p_s_c_c_pred = np.zeros(pow(2,5))


    p_goal_pred = [0.0, 0.0]
    p_goal = [0.0,0.0]


    for i in range(pow(2,14)):
        p_full[i] = p_sca[i//pow(2,7)]*p_c_i[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%pow(2,1))]*\
                    p_c_i[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2,4)*((i//pow(2,9))%pow(2,1))] * \
                    p_s[(i//pow(2,4))%pow(2,5) + pow(2,5) *(i//pow(2,11))]* \
                    p_a[0][(i // 2) % pow(2, 6)] * p_a[1][i % 2 + 2 * ((i // pow(2, 2)) % pow(2, 5))]
    #distr on var only in time point t
    for i in range(pow(2,7)):
        p_at[i%pow(2,2)] = p_at[i%pow(2,2)] + p_sca[i]
        p_ct2[(i // pow(2, 2)) % pow(2, 2)] = p_ct2[(i // pow(2, 2)) % pow(2, 2)] + p_sca[i]
        p_ct[0][(i//pow(2,3))%pow(2,1)] = p_ct[0][(i//pow(2,3))%pow(2,1)] + p_sca[i]
        p_ct[1][(i//pow(2,2))%pow(2,1)] = p_ct[1][(i//pow(2,2))%pow(2,1)] + p_sca[i]
        p_sa[i % pow(2, 2) + pow(2, 2) * ((i // pow(2, 4)) % pow(2, 3))] = p_sa[i % pow(2, 2) + pow(2, 2) * ( (i // pow(2, 4)) % pow(2, 3))] + p_sca[i]
        p_ca[i % pow(2, 4)] = p_ca[i % pow(2, 4)] + p_sca[i]

    for i in range(pow(2, 10)):
        p_full_predict[i] = p_sca[i // pow(2, 3)] * p_s_pred[i % pow(2, 7)]
    for i in range(pow(2, 10)):
        p_s_c_a_pred[i % pow(2, 5)] = p_s_c_a_pred[i % pow(2, 5)] + p_full_predict[i]
        p_s_c_c_pred[i % pow(2, 3) + pow(2, 3) * ((i // pow(2, 5)) % pow(2, 2))] = p_s_c_c_pred[ i % pow(2, 3) + pow(2, 3) * (( i // pow(2, 5)) % pow(2, 2))] + p_full_predict[i]
        p_s_pred2_sgoal[(i // 2) % pow(2, 6)] = p_s_pred2_sgoal[(i // 2) % pow(2, 6)] + p_full_predict[i]
        p_goal_pred[i % 2] = p_goal_pred[i % 2] + p_full_predict[i]
    for i in range(pow(2,4)):
        p_s_c_a_pred[i] = p_s_c_a_pred[i]/ p_at[i//pow(2,3)]
        p_s_c_c_pred[i] = p_s_c_c_pred[i]/ p_ct2[i//pow(2,3)]
    for i in range(pow(2,6)):
        p_s_pred2_sgoal[i] = p_s_pred2_sgoal[i] /p_ca[i//pow(2,2)]
    #distr on both var
    for i in range(pow(2,14)):
        p_cj_scj[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%2)] =p_cj_scj[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%2)] + p_full[i]
        p_cj_scj[1][(i // pow(2, 2)) % 2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)] = p_cj_scj[1][(i // pow(2, 2)) % 2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)]+ p_full[i]
        p_sncj[0][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,10))%2)] = p_sncj[0][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,10))%2)]+ p_full[i]
        p_sncj[1][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,9))%2)] = p_sncj[1][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,9))%2)] + p_full[i]

        p_sn_a[(i//pow(2,4))%pow(2,5)] = p_sn_a[(i//pow(2,4))%pow(2,5)] +p_full[i]

        p_a_n[i%pow(2,2)] = p_a_n[i%pow(2,2)] +p_full[i]
        p_s_n[(i//pow(2,4))%pow(2,3)] = p_s_n[(i//pow(2,4))%pow(2,3)] + p_full[i]
        p_c_n[(i//pow(2,2))%pow(2,2)] = p_c_n[(i//pow(2,2))%pow(2,2)] + p_full[i]

        p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,10))%pow(2,1))] = p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,10))%pow(2,1))] +p_full[i]
        p_cn_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 1))] = p_cn_c[1][(i//pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 1))] + p_full[i]

        p_an_c[0][(i//2)%pow(2,3)] = p_an_c[0][(i//2)%pow(2,3)] +p_full[i]
        p_an_c[1][ i%2 + 2*((i // 4) % pow(2, 2))] = p_an_c[1][ i%2 + 2*((i // 4) % pow(2, 2))]+ p_full[i]

        p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))] = p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))] + p_full[i]
        p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))] = p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))] + p_full[i]
        p_goal[(i // pow(2, 4)) % 2] = p_goal[(i // pow(2, 4) )% 2] + p_full[i]
    for i in range(pow(2,14)):
        p_mutful[i] = p_sca[i // pow(2, 7)] * p_c_n[(i // pow(2, 2)) % pow(2, 2)] * p_s_n[(i // pow(2, 4)) % pow(2, 3)] * p_a_n[i % pow(2, 2)]
 #   ###print("full2", np.sum(p_full), np.sum(p_cj_scj), np.sum(p_sncj), np.sum(p_a_n), np.sum(p_s_n), np.sum(p_c_n), p_goal)
    #conditioning

    for i in range(pow(2,5)):
        for j in range(2):
            if p_sncj[j][i//2] > 0:
                p_cj_scj[j][i] = p_cj_scj[j][i] / p_sncj[j][i//2]
            else:
                p_cj_scj[j][i] = 0.5
        if p_at[i//pow(2,3)] > 0:
            p_sn_a[i] = p_sn_a[i] / p_at[i//pow(2,3)]
        else:
            p_sn_a[i] = 0.5

    for i in range(pow(2,4)):
        for j in range(2):
            if p_s_n[i//2] > 0:
                p_an_s[j][i] = p_an_s[j][i] / p_s_n[i//2]
 #   ###print("t", np.sum(p_an_s), np.sum(p_s_n))
    for i in range(pow(2,3)):
        for j in range(2):
            if p_c_n[i//2]>0:
                p_an_c[j][i] = p_an_c[j][i] / p_c_n[i//2]
            else:
                p_an_c[j][i] = 0.5
    for i in range(pow(2, 2)):
        for j in range(2):
            if p_ct[j][i//2]>0:
                p_cn_c[j][i] =  p_cn_c[j][i] / p_ct[j][i//2]
            else:
                p_cn_c[j][i] = 0.5
   # ###print("t", np.sum(p_cn_c))
    int_int = 0.0
    morph = 0.0
    react = 0.0
    sensi = 0.0
    muti = 0.0
    comm = 0.0

    pred1 = 0.0
    pred2 = 0.0
    predfull = 0.0
    predgoal = 0.0
    for i in range(pow(2, 10)):
        if p_s_pred[i % pow(2, 7)] * p_s_c_a_pred[i % pow(2, 5)] > 0:
            pred1 = pred1 + p_full_predict[i] * (log2(p_s_pred[i % pow(2, 7)]) - log2(p_s_c_a_pred[i % pow(2, 5)]))
        if p_s_pred[i % pow(2, 7)] * p_s_c_c_pred[i % pow(2, 3) + pow(2, 3) * ((i // pow(2, 5)) % pow(2, 2))] > 0:
            pred2 = pred2 + p_full_predict[i] * (log2(p_s_pred[i % pow(2, 7)]) - log2(
                p_s_c_c_pred[i % pow(2, 3) + pow(2, 3) * ((i // pow(2, 5)) % pow(2, 2))]))
        if p_s_pred[i % pow(2, 7)] * p_s_n[i % pow(2, 3)] > 0:
            predfull = predfull + p_full_predict[i] * (log2(p_s_pred[i % pow(2, 7)]) - log2(p_s_n[i % pow(2, 3)]))
        if p_s_pred[i % pow(2, 7)] * p_s_n[i % pow(2, 3)] > 0:
            predgoal = predgoal +  p_full_predict[i] * (log2(p_s_pred[i%pow(2,7)]/(p_s_pred2_sgoal[(i//2)%pow(2,6)]* p_goal_pred[i%2])))

    ###print( "pred", np.sum(p_full_predict), np.sum(p_s_pred), np.sum(p_s_c_a_pred), np.sum(p_s_c_c_pred), np.sum(p_s_n))
    for i in range(pow(2,14)):
  #      ###print(p_cj_scj[0][(i//2)%pow(2,4) + pow(2,4)*((i//pow(2,8))%pow(2,1))]*  p_cj_scj[1][i%2 + 2*(i//4)%pow(2,3) + pow(2,4)*((i//pow(2,7))%pow(2,1))])

        if (p_c_i[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%pow(2,1))]*p_c_i[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3) )+ pow(2,4)*((i//pow(2,9))%pow(2,1))]*p_cj_scj[0][(i // pow(2, 3)) % pow(2, 4) + pow(2, 4) * ((i // pow(2, 10)) % 2)]*  p_cj_scj[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 4)) % pow(2, 3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)]) >0 :
                int_int = int_int + p_full[i]* (log2(p_c_i[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%pow(2,1))]*p_c_i[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2,4)*((i//pow(2,9))%pow(2,1))]) -
                                                log2(p_cj_scj[0][(i // pow(2, 3)) % pow(2, 4) + pow(2, 4) * ((i // pow(2, 10)) % 2)]*  p_cj_scj[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 4)) % pow(2, 3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)]))
        #else:
        #    int_int =
        if (p_c_i[0][(i // pow(2, 3)) % pow(2, 4) + pow(2, 4) * ((i // pow(2, 10)) % pow(2, 1))] * p_c_i[1][ (i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 4)) % pow(2, 3)) + pow(2, 4) * ( (i // pow(2, 9)) % pow(2, 1))]*p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,10))%pow(2,1))]*  p_cn_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 1))]) > 0:
            sensi = sensi + p_full[i]*  (log2(p_c_i[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%pow(2,1))]*p_c_i[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2,4)*((i//pow(2,9))%pow(2,1))]) -log2(p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,10))%pow(2,1))]*  p_cn_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 1))]))

        if  p_s[(i//pow(2,4))%pow(2,5) + pow(2,5)*(i//pow(2,11))] > 0:
         #   ###print(p_sn_a[(i//pow(2,4))%pow(2,5)] )
            morph = morph + p_full[i] * (log2(p_s[(i//pow(2,4))%pow(2,5) + pow(2,5)*(i//pow(2,11))])-log2(p_sn_a[(i//pow(2,4))%pow(2,5)] ) )
        if p_full[i]*p_mutful[i] > 0:
            muti = muti + p_full[i]*(log2(p_full[i])- log2(p_mutful[i]))
        if p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))]* p_an_c[0][(i//2)%pow(2,3)] *p_an_c[1][i%2 + 2*((i//pow(2,2))%pow(2,2))] >0:
            react = react + p_full[i]*(log2(p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))])-
                                       log2(p_an_c[0][(i//2)%pow(2,3)] *p_an_c[1][i%2 + 2*((i//pow(2,2))%pow(2,2))] ) )
        if p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))]   * p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))]*p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))]  > 0:
            comm = comm + p_full[i] * (log2(p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))])-
                                       log2(p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))]   * p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))] ))
       # else:
   #         ###print("else")
   # ###print("Integrated Infromation2", int_int, morph, react, sensi, muti, comm, p_goal[1])


    #iterative scaling
    p_it = np.zeros(pow(2,10)) + (1/pow(2,10))
 #   ###print(p_it)
    iterations = 20
    diff = 1
    while diff > 0.000001:
        oldp = np.copy(p_it)
        ####print("0", p_it)
        pit_st_at = np.zeros(pow(2,5))
        for j in range(pow(2,10)):
            pit_st_at[(j//pow(2,3))%pow(2,2) + pow(2,2)*((j//pow(2,7)))] = pit_st_at[(j//pow(2,3))%pow(2,2) + pow(2,2)*((j//pow(2,7)))] + p_it[j]
        for j in range(pow(2,10)):
            p_it[j] = p_it[j]*(p_sa[(j//pow(2,3))%pow(2,2) + pow(2,2)*((j//pow(2,7)))]/ pit_st_at[(j//pow(2,3))%pow(2,2) + pow(2,2)*((j//pow(2,7)))])
      #      ###print(p_sa[(j//pow(2,3))%pow(2,2) + pow(2,2)*((j//pow(2,7)))]/ pit_st_at[(j//pow(2,3))%pow(2,2) + pow(2,2)*((j//pow(2,7)))])
      #  ###print("first proj", np.sum(p_it))

      #  ###print("1", p_it)
        pit_s_a = np.zeros(pow(2,5))
        pit_a = np.zeros(pow(2,2))
        for j in range(pow(2,10)):
            pit_s_a[j%pow(2,5)] = pit_s_a[j%pow(2,5)] + p_it[j]
            pit_a[(j//pow(2,3))%pow(2,2)] = pit_a[(j//pow(2,3))%pow(2,2)] + p_it[j]
        for j in range(pow(2,5)):
            pit_s_a[j] = pit_s_a[j] / pit_a[j//pow(2,3)]
      #  ###print("second proj0", np.sum(p_s_c_a_pred), np.sum(pit_s_a))
        ###print(af.kl(p_full_predict, p_it))
        for j in range(pow(2,10)):
            p_it[j] = p_it[j]*(p_s_c_a_pred[j%pow(2,5)]/ pit_s_a[j%pow(2,5)])
      #  ###print("second proj", np.sum(p_it))
      #  ###print("2", p_it)
      #  ###print(af.kl(p_full_predict, p_it))

        pit_s_c = np.zeros(pow(2,5))
        pit_c = np.zeros(pow(2,2))
        for j in range(pow(2,10)):
            pit_s_c[j%pow(2,3) + pow(2,3)*((j//pow(2,5))%pow(2,2))] = pit_s_c[j%pow(2,3) + pow(2,3)*((j//pow(2,5))%pow(2,2))] + p_it[j]
            pit_c[((j//pow(2,5))%pow(2,2))] = pit_c[((j//pow(2,5))%pow(2,2))] + p_it[j]
        for j in range(pow(2,5)):
            pit_s_c[j] = pit_s_c[j] / pit_c[j//pow(2,3)]
        for j in range(pow(2,10)):
            p_it[j] = p_it[j]*(p_s_c_c_pred[j%pow(2,3) + pow(2,3)*((j//pow(2,5))%pow(2,2))]/ pit_s_c[j%pow(2,3) + pow(2,3)*((j//pow(2,5))%pow(2,2))])
        diff = af.kl(oldp, p_it)
     #   ###print("third proj", np.sum(p_it))
       # ###print("3", p_it)
   # ###print("Synergisic", af.kl(p_full_predict, p_it))
    syn = af.kl(p_full_predict, p_it)
    return int_int, morph, react, sensi, muti, comm, p_goal[1], pred1, pred2, syn, predfull, predgoal

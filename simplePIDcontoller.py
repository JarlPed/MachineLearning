# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:36:22 2018

@author: jarl

Simple PID controller for the PoleCart
"""

import gym
import scipy

import numpy as np



dtype = torch.float
device = torch.device("cpu")



def PIDu (err, K, n_int): return K[0]*err[0]+K[1]*scipy.integrate.simps(err[-n_int:]) +K[2]*(err[-1]-err[-2])





env = gym.make('CartPole-v1')
env.spec

# Initial Control parameters and weights

#K = [[1.0, 0.0, 1.0],[1.0, 0.0, 1.0],[1.0, 0.0, 1.0],[1.0, 0.0, 1.0]]
weights = [1.0, 1.0, 1.0, 1.0]
n_int = 200
 
# Solution a semi PD controller
K = [[1.0, 0.1, 1.0],[1.0, 0.1, 1.0],[1.0, 0.1, 1.0],[1.0, 0.1, 1.0]]


nnInput = 3*4+4+1


inputVec = []
for element in weights:
    inputVec.append(element)
for element in K:
    for entry in element:
        inputVec.append(entry)
inputVec.append(n_int)




def TestPIDPoleCart(matrix, batch_size):
    
    
    weights = matrix[0:4]
    n_int = int(matrix[-1])
    K = []
    for i in range(len(weights)):
        K.append(matrix[4+3*i:4+3*(i+1)])
        
    reward_array = []    
    
    for r in range(batch_size):
    
        # First a forward push
        env.reset()
        state, reward, done, _ = env.step(1)
    
        CartPos = [0.0,0.0]
        CartVel = [0.0,0.0]
        PoleAngle = [0.0,0.0]
        PoleTipVel = [0.0,0.0]
        
        running_reward = 10
        k_int = 0
    
        for t in range(10000):
            #View the game/simulation
            if renderBool == True:
                env.render()
            
            # get state information
            CartPos.append(state[0]) # Max +/- 2.4 
            CartVel.append(state[1])
            PoleAngle.append(state[2]*180.0/scipy.pi) # Max +/- 41.8 deg
            PoleTipVel.append(state[3])
            
            if k_int < n_int: 
                k_int += 1 
            
            # Compute weighted PID responce
            action = 0.0
            err = [CartPos, CartVel, PoleAngle, PoleTipVel]
            
            try:
                for h in range(4):
                    action += weights[h]*PIDu(err[h], K[h], k_int)
            #try:
            #    action = weights[0]*PIDu(CartPos, K[0], k_int) + weights[1]*PIDu(CartVel, K[1], k_int) + weights[2]*PIDu(PoleAngle, K[2], k_int) + weights[3]*PIDu(PoleTipVel, K[3], k_int)
            except:
                action = 0.0
            # Process PID signal to 0/1
            action = scipy.sign( action ) 
            action = int (action)
            if action < 0:
                action = 0
    
            # Next pole "time" step
            state, reward, done, _ = env.step(action)

            if done:
                break
        # end for
        reward_array.append(running_reward * 0.00 + t * 1.0)
    

    #end for
    return scipy.mean(reward_array), scipy.std(reward_array)

def PIDGradVector(vector, pertub, batch_size):
    # assumes linear vector
    pertubationParam = pertub*vector
    gradient = []
    grad_std = []
    for i in range(len(vector)):
        if abs(pertubationParam[i]) < 1e-9:
            pertubationParam[i] = pertub
        
        vector_trial = vector
        vector_trial[i] += pertubationParam[i]
        mean1, std1 = TestPIDPoleCart(vector, batch_size)
        
        vector_trial = vector
        vector_trial[i] -= pertubationParam[i]
        mean2, std2 = TestPIDPoleCart(vector, batch_size)
        
        gradient.append( (mean1-mean2)/pertubationParam[i]/2.0 )
        #grad_std.append( scipy.mean([std1, std2]) )
        grad_std.append(0)
    return gradient, grad_std
    
    
def NewtonRaphsonMinimizer(vecInit, pertub, batch_size, maxIter, lr):
    vec = vecInit
    
    if verboseBool:
        print "Start Newton-Raphson error minimizer for PID controller, Learn rate:", lr , ", Iterations: ", maxIter
        print "Pertubing gradient parameter", pertub, "\tBatch size", batch_size
    
    for iteration in range(maxIter):
        
        f, f_std = TestPIDPoleCart(vec, batch_size)
        if f_std < 1e-6:
            if verboseBool:
                print "NO standard deviation -> maxima achived??"
            break
    
        gradf, gradf_std = PIDGradVector(scipy.array(vec), pertub, batch_size/2)
        vec = np.array(vec) + np.dot(lr,gradf) /np.array(f)
        
        
        if verboseBool:
            print "Iteration:",iteration+1 ,"\tAverage reward:", f, (f_std)
            #print "Vector =", vec
            #print "Gradient vector =", gradf
        
        
        
    # endfor    
    return vec    
    
if __name__ == "__main__":

    verboseBool = True
    renderBool= False
    vec = NewtonRaphsonMinimizer(inputVec, 0.05, 100, 10, 0.002)
    renderBool = True
    TestPIDPoleCart(vec, 10)
    

    
    
    

    

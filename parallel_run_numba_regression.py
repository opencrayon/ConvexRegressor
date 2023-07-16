# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 13:42:33 2021

@author: as036
"""


from numba.cuda.api import synchronize
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import time as time
import operator

from numba import cuda, float32, int32, void


#import data sets and load data
#this should be your data location
box = np.load('C:\\Users\\as036\Downloads\\value_fun (4).npy')
data = box[:,:,0,0,0,0,:]

#15*4 domain configuration cannot be changed
n=15
m=4

#broadcasts scalar with vector sum
@cuda.jit(void(float32[:],float32), device = True)
def broadcast_pointvector(vec, pt):
    for ips in range(vec.shape[0]):
        vec[ips] = vec[ips]+pt


#zeros an entire vector
@cuda.jit(void(float32[:]), device = True)
def zerovector(v1):
    for t in range(v1.shape[0]):
        v1[t] = 0

#transfers memory from vector one to vector two
@cuda.jit(void(float32[:],float32[:]), device = True)
def vectrans(d1,d2):
    for t in range(d1.shape[0]):
        d1[t] = d2[t]

#transfers array 1 to array 2, memory transfer
@cuda.jit(void(float32[:,:],float32[:,:]),device = True)
def arraytransfer(d1,d2):
    for t in range(d1.shape[0]):
        for z in range(d2.shape[1]):
            d1[t,z] = d2[t,z]
            
#flattns array into vector
@cuda.jit(void(float32[:],float32[:,:]), device = True)
def wierdtransfer(d1,d2):
    for t in range(d2.shape[0]):
        for z in range(d2.shape[1]):
            d1[z] = d2[t,z]

#multiplies a vector by a scalar
@cuda.jit(void(float32[:],float32, float32[:]), device = True)
def vecscalmult(vec, scale, retvec):
    for iter in range(vec.shape[0]):
        retvec[iter] = vec[iter]*scale
        
#adds a vector of values to some data vector with index
@cuda.jit(void(float32[:], float32[:,],int32[:]), device = True)
def addgrid(vec, data,index):
    for iter in range(vec.shape[0]):
        data[index[iter],] += vec[iter]

#same as above, alternative for data type [:] vs [:,]
@cuda.jit(void(float32[:], float32[:,],int32[:]), device = True)
def movedata(vec, data,index):
    for iter in range(vec.shape[0]):
        data[index[iter],] = vec[iter]
        
#same as addgrid        
@cuda.jit(void(float32[:],float32[:],int32[:]),device = True)
def vecindextransfer(vec,data,index):
    for iter in range(vec.shape[0]):
        data[index[iter]] = vec[iter]
        
#transfers point into data over index grid k
@cuda.jit(void(float32[:],float32,int32[:]),device = True)
def pointtransfer(d1,pt,index):
    for k in range(index.shape[0]):
        d1[index[k]] = pt


#transfers data2 into data1 over respecive indexes index, index1
@cuda.jit(void(float32[:],float32[:], int32[::1], int32[:]), device = True)
def transferdata(data1, data2, index,index1):
    for k in range(data1.shape[0]):
        p1 = index[k]
        p2 = index1[k]
        point = data2[p2]
        data1[p1] = point

#subracts two data vectors at particular indexes, (for loop causes memory leak problem)
@cuda.jit(void(float32[:,],float32[:,],float32[:],int32[:],int32[:]), device = True)
def tripleindexsubtraction(data1, data2, result,index1,index2):
    result[0] = data1[index1[0]]-data2[index2[0]]
    result[1] = data1[index1[1]]-data2[index2[1]]
    result[2] = data1[index1[2]]-data2[index2[2]]


#same as above but for 2 points (indexed)
@cuda.jit(void(float32[:,],float32[:,],float32[:],int32[:],int32[:]), device = True)
def doubleindexsubtraction(data1, data2, result,index1,index2):
    result[0] = data1[index1[0]]-data2[index2[0]]
    result[1] = data1[index1[1]]-data2[index2[1]]
    #
    #
    #for indexer in range(index1.shape[0]-1):
    #   showindex[5] = index1[indexer]
        
        
#same as above but for 6 points indexed     
@cuda.jit(void(float32[:,],float32[:,],float32[:],int32[:],int32[:],float32[:]), device = True)
def sixindexsubtraction(data1, data2, result,index1,index2,showindex):
    result[0] = data1[index1[0]]-data2[index2[0]]
    result[1] = data1[index1[1]]-data2[index2[1]]
    result[2] = data1[index1[2]]-data2[index2[2]]
    result[3] = data1[index1[3]]-data2[index2[3]]
    result[4] = data1[index1[4]]-data2[index2[4]]
    result[5] = data1[index1[5]]-data2[index2[5]]
    
    

            
#sums an array over all indices
@cuda.jit(float32(float32[:,:]),device = True)
def arraysum(array):
    number = 0
    for t in range(array.shape[0]):
        for z in range(array.shape[1]):
            number += abs(array[t,z])

    return number


#subtracts two arrays from each other, elementwis
@cuda.jit(void(float32[:,:],float32[:,:],float32[:,:]), device = True)
def arraysubtract(a1,a2, res):
    for t in range(a1.shape[0]):
        for z in range(a2.shape[1]):
            res[t,z] = a1[t,z] - a2[t,z]
            
#multiplys an array by a scalar
@cuda.jit(void(float32[:,:],float32,float32[:,:]), device = True)
def arrayscalarmult(a1,pt,a2):
    for t in range(a1.shape[0]):
        for z in range(a1.shape[1]):
            a1[t,z] = a2[t,z]*pt

#adds two arrays to each other, elementwise
@cuda.jit(void(float32[:,:],float32[:,:],float32[:,:]), device = True)
def arrayadd(a1,a2,a3):
    for t in range(a1.shape[0]):
        for z in range(a2.shape[1]):
            a3[t,z] = a1[t,z]+a2[t,z]
            
            
#indexer for flattened 4d array, returns single array index
@cuda.jit(int32(int32,int32,int32,int32), device = True)
def arrayindex(i1,i2,i3,i4):
    pop = int(24*i1+6*i2+2*i3+i4)

    return pop



#indexer for flattened 4d array, returns index for [i,j,:,p] (3pts)
@cuda.jit(void(int32,int32,int32,int32,int32[:]), device = True)
def indexmaker2(v1,v2,v3, range_num,index):
    for iter in range(range_num):
        index[iter] = int(24*v1+6*v2 + 2*iter+v3)
        
        
        
#index for flattened 4d array, returns index for [i,j.p,:] (2pts)
@cuda.jit(void(int32,int32,int32,int32,int32[:]), device = True)
def indexmaker1(v1,v2,v3,range_num,index):
    
    for iter in range(range_num):
        
        index[iter] = int(24*v1+6*v2+2*v3+iter)

#index for flattened 4d array, return [i,j,:,:]
@cuda.jit(void(int32,int32,int32[:]), device = True)
def fullindex(p1,p2,index):
    for its in range(6):    
        index[its] = 24*p1 + 6*p2 +its



#multiplies data by vector elementwise linearly
@cuda.jit(void(float32[:,], float32[:,], float32[:]), device = True)
def vectorelementmulti(data1, data2, result):
    for index in range(data1.shape[0]):
        result[index] = data1[index] *data2[index] 

#subtracts two data vectors linearly and accumlates residuals
@cuda.jit(float32(float32[:],float32[:]),device = True)
def accumulateregression(data1, data2):
    accumulator = 0
    for iter in range(data1.shape[0]):
        
        accumulator += data1[iter]-data2[iter]

    return accumulator


#sum of vector elements
@cuda.jit(float32(float32[:]),device = True)
def vectorsum(data):
    number = 0
    for ind in range(data.shape[0]):
        number += data[ind]
    return number

#subtracts 2 vectors at indexes at accumulates residuals
@cuda.jit(float32(float32[:],float32[:],int32[:],int32[:]), device = True)
def indexvectorsubtract(v1,v2,ind1,ind2):
    reg = 0
    for ind in range(ind1.shape[0]):
        reg += v1[ind1[ind]]-v2[ind2[ind]]

    return reg

      
#function that computes penalties for regression points. 
#This function returns an array of values that correspond to a small sample
#residual gain or loss on all planes 

#used to add penalty values on regression to avoid pathological results 
@cuda.jit(void(float32[:],float32[:],float32[:,:],int32[:]), device = True)
def penaltymaker(dataup,uptris,penaltyarray,index):
    index[0:-1] = int32(0)
    
    for i in range(penaltyarray.shape[0]):
        for j in range(penaltyarray.shape[1]):
            p1 = int32(penaltyarray.shape[0]-(i+1))
            

            p2 = int32(penaltyarray.shape[1]-(j+1))
            
            indexmaker2(p1,p2,0,3,index)
           

            p1 = indexvectorsubtract(uptris,dataup,index,index)
            
            penaltyarray[penaltyarray.shape[0]-(i+1),penaltyarray.shape[1]-(j+1)] = p1
            
            
#is the same as np.sum(array[i:,j:])
#used in penaltymaker to compute total residual error proceeding some plane
@cuda.jit(float32(float32[:,:],int32,int32),device = True)
def addarraysection(array,ind1,ind2):
    num =0
   
    for pick in range(16-ind1):
        for rick in range(4-ind2):
            num += array[15-pick,3-rick]
            

        
    return num
    
            
#sums an entire array
@cuda.jit(float32(float32[:,:]),device = True)
def arraysum(array):
    number = 0
    for t in range(array.shape[0]):
        for z in range(array.shape[1]):
            number += abs(array[t,z])

    return number

#sets uptris to be 0 in desired places
@cuda.jit(void(float32[:]),device = True)
def uptriszero(data):
    for i in range(16):
        for j in range(5):
            data[24*i+6*j+2+1] = 0
            data[24*i+6*j+4+1] = 0
    
#sets downtris to be 0 in desired places
@cuda.jit(void(float32[:]),device = True)
def downtriszero(data):
    for pick in range(16):
        for em in range(5):
            data[24*pick+6*em+0] = 0
            data[24*pick+6*em+2] = 0
            
#zeros an entire array
@cuda.jit(void(float32[:,:]),device = True)
def zeroarray(array):
    for pick in range(array.shape[0]):
        for tick in range(array.shape[1]):
            array[pick,tick] = 0
            

            
#interpolates alterate parameters from current sets (uptris and downtris)
@cuda.jit(void(float32[:],float32[:],float32[:,:]),device = True)
def get_full_parameters(uptris, downtris, alt_params):
    
    
    for iters in range(15):
        for piters in range(4):
            
            
            
            idx = arrayindex(iters,piters,0,1)
            pdx = arrayindex(iters,piters,0,0)
            
            
            alt_params[iters,piters] = (uptris[idx] - uptris[pdx])/float32(1.25)
            
            
    for piters in range(4):
        idx = arrayindex(14,piters,2,1)
        pdx = arrayindex(14,piters,2,0)
        #over 1.25 is constant grid value of y. Can be altered with some grid valus
        alt_params[15,piters] = (downtris[idx]- downtris[pdx])/1.25
        
        

#gradient computation of alternate parameters 
#simple functional gradient, no penalties used as this is used to enforce concavity          
@cuda.jit(void(float32[:],float32[:],float32[:],float32[:],float32[:,:]),device = True)
def full_param_grads(uptris,true_up,downtris,true_down,grad_list):
    
    for iters in range(15):
        for piters in range(4):
            idx = arrayindex(iters,piters,0,1)
            grad_list[iters,piters] = (true_up[idx]-uptris[idx])

    for piters in range(4):
        idx = arrayindex(14,piters,2,1)
        grad_list[15,piters] = true_down[idx]- downtris[idx]
        
#barrier function for alternate parameters
#alter line 353 and magnitude of values to tune to data.
@cuda.jit(void(float32[:,:],float32[:,:]),device = True)
def full_barriers(alt_params,barrs):
    for iters in range(alt_params.shape[0]):
        for piters in range(alt_params.shape[1]-1):
            if (alt_params[iters,piters] > alt_params[iters,piters+1]+.7):
                barrs[iters,piters] += 0
            elif (alt_params[iters,piters+1] > alt_params[iters,piters]):
                barrs[iters,piters+1] += 10
                barrs[iters,piters] += -8
            else:
                barrs[iters,piters + 1] += min((abs(1 / (alt_params[iters,piters] - alt_params[iters,piters+1]) ** .2), 10))
                barrs[iters,piters] += -min((abs(1 / (alt_params[iters,piters] - alt_params[iters,piters+1]) ** .2), 8))


#interpolates slope from 2 points, used for the laternateve charecterization of the piecewise
#function
@cuda.jit(float32(float32,float32,float32),device = True)
def interpolate(x1,x2,grid):
    pt = (x2-x1)/grid
    return pt

#creates grid of values from slopes of domains given by x_params,y_params, and grids
@cuda.jit(void(float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32,float32[:],int32[:], int32[:],float32[:],float32[:],float32[:],float32[:]),device = True)
def alt_piecewise(x_params,y_params,xgrid,ygrid,initpt,general,index1,index2,memory,memory2,uptris,downtris):
    
    idx = arrayindex(0,0,0,0)
    #creates grid of points in general vector, same dimension as "upris" and "downtris"
    general[idx] = initpt
    
    for iters in range(15):
        if iters == 0:
            
            
            indexmaker2(iters,0,0,3,index1)
            
            pt = arrayindex(iters,0,0,0)
            pt3 = arrayindex(iters,0,2,0)
            vecscalmult(xgrid[:,iters],x_params[iters,0],memory)
            
            broadcast_pointvector(memory,general[pt])
            #broadcasts initial point into domain and adds proper values given by
            #grid and parameter size
            
            
            vecindextransfer(memory, general,index1)
            
            pt1 = arrayindex(iters,0,0,1)
            
            general[pt1] = general[pt]+ygrid[0,-1]*y_params[iters,0]
            
            
            
            pt2 = arrayindex(iters,0,2,1)
            pt3 = arrayindex(iters,0,2,0)
            
            general[pt2] = general[pt3]+ygrid[0,-1]*y_params[iters+1,0]
            
            x_interpolant = interpolate(general[pt1],general[pt2],xgrid[-1,iters])
            
            
            
            pt4 = arrayindex(iters,0,1,1)
            
            general[pt4] = general[pt1]+xgrid[1,iters]*x_interpolant
            
            
        if iters > 0:
            
            
            
            #works through same process as above via broadcast and add grid*params
            #for [:,0] domains
            indexmaker2(iters,0,0,3,index1)
            
            pt = arrayindex(iters-1,0,2,0)
            
            vecscalmult(xgrid[:,iters],x_params[iters,0],memory)
            broadcast_pointvector(memory,general[pt])
            
            
            vecindextransfer(memory,general,index1)
            
            pt1 = arrayindex(iters,0,0,1)
            pt2 = arrayindex(iters,0,0,0)
            
            general[pt1] = general[pt2] +ygrid[0,-1]*y_params[iters,0]
           
            
            pt3 = arrayindex(iters,0,2,1)
            pt4 = arrayindex(iters,0,2,0)
            
            general[pt3] = general[pt4] + ygrid[0,-1]*y_params[iters+1,0]
            x_interpolant = interpolate(general[pt1],general[pt3],xgrid[-1,iters])
            
            pt5 = arrayindex(iters,0,1,1)
            general[pt5] = general[pt1]+xgrid[1,iters]*x_interpolant
            
            
    for index in range(15):
        for pindex in range(3):
            
            #broadcast and add method for entire domain
            pt = arrayindex(index,pindex+1,0,0)
            pt1 = arrayindex(index,pindex,0,1)
            general[pt]  = general[pt1]
            
            
            
            
            pt = arrayindex(index,pindex+1,2,0)
            pt1 = arrayindex(index,pindex,2,1)
            
            general[pt] = general[pt1]
            pt = arrayindex(index,pindex+1,0,1)
            pt1 = arrayindex(index,pindex+1,0,0)
            general[pt] = general[pt1] +ygrid[0,-1]*y_params[index,pindex+1]
            
            pt = arrayindex(index,pindex+1,2,1)
            pt1 = arrayindex(index,pindex+1,2,0)
            
                

            general[pt] = general[pt1] +ygrid[0,-1] * y_params[index+1,pindex+1]
            
            pt = arrayindex(index,pindex+1,0,0)
            pt1 = arrayindex(index,pindex+1,2,0)
            x_interpolant = interpolate(general[pt],general[pt1],xgrid[-1,index])
            
            pt = arrayindex(index,pindex+1,0,0)
            pt1 = arrayindex(index,pindex+1,1,0)
            general[pt1] = general[pt]+xgrid[1,pindex]*x_interpolant
            
            x_params[index,pindex+1]  = x_interpolant
            pt = arrayindex(index,pindex+1,0,1)
            pt1 = arrayindex(index,pindex+1,2,1)
            x_interpolant = interpolate(general[pt],general[pt1],xgrid[-1,index])
            
            pt = arrayindex(index,pindex+1,1,1)
            pt1 = arrayindex(index,pindex+1,0,1)
            general[pt] = general[pt1] + xgrid[1,index]*x_interpolant
            
            
        
    vectrans(uptris,general)
        
    vectrans(downtris,general)
    

        

        
            
        






#standard piecewis function. We must use this in first and last line of regression code
#to pull uptris and parameters



#Function does the same thing as alt_piecewise, but works with parameters facing
#the power grid direction. I.E this provides alternate charecterization using alternate
#parameters
@cuda.jit(void(float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32, float32[:],float32[:], float32[:],int32[::1],int32[::1]), device = True)
def piecewisefunction(x_params, y_params, xgrid,ygrid,initpt, uptris, downtris, multivec,index,index2):
    #indexmaker 2 creates an index of 3 points to transfer 4d array to 1d array
    # that is (i,j,:,k ) -> flat index
    zerovector(downtris)
    zerovector(uptris)
    

    indexmaker2(0,0,0,3,index)

    #point transfer for initial point to origonal data
    pointtransfer(uptris,initpt,index)
    
    
    
    
    
    
    
    for tick in range(4):
        
        upindex = arrayindex(0,tick,0,1) 
        upindexeq = arrayindex(0,tick,0,0)
        final = arrayindex(0,tick+1,0,0)
        
        uptris[upindex] = uptris[upindexeq]+ygrid[tick,-1]*y_params[tick,0]
        downtris[upindex] = uptris[upindex]
        
        
        
        if tick < 3:
            final = arrayindex(0,tick+1,0,0)
            uptris[final] = uptris[upindex]
            
        


    for i in range(15):
        indexmaker2(i,0,0,3,index)
        vecscalmult(xgrid[:, i],x_params[i,0], multivec)
        
        
        addgrid(multivec, uptris, index)
       
        


        
        
        for j in range(4):
            
            indexmaker2(i,j,1,3,index)
            if i == 0:
                if j == 0:
                    i0 = index[0]
                    i1 = index[1]
                    i2 = index[2]
            
            downindex = arrayindex(i,j,0,1)
            pointtransfer(downtris,downtris[downindex],index)


            

            vecscalmult(xgrid[:,i],x_params[i,j+1],multivec)
            

           
            addgrid(multivec, downtris,index)
            
            
            downindex = arrayindex(i,j,2,0)
            downtris[downindex] = uptris[downindex]
            
            
            
            if j < 3:
                
                indexmaker2(i,j+1,0,3,index)
                
                indexmaker2(i,j,1,3,index2)
                
                for q in range(index.shape[0]):
                    p1 = index[q]
                    p2 = index2[q]
                    point = downtris[p2]
                    uptris[p1] = point
                    
                
                
                
                
            


            if i < 14:
                
                indexmaker2(i+1,j,0,3,index)
                
                pt = arrayindex(i,j,2,0)

                pointtransfer(uptris, uptris[pt],index)
                pt2 = arrayindex(i+1,j,0,1)
                pt = arrayindex(i,j,2,1)
                uptris[pt2] = downtris[pt]
                indexmaker2(i+1,j,1,3,index)
                
                pointtransfer(downtris,downtris[pt],index) 
                
                
                
                
    
     
    
    #spare[3] = uptris[0]
    
    #uptriszero(uptris)
    downtriszero(downtris)
    uptris[3] = 0

    





#This is a complete chain rule calculator used for numba

@cuda.jit(void(float32[:],float32[:],float32[:],float32[:],float32[:,:],float32[:,:], float32[:,:], float32[:], float32[:],float32[:,:],int32[:],int32[:],int32[:]), device = True)
def derivatives(uptris,downtris, data_up,data_down,xgrid, gradvals_long, gradvals_y,multivec, arr_subtract,penalties,index,index_long,smolindex):
    
    #First compute pnealties in one operation to reduce comp time
    penaltymaker(uptris,data_up,penalties,index_long)
    
    
    
    #Now do full run over computing residuls of true data up and uptris
    #use penaltymaker array to add penalties at every step
    #chain rule computation is given in write up
    for i in range(15):
        for j in range(5):

            if j == 0:
                if i < 14:
                    indexmaker2(i,j,0,3,index)
                    
                        
                    zerovector(arr_subtract)
                    zerovector(multivec)
                    tripleindexsubtraction(data_up,uptris,arr_subtract,index,index)
                    
                    
                    
                   
                    vectorelementmulti(arr_subtract, xgrid[:,i],multivec)
                    
                    
                    
                        
                    
                   #.032 seems to be the magic number after testing. Flavor according to taste...
                    gradvals_long[i,j] = vectorsum(multivec)+.032*addarraysection(penalties,i+1,j)
                    
                    
                       
                else:
                    indexmaker2(i,j,0,3,index)
                    zerovector(arr_subtract)
                    
                    

                        
                    zerovector(multivec)
                    tripleindexsubtraction(data_up,uptris,arr_subtract,index,index)
                    
                        
                    vectorelementmulti(arr_subtract,xgrid[:,i],multivec)

                    
                    gradvals_long[i, j] = vectorsum(multivec)


            else:
                
                if i < 13:
                    
                    zerovector(arr_subtract)
                    zerovector(multivec)
                    pt = arrayindex(i,j-1,2,0)
                    
                    indexmaker2(i,j-1,1,3,index)
                    tripleindexsubtraction(data_down,downtris,arr_subtract,index,index)
                    vectorelementmulti(arr_subtract,xgrid[:,i],multivec)
                    
                    
                    
                    gradvals_long[i, j] = vectorsum(multivec)+.032*addarraysection(penalties,i+1,j-1)
                  
                else:
                    
                    zerovector(arr_subtract)
                    zerovector(multivec)
                    
                    indexmaker2(i,j-1,1,3,index)
                    tripleindexsubtraction(data_down,downtris,arr_subtract,index,index)
                    
                    vectorelementmulti(arr_subtract,xgrid[:,i],multivec)
                    gradvals_long[i,j] = vectorsum(multivec)
                    
                    
    

                   




    for j in range(1):
        for i in range(4):
            zerovector(arr_subtract)
            
            indexmaker1(j,i,0,2,smolindex)
            

            doubleindexsubtraction(data_up,uptris,arr_subtract,smolindex,smolindex)

            gradvals_y[j,i] = vectorsum(arr_subtract)
  
    
            


    

#sets up barrier function over the standard parameters corresponding to
#the standard piecewise function
@cuda.jit(void(float32[:,:], float32[:,:]),device = True)
def new_barrier(params, barriers):
    
    p = params.shape[0]
    g = params.shape[1]
    for i in range(p-1):
        for j in range(g):
            value = params[i+1,j]

            if params[i,j] > value+1:
                barriers[i,j] += 0
            elif params[i,j] < value:

                barriers[i,j] += -5
                barriers[i+1,j] += 6




            else:

                barriers[i,j] += - min((abs(.1/ (params[i,j]-value+1e-10)),5))
                barriers[i+1, j] += min((abs(.09 / (params[i, j] - value + 1e-10)), 6))


    for i in range(p):
        for j in range(g-1):
            value = params[i, j+1]

            if params[i, j] > value + 1:
                barriers[i, j] += 0
            elif params[i, j] < value:

                barriers[i, j] += -5
                barriers[i, j+1] += 6




            else:

                barriers[i, j] += - min((abs(.1 / (params[i, j] - value + 1e-10)), 10))
                barriers[i, j+1] += min((abs(.09 / (params[i, j] - value + 1e-10)), 9)) 
                
                
                #returns barrier array
                
        

  #barrier function used used for y_parameters for standard piecewise function  
@cuda.jit(void(float32[:,:],float32[:,:]),device = True)
def barrier(ystuff, barriers):
    
    params = ystuff

    

    for iterational in range(params.shape[0] - 1):
        

        if (params[0,iterational] > params[0,iterational + 1] + 1):
            
            barriers[0,iterational + 1] = 0
        elif (params[0,iterational + 1] > params[0,iterational]):
            
            barriers[0,iterational + 1] = 7
        else:
            
            

            barriers[0,iterational + 1] = min((abs(.1 / (params[0,iterational] - params[0,iterational+1])), 7))
            
    #returns vector of barriers so everything can be vectorized
    
    
#updates eta based on the current and previous gradients
@cuda.jit(float32(float32,float32[:,:],float32 ), device = True)
def update_eta(eta,grads,grad_prev):
    

    if arraysum(grads)> grad_prev:
        eta = .985*eta
    else:
        eta = eta

    return eta


#checks the concavity of the x facing parameters (associated with piecewise function)
@cuda.jit(int32(float32[:,:]),device = True)
def concheck(params):
    problems = int32(0)
    for i in range(params.shape[0]-1):
        for j in range(params.shape[1]-1):
            if params[i,j] < params[i+1,j]:

                problems += 1
            elif params[i,j] < params[i,j+1]:

                problems += 1
                
    for j in range(params.shape[1]-1):
        if params[14,j] < params[14,j+1]:
            problems += 1
            
        

    return problems


# checks the concavity of the b facing parameters
@cuda.jit(int32(float32[:,:]),device = True)
def altconcheck(params):
    concheck = 0
    for look1 in range(16):
        for look2 in range(3):
            if params[look1,look2+1]> params[look1,look2]:
                concheck += 1
                
    return concheck





#THIS IS THE MAIN DRIVER
#this is run identically to algorithm 1 in the write-up and 
#v3 global update on github


#returns x_params and y_params, in array run by kernel function
#use piecewise maker from v3globalupdate to get data and graphs
@cuda.jit(void(float32[:,],float32[:,],float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32),device = True)
def runner(true_up,true_down, x_params, y_params,xlist,ylist,iniotpt):
    
    #setting up internal memory, will not be accesed after threading
    initpt = true_up[0]
    arr_subtract = cuda.local.array(3, dtype = float32)
    barrier2darray = cuda.local.array((15,5), dtype = float32)
    barrier1darray = cuda.local.array((4,1), dtype = float32)

    
    index1 = cuda.local.array(3, dtype = int32)
    index2 = cuda.local.array(3,dtype = int32)
    smolindex = cuda.local.array(2, dtype = int32)
    indexlong  =cuda.local.array(6, dtype = int32)
    gradient_penalities = cuda.local.array((15,5), dtype = float32)
    eta= .02
    eta1 = .02
    grad_prev = 10e15
    etay = .1
    memory = cuda.local.array(3,dtype=float32)
    memory1 = cuda.local.array(3,dtype = float32)
    uptris = cuda.local.array(15*4*3*2,dtype = float32)
    downtris =cuda.local.array(15*4*3*2,dtype = float32)
    general = cuda.local.array(15*4*3*2,dtype = float32)
    funky_params = cuda.local.array((16,4),dtype = float32)
    full_param_gradients = cuda.local.array((16,4),dtype = float32)
    big_barriers = cuda.local.array((16,4),dtype = float32)
    multivec = cuda.local.array(3, dtype = float32)
    gradvals_long = cuda.local.array((15,5), dtype = float32)
    pens = cuda.local.array((15,4),dtype = float32)
    gradvals_y = cuda.local.array((1,4), dtype = float32)
    
    #gets an initial set of uptris and downtris
    
    piecewisefunction(x_params,y_params,xlist,ylist,initpt,uptris,downtris,multivec,index1,index2)
    
    #main running function 
    for pablano in range(1000):
        
        #Does one run of fetching alternate parameters from uptris and downtris
        get_full_parameters(uptris,downtris,funky_params)
        
        #computes gradients
        full_param_grads(uptris,true_up,downtris,true_down,full_param_gradients)
        
        #computes barriers
        full_barriers(funky_params,big_barriers)
        
        #the following completes the regression
        arrayscalarmult(full_param_gradients,eta1,full_param_gradients)
        
        arrayscalarmult(big_barriers,eta1,big_barriers)
        
       
        
        arrayadd(funky_params,full_param_gradients,funky_params)
        
        arraysubtract(funky_params,big_barriers,funky_params)
        for ancho in range(4):
           y_params[ancho,0] = funky_params[0,ancho]
           
          #computes a new set of uptris and downtris from these, as well as x_params
        alt_piecewise(x_params,funky_params,xlist,ylist,initpt,general,index1,index2,memory,memory1,uptris,downtris)
        
        #slowly reduces eta1, in case of divergence
        if pablano%5 == 0:
            eta1 = eta1*0.995
            
        #convexity check and zeroing arrays
        zeroarray(big_barriers)
        ct = altconcheck(funky_params)
        
        
        #runs standard regression of x facing parameters 7 times
        for chipotle in range(7):   
            
            
            piecewisefunction(x_params,y_params,xlist,ylist,initpt,uptris,downtris,multivec,index1,index2)
        
            
        #gradient solver
            derivatives(uptris,downtris,true_up,true_down,xlist,gradvals_long,gradvals_y,multivec, arr_subtract,pens,index1,indexlong,smolindex)
            
            # more complicated gradient updates depending on current gradients
            if chipotle%5 == 0:
                
                eta = max(eta*0.9945,0.0005)
            
            etay = .01
        
            #computes new barriers
            new_barrier(x_params,barrier2darray)
            
            #full regression
            arrayscalarmult(barrier2darray,1.5,barrier2darray)
            arraytransfer(gradient_penalities,gradvals_long)
            arraysubtract(gradvals_long,barrier2darray,gradient_penalities)
            arrayscalarmult(gradvals_long,eta,gradvals_long)
            arrayadd(x_params,gradvals_long,x_params)
            
            arrayscalarmult(barrier2darray,eta,barrier2darray)
            arraysubtract(x_params,barrier2darray, x_params)
      
        arrayscalarmult(gradvals_y,0,gradvals_y)
        
        arrayadd(y_params,gradvals_y.T,y_params)
        
        grad_prev = arraysum(gradient_penalities)
        
        zeroarray(gradient_penalities)
        zeroarray(barrier1darray)
        zeroarray(barrier2darray)
        zeroarray(gradvals_long)
        zeroarray(pens)
        #breaks once concavity is achieved
        if pablano > 100:
            if ct + concheck(x_params) == 0:
                break
        
        
    piecewisefunction(x_params,y_params,xlist,ylist,initpt,uptris,downtris,multivec,index1,index2)

    

#kernel function that runs runner from above based on data sets defined by user. Thread properly
@cuda.jit(void(float32[:,:],float32[:,:],float32[:,:,:],float32[:,:,:],float32[:,:],float32[:,:],float32,int32[:]))
def kernel(true_up,true_down, x_params, y_params,xlist,ylist,initpt,ping):
    
    pos = cuda.grid(1)
    if pos < 256:
        runner(true_up[pos,:],true_down[pos,:], x_params[:,:,pos], y_params[:,:,pos],xlist,ylist,initpt)



#makes data lists from data set given (of 7 dimensions)
def make_data_lists(data):
    
    number = len(data[0,0,0,0,0,0,:])*len(data[0,0,0,0,0,:,0])*len(data[0,0,0,0,:,0,0])*len(data[0,0,0,:,0,0,0])*len(data[0,0,:,0,0,0,0])
    lists = np.zeros((31,5,200,int(np.ceil(number/200))))
    j = 0
    counter = 0
    for i in range(len(data[0,0,0,0,0,0,:])):
        for k in range(len(data[0,0,0,0,0,:,i])):
            for t in range(len(data[0,0,0,0,:,k,i])):
                for r in range(len(data[0,0,0,:,t,k,i])):
                    for q in range(len(data[0,0,:,r,t,k,i])-1):
                        j += 1
                        lists[:,:,j,counter] = data[:,:,q,r,t,k,i]
                        if j == 199:
                            j = 0
                            counter += 1
    return lists
def reindex(index,d1,d2,d3,d4,d5):
    newindex = []
    for indexes in index:
        ind1 = int(np.floor(indexes / (d1*d2*d3*d4)))
        indexes = indexes - ind1*(d1*d2*d3*d4)
        
        ind2 = int(np.floor(indexes / (d1*d2*d3)))
        indexes = indexes - ind2*(d1*d2*d3)
        
        ind3 = int(np.floor(indexes / (d1*d2)))
        indexes = indexes - ind3*(d1*d2)
        
        ind4 = int(np.floor(indexes / (d1)))

        indexes = indexes - ind4*d1
        
        ind5 = indexes
        
        newindex.append([ind1,ind2,ind3,ind4,ind5])
    return newindex

        
        
        
#sets up grids for the data, and gives corresponding up and down sets
def organizedata(data):
    x_grid = np.array(0 + (15 - 0) * np.linspace(0, 1, 31) ** 2, dtype='float32')

    y_grid = np.array(np.linspace(0,5,5), dtype = 'float32')

    xlist = np.array(np.zeros((3,15)), dtype = 'float32')
    ylist = np.array(np.zeros((5,2)), dtype = 'float32')
    for i in range(15):

        xlist[:,i] = x_grid[2*(i):2*(i+1)+1]- x_grid[2*(i)]





    for i in range(4):
        ylist[i,:] = y_grid[i:(i+2)]-y_grid[i]




    functional_val = np.zeros((15,4,3,2))
    for i in range(15):
        for j in range(4):
            
            functional_val[i,j,:,:] = data[2*i:2*(i+1)+1,j:j+2]
    true_up = np.array(functional_val)
    true_down = np.array(functional_val)

    true_up[:,:,1:3,1] = 0

    return true_up,true_down, xlist, ylist


#used for initial parameters, we need to make batches of parameters for each thread
def arrayrepeat(size, array):
    new_array = np.zeros((array.shape[0],array.shape[1], size))
    for i in range(size):
        new_array[:,:,i] = array

        return new_array
    
def interpol(true_up,true_down,xgrid):
    x_params = np.zeros((15,5))

    for i in range(x_params.shape[0]):
        for j in range(x_params.shape[1]-1):
            x_params[i,j] = (true_up[i,j,2,0] - true_up[i,j,0,0])/xgrid[2,i]

    for i in range(x_params.shape[0]):
        x_params[i,4] = (true_down[i,3,2,1]- true_down[i,3,0,1])/xgrid[2,i]

    return x_params    

def get_full_params(uptris,downtris):
    y_params_list = np.zeros((16,4))
    for i in range(uptris.shape[0]):

        for j in range(4):
            y_params_list[i,j] = (uptris[i,j,0,1] - uptris[i,j,0,0])/1.25


    for j in range(4):
        y_params_list[15,j] = (downtris[14,j,2,1]-downtris[14,j,2,0])/1.25

    return y_params_list



def dconcheck(params):
    problems = 0
    for i in range(len(params[:,0])-1):
        for j in range(len(params[0,:])-1):
            if params[i,j] < params[i+1,j]:

                problems += 1
            elif params[i,j] < params[i,j+1]:

                problems += 1
    for j in range(len(params[0,:])-1):
        if params[-1,j] < params[-1,j+1]:
            problems += 1
    return problems

def alt_concheckwe(params):
    concheck = 0
    for i in range(16):
        for q in range(3):
            if params[i,q+1] > params[i,q]:
                concheck += 1

    return concheck
def masterfunction(box):
    n=15
    m=4
    data1 = box.reshape(31,5,box.shape[2]*box.shape[3]*box.shape[4]*box.shape[5]*box.shape[6])
    data2 = np.zeros(data1.shape)
    x_param_lists = np.zeros((15,5,data1.shape[2]))
    full_param_lists = np.zeros((4,1,data1.shape[2]))
    pickem = 0
    
    x_storage = np.zeros((15,5,box.shape[2],box.shape[3],box.shape[4],box.shape[5],box.shape[6]))
    y_storage = np.zeros((4,1,box.shape[2],box.shape[3],box.shape[4],box.shape[5],box.shape[6]))
    losindexes = []
    posindexes = []
    tick = reindex([357],box.shape[6],box.shape[5],box.shape[4],box.shape[3],box.shape[2])
    
    for i in range(data1.shape[2]):
        if i % 1000 == 0:
            print(i)
        true_up_sets = np.zeros((1,15,4,3,2))
        true_down_sets = np.zeros((1,15,4,3,2))
        
        true_up_sets[0,:,:], true_down_sets[0,:,:],xlist,ylist = organizedata(data1[:,:,i])
                    
                
                
                
                    
        
        true_up_flattened = np.array(np.zeros((15*4*3*2)),dtype = 'float32')
        true_down_flattened = np.array(np.zeros((15*4*3*2)),dtype = 'float32')
        
        true_up_flattened[:,] = true_up_sets[0,:,:,:,:].flatten()
                    
        true_down_flattened[:,] = true_down_sets[0,:,:,:,:].flatten()
            
        x_params = interpol(true_up_sets[0,:,:,:,:],true_down_sets[0,:,:,:,:],xlist)
        
        x_param_lists[:,:,i] = x_params
        yper = get_full_params(true_up_sets[0,:,:,:,:],true_down_sets[0,:,:,:,:])
        full_param_lists[:,:,i] = yper[0,:].T.reshape(4,1)
        if alt_concheckwe(yper)+dconcheck(x_params) != 0:
            data2[:,:,pickem] = data1[:,:,i]
            pickem += 1
            losindexes.append(i)
        else: 
            idx = reindex([i],box.shape[6],box.shape[5],box.shape[4],box.shape[3],box.shape[2])
            x_storage[:,:,idx[0][0],idx[0][1],idx[0][2],idx[0][3],idx[0][4]] = x_param_lists[:,:,i]
            y_storage[:,:,idx[0][0],idx[0][1],idx[0][2],idx[0][3],idx[0][4]] = full_param_lists[:,:,i]
     
    data2 = data2[:,:,0:pickem]
    
    tic = time.perf_counter()
    for itt in range(int(np.ceil(data2.shape[2]/(256)))):
        
                print(np.floor(256/37))
                
                data = data2[:,:,itt*(256):(itt+1)*256]
                print(data.shape)
                    
                true_up_sets = np.zeros((256,15,4,3,2))
                true_down_sets = np.zeros((256,15,4,3,2))
                for i in range(data.shape[2]):
                    true_up_sets[i,:,:], true_down_sets[i,:,:],xlist,ylist = organizedata(data[:,:,i])
                    
                
                
                
                    
                
                true_up_flattened = np.array(np.zeros((256,15*4*3*2)),dtype = 'float32')
                true_down_flattened = np.array(np.zeros((256,15*4*3*2)),dtype = 'float32')
                for i in range(true_up_sets.shape[0]):
                    true_up_flattened[i,:,] = true_up_sets[i,:,:,:,:].flatten()
                    
                    true_down_flattened[i,:,] = true_down_sets[i,:,:,:,:].flatten()
                    
                
                def arrayrepeat(size, array):
                    new_array = np.zeros((array.shape[0],array.shape[1], size))
                    for i in range(256):
                        new_array[:,:,i] = array
                        
                    return new_array
                
                
                x_params = np.array(np.flip(np.linspace(0, 40, n)),dtype = 'float32')
                
                y_params = np.array(np.flip(np.linspace(0, 10, m)), dtype = 'float32')
                
                
                x_params = np.ascontiguousarray(np.repeat(x_params.reshape(n, 1), 5,axis=1))
                y_params = np.ascontiguousarray(np.repeat(y_params.reshape(m, 1), 1, axis=1))
                
                y_params_list = np.array(arrayrepeat(256,y_params) ,dtype = 'float32')
                print(y_params_list.shape)
                x_params_list = np.array(arrayrepeat(256,x_params), dtype = 'float32')
                print(x_params_list.shape)
                for i in range(256):
                    x_params_list[:,:,i] = interpol(true_up_sets[i,:,:,:,:],true_down_sets[i,:,:,:,:],xlist)
                    yper = get_full_params(true_up_sets[i,:,:,:,:],true_down_sets[i,:,:,:,:])
                    y_params_list[:,:,i] = yper[0,:].T.reshape(4,1)
                ''' 
                x_params_list = x_param_lists[:,:,i*6*37:(i+1)*6*37].copy()
               
                y_params_list = full_param_lists[:,:,i*6*37:(i+1)*6*37].copy()
                '''
                dx_params_list = cuda.to_device(x_params_list)
                dy_params_list = cuda.to_device(y_params_list)
                dtrue_up_sets = cuda.to_device(true_up_flattened)
                dtrue_down_sets = cuda.to_device(true_down_flattened)
                
                initpts = np.zeros((256))
                
                initpts[:] = true_up_flattened[:,0]
                
                dinitpts_set = cuda.to_device(initpts)
                
                box = np.load('C:\\Users\\as036\Downloads\\value_fun (4).npy')
                data = box[:,:,0,0,0,0,0]
                
                
                
                
                n = 15
                
                m = 4
                
                initpt = data[0,0]
                
                
                
                x_grid = np.array(0 + (15 - 0) * np.linspace(0, 1, 31) ** 2, dtype='float32')
                
                y_grid = np.array(np.linspace(0,5,5), dtype = 'float32')
                
                    
                
                
                
                
                    
                    
                
                twitch = np.linspace(0,1,5)*5
                twitch = np.array(np.repeat(twitch.reshape(1,5),15,axis = 0), dtype = 'float32')
                
                xlist = np.array(np.zeros((3,15)), dtype = 'float32')
                ylist = np.array(np.zeros((5,2)), dtype = 'float32')
                for i in range(15):
                
                    xlist[:,i] = x_grid[2*(i):2*(i+1)+1]- x_grid[2*(i)]
                
                
                
                
                
                for i in range(4):
                    ylist[i,:] = y_grid[i:(i+2)]-y_grid[i]
                
                
                
                
                functional_val = np.zeros((15,4,3,2))
                for i in range(15):
                    for j in range(4):
                        functional_val[i,j,:,:] = data[2*i:2*(i+1)+1,j:j+2]
                true_up = np.array(functional_val)
                true_down = np.array(functional_val)
                
                true_up[:,:,1:3,1] = 0
                
                
                true_down[:,:,0:2,0] = 0
                true_up = true_up.flatten()
                
                
                
                
                true_down = true_down.flatten()
                
                xlist = np.array(xlist, dtype = 'float32')
                ylist = np.array(ylist, dtype = 'float32')
                true_up = np.array(true_up, dtype = 'float32')
                true_down = np.array(true_down, dtype = 'float32')
                
                d_xlist = cuda.to_device(xlist)
                d_ylist = cuda.to_device(ylist)
                
                d_true_up = cuda.to_device(true_up)
                d_true_down = cuda.to_device(true_down)
                
                ping = np.array(np.zeros(1),dtype = 'int32')
                dping = cuda.to_device(ping)
                
                
                
                print('################################')
                print('SMOOTHING DATA...This may take awhile')
                print('##################################')
                kernel[1,256](dtrue_up_sets,dtrue_down_sets,dx_params_list,dy_params_list,d_xlist,d_ylist,initpt,dping)
                initpts = dinitpts_set.copy_to_host()
                
                x_params_list = dx_params_list.copy_to_host()
                for itch in range(x_params_list.shape[2]):
                    if dconcheck(x_params_list[:,:,itch]) > 0:
                        print(itch)
                
                y_params_list = dy_params_list.copy_to_host()
                altlist = reindex(losindexes[itt*256:(itt+1)*256],box.shape[6],box.shape[5], box.shape[4], box.shape[3], box.shape[2])
                
    
                iterable = 0
                for indexes in altlist:
                    x_storage[:,:,indexes[0], indexes[1], indexes[2],indexes[3], indexes[4]] = x_params_list[:,:,iterable]
                    y_storage[:,:,indexes[0], indexes[1], indexes[2],indexes[3], indexes[4]] = y_params_list[:,:,iterable]
                    iterable += 1
                ping = dping.copy_to_host()
                
                params = x_params_list[:,:,0]
                y_params = y_params_list[:,:,0]
                
                
    
    #graphing tool, pulls uptris and downtris from returned parameters
    toc = time.perf_counter()
    print(toc-tic)
def piecewisefunction(x_params, y_params, xgrid,ygrid,initpt):
    uptris = np.zeros((15,4,3,2))
    downtris = np.zeros((15,4,3,2))

    uptris[0,0,:,0] = initpt
    for i in range(4):
        uptris[0,i,0,1] = uptris[0,i,0,0]+ygrid[i,-1]*y_params[i,0]
        downtris[0,i,0,1] = uptris[0,i,0,1]
        if i < 3:

            uptris[0,i+1,0,0] = uptris[0,i,0,1]





    for i in range(15):
        uptris[i, 0, :, 0] += xgrid[:, i] * x_params[i,0]



       

        for j in range(4):


            downtris[i,j,:,1] = downtris[i,j,0,1]



            downtris[i,j,:,1] += xgrid[:,i]*x_params[i,j+1]


            downtris[i, j, -1, 0] = uptris[i, j, -1, 0]

            if j < 3:

                uptris[i,j+1,:,0] = downtris[i,j,:,1]



            if i < 14:

                uptris[i+1,j,:,0] = uptris[i,j,2,0]
                uptris[i + 1, j, 0, -1] = downtris[i, j, -1, -1]
                downtris[i+1,j,:,1] = downtris[i,j,-1,-1]




    uptris[:,:,1:3,1] = 0
    downtris[:,:,0:2,0] = 0




    return uptris, downtris
masterfunction(box)
uptris,downtris = piecewisefunction(params,y_params,xlist,ylist,initpt)


#makes final sets for graphing
def make_table(uptris, downtris):
    make_table = np.zeros((5,31))
    make_table[0,0:3] = uptris[0,0,:,0]
    make_table[1,0:3] = downtris[0,0,:,1]
    make_table[2, 0:3] = downtris[0, 1, :, 1]
    make_table[3, 0:3] = downtris[0, 2, :, 1]
    make_table[4, 0:3] = downtris[0, 3, :, 1]

    for i in range(14):

        make_table[0,2*(i+1)+1:2*(i+2)+1] = uptris[i+1,0,1:,0]
        
        for j in range(4):
            make_table[j+1,2*(i+1)+1:2*(i+2)+1] = downtris[i+1,j,1:,1]


    return make_table


table = make_table(uptris,downtris)


fig = plt.figure()
ax = plt.axes(projection = '3d')



x_grid = np.array(0 + (15 - 0) * np.linspace(0, 1, 31) ** 2, dtype='float32')

y_grid = np.linspace(0,5,5)

X,Y = np.meshgrid(x_grid,y_grid)

ax.plot_wireframe(Y,X,table)
ax.scatter3D(Y,X,data.T,cmap='binary')

plt.show()


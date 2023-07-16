import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time as time
import parallel_run_numba_regression
from numba.cuda.api import synchronize
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import time as time
import operator

from numba import cuda, float32, int32, void
#from tdqm import tdqm


box = np.load('/Users/ari/Downloads/value_fun.npy')

data = box[:,:,0,0,0,0,0]
n = 15
m = 14


data = box[:,:,3,5,0,0,15]
x_grid = np.array(0 + (15 - 0) * np.linspace(0, 1, 31) ** 2, dtype='float32')
y_grid = np.linspace(0,5,5)
value_grid = np.zeros(data.shape)
x_vals = np.zeros((int(np.ceil(len(data[:,0])/2)),5,3))
y_vals = np.zeros((int(np.ceil(len(data[:,0])/2)),5,2))
x_params = np.flip(np.linspace(0,40,n))
y_params = np.flip(np.linspace(0,10,m))
x_params = np.repeat(x_params.reshape(n,1),5,axis = 1)*.1
y_params = np.repeat(y_params.reshape(m,1),1, axis = 1)
xlist = np.zeros((3,15))
ylist = np.zeros((5,2))
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
print(true_up.shape)
print(xlist)
print(ylist)
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


def concheck(params):
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

def alt_concheck(params):
    concheck = 0
    for i in range(16):
        for q in range(3):
            if params[i,q+1] > params[i,q]:
                concheck += 1

    return concheck

x_params = interpol(true_up,true_down,xlist)
y_params = get_full_params(true_up,true_down)
print(y_params.shape)

p1 = concheck(x_params)
p2 = alt_concheck(y_params)
print(p1)
print(p2)
type = 0
countlavich = 0
index = np.zeros()
for d1 in range(box[:,:,0,0,0,0,:].shape[2]):
    for d2 in range(box[:,:,0,0,0,:,d1].shape[2]):
        for d3 in range(box[:, :, 0, 0, :, d2, d1].shape[2]):
            for d4 in range(box[:, :, 0, :, d3, d2, d1].shape[2]):
                for d5 in range(box[:, :, :, d4, d3, d2, d1].shape[2]):
                    data = box[:,:,d5,d4,d3,d2,d1]
                    type += 1
                    if type % 100 == 0:
                        print(type)
                    for i in range(15):
                        for j in range(4):
                            functional_val[i, j, :, :] = data[2 * i:2 * (i + 1) + 1, j:j + 2]
                    true_up = np.array(functional_val)
                    true_down = np.array(functional_val)
                    true_up[:, :, 1:3, 1] = 0
                    true_down[:, :, 0:2, 0] = 0
                    x_params = interpol(true_up,true_down,xlist)
                    y_params = get_full_params(true_up,true_down)
                    c1 = concheck(x_params)
                    c2 = alt_concheck(y_params)
                    if c1+c2 == 0:
                        countlavich += 1
                    else:
                        index.append([d5,d4,d3,d2,d1])



print(countlavich/type)
print(index)
time.sleep(10)
print(x_params)
print(y_params)
def locate_index(pt,xgrid,ygrid):
    newvallist = [0]
    ind = 0
    pind = 0
    for i in range(len(xgrid)-1):
        print(xgrid[i])
        time.sleep(1)
        if pt[0] > xgrid[i] and pt[0] < xgrid[i+1]:
            tanpt = pt[0]- xgrid[i]
            break

        ind += 1
    print('hi')
    for i in range(len(ygrid)-1):
        print(ygrid[i])
        time.sleep(1)
        if pt[1] > ygrid[i] and pt[1] < ygrid[i+1]:
            tpt2 = pt[1] - ygrid[i]
            break

        pind += 1
    print(tanpt)
    print(tpt2)
    angle = np.arctan(tanpt/tpt2)
    print(angle)
    lineval = np.tan(angle)*tanpt
    print(lineval)
    if tpt2 > lineval:
        up = 1
    else:
        up = 0



    return [ind,pind, up]



print(locate_index([5,3],x_grid,y_grid))



def functional_estimate(x_params,y_params,xgrid,ygrid,pt,initpt,ind,pind,up):
    app = initpt
    for i in range(ind):
        app += x_params[i,0]*xgrid[2,i]
    for j in range(pind):
        app += y_params[ind,j]*1.25

    if up == 0:
        app+= pt[0]*x_params[ind,pind]
        app += pt[1] * y_params[ind,pind]
    else:
        app +=  pt[1] * y_params[ind,pind]
        app += pt[0] * x_params[ind,pind+1]
        app -= pt[1] * y_params[ind,pind+1]

    return app
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

    return true_up,true_down
def masterdriver(data,gridsize):
    space = np.zeros((15,5,data.shape[2],data.shape[3]*data.shape[4]*data.shape[5]*data.shape[6]))
    yspace = np.zeros((1,4,space.shape[2]))
    index = []
    countlavich= 0
    for d1 in range(box[:, :, 0, 0, 0, 0, :].shape[2]):
        for d2 in range(box[:, :, 0, 0, 0, :, d1].shape[2]):
            for d3 in range(box[:, :, 0, 0, :, d2, d1].shape[2]):
                for d4 in range(box[:, :, 0, :, d3, d2, d1].shape[2]):
                    for d5 in range(box[:, :, :, d4, d3, d2, d1].shape[2]):
                        data = box[:, :, d5, d4, d3, d2, d1]
                        type += 1
                        if type % 100 == 0:
                            print(type)
                        for i in range(15):
                            for j in range(4):
                                functional_val[i, j, :, :] = data[2 * i:2 * (i + 1) + 1, j:j + 2]
                        true_up = np.array(functional_val)
                        true_down = np.array(functional_val)
                        true_up[:, :, 1:3, 1] = 0
                        true_down[:, :, 0:2, 0] = 0
                        x_params = interpol(true_up, true_down, xlist)
                        space[15,5,d5,d4,d3,d2,d1] = x_params

                        y_params = get_full_params(true_up, true_down)
                        c1 = concheck(x_params)
                        c2 = alt_concheck(y_params)
                        if c1 + c2 == 0:
                            countlavich += 1
                        else:
                            index.append([d5, d4, d3, d2, d1])
                        yspace[1, 4, d5, d4, d3, d2, d1] = y_params[:,0]
    aggregatedata=np.array((15,4,1))
    aggregatedatay = np.array((1,4,1))
    for indexes in index:
        aggregatedata.append(space[:,:,indexes[0],indexes[1],indexes[2],indexes[3],indexes[4]])
        aggregatedatay.append(yspace[:,:,indexes[0],indexes[1],indexes[2],indexes[3],indexes[4]])
    n = int(np.ciel(len(index)/gridsize))
    for i in range(n):
        true_up_set = np.zeros((15,4,3,2,gridsize))
        true_down_set = np.zeros((15,4,3,2,gridsize))
        for j in range(gridsize):
            true_up_set[:,:,:,:,j], true_down_set[:,:,:,:,j] = organizedata(data[:,:,index[j][0],index[j][1],index[j][2],index[j][3],index[j][4]])
            
        usedata = aggregatedata[i*gridsize:(i+1)*gridsize]
        usedatay = aggregatedata[i*gridsize:(i+1)*gridsize]
        d_true_up_set = cuda.to_device(true_up_set)
        d_true_down_set = cuda.to_device(true_down_set)
        d_aggregatedata = cuda.to_device(aggregatedata)
        d_aggregatedatay = cuda.to_device(aggregatedatay)
        d_xlist = cuda.to_device(xlist)
        d_ylist = cuda.to_device(ylist)
        d_initpt = 0
        ping = np.array(np.zeros(1),dtype = 'int32')
        d_ping = cuda.to_device(ping)
        parallel_run_numba_regression.kernel[gridsize,1](d_true_up_set,d_true_down_set,d_aggregatedata,d_aggregatedatay,d_xlist,d_ylist,d_initpt,d_ping)
        aggregatedata[i*gridsize:(i+1)*gridsize] = d_aggregatedata.copy_to_host()
        aggregatedatay[i*gridsize:(i+1)*gridsize] = d_aggregatedatay.copy_to_host()
        true_up_set = d_true_up_set.copy_to_host()
        true_down_set = d_true_down_set.copy_to_host()
        ping = d_ping.copy_to_host()
    
    
    for indexes in index:

        space[:,:,indexes[0],indexes[1],indexes[2],indexes[3],indexes[4]] = aggregatedata[:,:,i]
        yspace[:, :, indexes[0], indexes[1], indexes[2], indexes[3], indexes[4]] = aggregatedatay[:,:,i]

print(functional_estimate(x_params,y_params,xlist,ylist,[5,3],true_up[0,0,0,0],int(np.floor(17/2)),2,1),'ding')
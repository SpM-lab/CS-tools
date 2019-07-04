# Example code for basis pursuit method
# Copyright (C) 2019 ISSP, Univ. of Tokyo

# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. 

# This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
# GNU General Public License for more details. 

# You should have received a copy of the GNU General Public License 
# along with this program.  If not, see <http://www.gnu.org/licenses/>. 

import argparse
import numpy as np

def SoftThr(x, lamb):
    y = np.zeros_like(x)
    temp = np.array(x >lamb)
    y[temp] = x[temp] - lamb
    temp = np.array(x < -lamb) 
    y[temp] = x[temp] + lamb
    return y

if __name__ == '__main__':
    # Dimension of the signal
    N = 1000
    
    M = 100
    K = 20    
    seed = 1234
    np.random.seed(seed)
    A = np.random.randn(M,N);
    
    #Make answer vector
    xanswer = np.zeros(N)
    xanswer[:K] = np.random.randn(K)
    xanswer = np.random.permutation(xanswer)

    y_calc = np.dot(A, xanswer)

    x = np.zeros(N)
    z = np.zeros(N)
    u = np.zeros(N)

    mu = 1.0
    
    A1 = np.dot(A.T, np.linalg.inv(np.dot(A, A.T)))
    A2 = np.eye(N) - np.dot(A1, A)

    T = 10000
    for t in range(T):
        zold = z
        x = np.dot(A1, y_calc) + np.dot(A2, z - u)
        z = SoftThr(x+u, 1./mu)
        u = u + x - z
        res = np.linalg.norm(z-zold)/np.linalg.norm(z)
        print(t, res)
        if res < 1e-5:
            break

    x_ridge = np.dot(A1, y_calc)
    
    with open("predict_x.dat", "w") as f:
        for i in range(N):
            f.write("{} {} {} {}\n".format(i, z[i], x_ridge[i], xanswer[i] ))

    with open("y_answer.dat", "w") as f:
        for i in range(M):
            f.write("{} {}\n".format(i, y_calc[i]))

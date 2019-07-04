# Example code for LASSO
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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from scipy.special import legendre
from sklearn.linear_model import Lasso
from scipy.integrate import simps
import numpy as np
import csv
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='legendrefit.py',
        usage='Fit function to map Legendre basis.',
        description='description',
        epilog='end',
        add_help=True,
    )

    parser.add_argument('-nr', dest='nrow', \
                        help='Matrix Row size.', \
                        type= int, default=1000)
    parser.add_argument('-nc', dest='ncol', \
                        help='Matrix Column size.', \
                        type= int, default=100)

    parser.add_argument('-nv', dest='nv', \
                        help='non-zero vector components.', \
                        type= int, default=20)

    args = parser.parse_args()
    nrow = args.nrow
    ncol = args.ncol
    nvec = args.nv

    np.random.seed(1234)
    #Make Random matrix
    AMat = np.random.randn(ncol, nrow)

    #Make answer vector
    xanswer = np.zeros(nrow)
    xanswer[:nvec] = np.random.randn(nvec)
    xanswer = np.random.permutation(xanswer)
    y_answer = np.dot(AMat, xanswer)

    #Add noise
    y_rand = abs(y_answer).mean()*1e-1*np.random.randn(y_answer.shape[0])
    y_calc = y_answer+y_rand
    
    with open("y_with_noise.dat", "w") as f:
        for i, _y in enumerate(y_calc):
            f.write("{} {} {}\n".format(i+1, _y, y_answer[i]))

    
    score_list = []
    scores_CV_list = []
    l0_list = []
    for _alpha in 10**(np.arange(-4, 1, 0.05)):
        #Elbow method
        clf = Lasso(alpha=_alpha, max_iter = 10000)
        clf.fit(X=AMat, y=y_calc)
        x_predict = clf.coef_
        y_predict = clf.predict(X=AMat)
        u = np.sum(np.abs(y_predict-y_calc)**2)
        v = np.sum(np.abs(y_calc-y_calc.mean())**2)
        score_list.append([_alpha, clf.score(X=AMat, y=y_calc), x_predict])
        l0_list.append((_alpha, np.count_nonzero(x_predict)))

        scores_CV = []
        kf = KFold(n_splits = 5)
        for train_index, test_index in kf.split(AMat):
            X_train, X_test = AMat[train_index], AMat[test_index]
            y_train, y_test = y_calc[train_index], y_calc[test_index]
            clf.fit(X=X_train, y=y_train)
            scores_CV.append(clf.score(X=X_test, y=y_test))
        scores_CV_list.append([_alpha, np.mean(scores_CV), np.std(scores_CV)])

    with open("l0.dat", "w") as f:
        for l0 in l0_list:
            f.write("{} {}\n".format(l0[0], l0[1]))        
        
    with open("score_elbow.dat", "w") as f:
        for score in score_list:
            f.write("{} {}\n".format(score[0], score[1]))

    with open("score_CV.dat", "w") as f:
        for score in scores_CV_list:
            f.write("{} {} {}\n".format(score[0], score[1], score[2]))

    max_idx = np.argmax(np.array(scores_CV_list)[:,1])
    clf = Lasso(alpha=scores_CV_list[max_idx][0], max_iter = 10000)
    clf.fit(X=AMat, y=y_calc)
    x_predict = clf.coef_
    with open("lasso_cv.dat", "w") as f:
        f.write("#alpha={}".format(scores_CV_list[max_idx][0]))
        for i in range(nrow):
            f.write("{} {} {}\n".format(i, xanswer[i], x_predict[i]))
    
    output_idx = [i for i in range(0, 100, 10)]
    for _idx in output_idx:
        with open("lasso_alpha{}.dat".format(_idx), "w") as f:
            f.write("#alpha={}".format(score_list[_idx][0]))
            for i, _x in enumerate(score_list[_idx][2]):
                f.write("{} {} {}\n".format(i, xanswer[i], _x))
                

    
    

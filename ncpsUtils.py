'''
author: Rohan Vardhan
PID: 4177660
Course: Modeling and Analysis of Networked Cyber-Physical Systems

TERM PROJECT

Created on 3/20/2017

DATE 		COMMENT

3/20/2017	added function to read data from respective directories
3/20/2017	added functions to classify, filter and split data
3/21/2017 	added functions for confusion matrix and to compute scores
'''

# importing dependencies
import csv
import os
import numpy as np

# function for reading data from each file given the directory
def readData(filePath, n):
	dataset = []
    	count = 1
    	for file in os.listdir(filePath):
        	if count > n:
            		break
        	with open(filePath + '9000.txt') as fx:
            		reader = csv.reader(fx)
            		for line in reader:
                		dataset.append(map(float, line))
        	count = count + 1
    	return dataset	

# function to classify labels in 1 or 0 given warning range and separation distance
def classifyLabels(wr, sd):
	if len(wr) == len(sd):
		l = [1 if wr[i] >= sd[i] else 0 for i in range(0,len(sd))]
	return l

# function to filter out data where the separation distance is less than 200
def filtered(ground_truths, training_data, important_range=200):
    	important_indices = [i for i in range(len(ground_truths)) if ((ground_truths[i][4] - ground_truths[i][5]) < important_range)]
    	gt = [ground_truths[i] for i in important_indices]
    	train = [training_data[i] for i in important_indices]
    	return gt, train

# function for splitting data into training and test sets
def splitData(gt_data, v_data, percent):
	v_data = np.asarray(v_data, dtype = np.float32)
	gt_data = np.asarray(gt_data, dtype = np.float32)
	v_data = np.reshape(v_data, (-1, 6))
	gt_data = np.reshape(gt_data, (-1, 6))
	split = int(round(percent*len(v_data))) #percent for training size

	gt_train = gt_data[:split]
	gt_test = gt_data[split:]
	v_train = v_data[:split]
	v_test = v_data[split:]

	vfeatures_train = [a[:4] for a in v_train]
	vfeatures_test = [c[:4] for c in v_test]
	vlabels_train = [1 if(v_train[i][5] > v_train[i][4]) else 0 for i in range(len(v_train))]
	vlabels_test = [1 if(v_test[i][5] > v_test[i][4]) else 0 for i in range(len(v_test))]

	vfeatures_train = np.reshape(vfeatures_train, (-1,4))
	vlabels_train = np.reshape(vlabels_train, (-1,1))
	vfeatures_test = np.reshape(vfeatures_test, (-1,4))
	vlabels_test = np.reshape(vlabels_test, (-1,1))

	gtfeatures_train = [a[:4] for a in gt_train]
	gtfeatures_test = [c[:4] for c in gt_test]
	gtlabels_train = [1 if(gt_train[i][5] > gt_train[i][4]) else 0 for i in range(len(gt_train))]
	gtlabels_test = [1 if(gt_test[i][5] > gt_test[i][4]) else 0 for i in range(len(gt_test))]

	gtfeatures_train = np.reshape(gtfeatures_train, (-1,4))
	gtlabels_train = np.reshape(gtlabels_train, (-1,1))
	gtfeatures_test = np.reshape(gtfeatures_test, (-1,4))
	gtlabels_test = np.reshape(gtlabels_test, (-1,1))


	return vfeatures_train, vlabels_train, vfeatures_test, vlabels_test, gtfeatures_train, gtlabels_train, gtfeatures_test, gtlabels_test

# following 4 functions compute elements of confusion matrix as shown below
def both_safe(pred, gt):
	return (pred == gt and pred == 0)
  
def both_threat(pred, gt):
	return (pred == gt and pred == 1)
  
def false_threat(pred, gt):
	return (pred == 1 and gt == 0)

def false_safe(pred, gt):
	return (pred == 0 and gt == 1)

# function to compute scores from confusion matrix
def computeScore(pred, gt):
	
#        Confusion matrix ->
#                                     Actual data
#             pred                 Negative   Positive
#     	Negative (safe)               a       c
#     	Positive (threatening)        b       d
    	
	
	a = float(sum([1 if(both_safe(pred[i], gt[i])) else 0 for i in range(len(pred))]))
	b = float(sum([1 if(false_threat(pred[i], gt[i])) else 0 for i in range(len(pred))]))
	c = float(sum([1 if(false_safe(pred[i], gt[i])) else 0 for i in range(len(pred))]))
	d = float(sum([1 if(both_threat(pred[i], gt[i])) else 0 for i in range(len(pred))]))

	if(a+d > 0):
		acc1 = (a+d)/(a+b+c+d)

	if(b+d > 0):
		precision = d/(b+d)


	if(c+d > 0):
    		tp = d/(c+d)
    		fn = c/(c+d)

	if(a+b > 0):
    		tn = a/(a+b)
    		fp = b/(a+b)

	acc1 = round(acc1,3)
	precision = round(precision,3)
	tp = round(tp,3)
	fn = round(fn,3)
	tn = round(tn,3)
	fp = round(fp,3)
	
	return acc1, precision, tp, fn, tn, fp

# function to save scores into files for each case
def saveResults(name, s, s1, s2, s3, s4):
	s = np.asarray(s, dtype = np.float32)
	s1 = np.asarray(s1, dtype = np.float32)
	s2 = np.asarray(s2, dtype = np.float32)
	s3 = np.asarray(s3, dtype = np.float32)
	s4 = np.asarray(s4, dtype = np.float32)	
	
	with open(name, "wb") as f:
		savedata = csv.writer(f)
		savedata.writerow(s)
		savedata.writerow(s1)
		savedata.writerow(s2)
		savedata.writerow(s3)
		savedata.writerow(s4)
	f.close()


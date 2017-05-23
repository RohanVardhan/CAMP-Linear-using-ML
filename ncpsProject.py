'''
author: Rohan Vardhan
PID: 4177660
Course: Modeling and Analysis of Networked Cyber-Physical Systems

TERM PROJECT

Created on 3/24/2017

DATE 		COMMENT

3/24/2017	added function for training and testing of machine learning classifiers
3/24/2017 	added main function
4/25/2017	added code to plot and compare warning generated

'''

# importing dependencies
import csv
import os
import numpy as np
import ncpsUtils as fu
import matplotlib.pyplot as plt
import sys
from time import time

# number of scenarios
#n = 823
n=1
def ml(per):

	###################################################################################################################
	# reading ground truth
	print "Reading data from ground truth directory . . ."
	gt_dir = r'./cleanData/PER_0/'
	gtData = fu.readData(gt_dir, n)

	# reading other vehicular data
	print "Reading data from vehicle directory . . ."
	vehicle_dir = r'./cleanData/PER_'+str(per)+'/'
	vData = fu.readData(vehicle_dir, n)

	groundTruthData, vehicleData = fu.filtered(gtData, vData)

	###################################################################################################################
	#print "Splitting dataset to obtain training and testing datasets"
	percent = [0.7] #[0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

	for i in range(len(percent)):

		vfeatures_train, vlabels_train, vfeatures_test, vlabels_test, gtfeatures_train, gtlabels_train, gtfeatures_test, gtlabels_test = fu.splitData(groundTruthData, vehicleData, percent[i])
	
		###################################################################################################################
		print "Predicting . . ." 
		from sklearn import linear_model
		from sklearn import tree
		from sklearn import ensemble
		from sklearn import neural_network 

		# defining classifiers with appropriate parameters
		clf1 = linear_model.LogisticRegression()
		clf2 = tree.DecisionTreeClassifier(min_samples_split = 40)
		clf3 = neural_network.MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(20,))
		clf4 = ensemble.RandomForestClassifier(n_estimators = 100)
		###################################################################################################################
		print "Training & testing . . .\n"
		t1 = time()
		clf1.fit(vfeatures_train, vlabels_train) 
		print "Time for LR: ", round(time()-t1,3)
		pred1 = clf1.predict(vfeatures_test)
		#print "\nCoefficients of LR: \n", clf1.coef_ 
		# uncomment the above line to get coefficients of LR

		t2 = time()
		clf2.fit(vfeatures_train, vlabels_train) 
		print "Time for DT: ", round(time()-t2,3)
		pred2 = clf2.predict(vfeatures_test)
		#print "\nFeature importance of DT: \n", clf2.feature_importances_ 
		# uncomment the above line to get important features of DT

		t3 = time()
		clf3.fit(vfeatures_train, vlabels_train)
		print "Time for NN: ", round(time()-t3,3) 
		pred3 = clf3.predict(vfeatures_test)
		#print "\nCoefficients of NN: \n", clf3.coefs_
		# uncomment the above line to get coefficients of NN

		t4 = time()
		clf4.fit(vfeatures_train, vlabels_train) 
		print "Time for RF: ", round(time()-t4,3)
		pred4 = clf4.predict(vfeatures_test)
		#print "\nFeature importance of RF: \n", clf4.feature_importances_ 
		# uncomment the above line to get important features of RF

		# uncomment this block to get alerts generated
		'''
		plt.subplot(5,1,1)
		plt.plot([i for i in range(len(vlabels_test))], vlabels_test, label="Camp", marker='8')
		plt.xlabel("Time step")
		plt.ylabel("CAMP")

		plt.subplot(5,1,2)
		plt.plot([i for i in range(len(vlabels_test))], pred1, label="LR", marker='8')
		plt.xlabel("Time step")
		plt.ylabel("LR")

		plt.subplot(5,1,3)
		plt.plot([i for i in range(len(vlabels_test))], pred2, label="DT", marker='8')
		plt.xlabel("Time step")
		plt.ylabel("DT")

		plt.subplot(5,1,4)
		plt.plot([i for i in range(len(vlabels_test))], pred3, label="NN", marker='8')
		plt.xlabel("Time step")
		plt.ylabel("NN")

		plt.subplot(5,1,5)
		plt.plot([i for i in range(len(vlabels_test))], pred4, label="RF", marker='8')
		plt.xlabel("Time step")
		plt.ylabel("RF")
		plt.savefig('./warning2.png')
		plt.show()
		'''

		###################################################################################################################
		print "Computing scores . . ."
		print "\nScores using CAMP Linear and ML Classifiers -->\n"
		s = fu.computeScore(vlabels_test, gtlabels_test) #CAMP Linear algo	
		s1 = fu.computeScore(pred1, gtlabels_test)
		s2 = fu.computeScore(pred2, gtlabels_test)
		s3 = fu.computeScore(pred3, gtlabels_test)	
		s4 = fu.computeScore(pred4, gtlabels_test)	
		
		# save score in appropriate directories to plot
		results_dir = r'./scores/' + str(percent[i]) + '_percent/'
		fileName = 'score_' + str(per) + '_' + str(percent[i])+ '_percent.txt'
		name = results_dir + fileName
		fu.saveResults(name, s, s1, s2, s3, s4)
		

# various values of PER used
per = [0]#, 20, 40, 60, 80, 90]
for i in range(len(per)):
	ml(per[i])

'''
author: Rohan Vardhan
PID: 4177660
Course: Modeling and Analysis of Networked Cyber-Physical Systems

TERM PROJECT

Created on 3/25/2017

DATE 		COMMENT

3/25/2017 	added code to plot various performance parameters
4/24/2017	added code to visualize warning range and separation distance
'''

# importing dependencies
import matplotlib.pyplot as plt
import os
import csv

per = [0, 20, 40, 60, 80, 90] # varying PER
percent = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # varying training size

# directory for storing plots for accuracies, precision, TPR, TNR, FPR, FNR
save_dir = r'./plots/'
# directory for storing warning range and separation distance (for visualization)
save_CDFdir = r'./cdf/'

for j in range(len(percent)):
	resultsData = []

	# reading data from scores directory for plotting
	results_dir = r'./scores/' + str(percent[j]) + '_percent/'
	for file in sorted(os.listdir(results_dir)):
		with open(results_dir+file) as fx:
			reader = csv.reader(fx)
			for r in reader:
				resultsData.append(map(float, r))

	# plotting various performance parameters
	# comment block of code corresponding to parameters which are not required
	plt.figure(1)
	plt.ylabel("Accuracy")
	plt.xlabel("PER")
	plt.title('Accuracy vs PER')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+0][0] for i in range(len(per))], label = "CAMP", marker='8')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+1][0] for i in range(len(per))], label = "LR", marker='d')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+2][0] for i in range(len(per))], label = "DT", marker='>')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+3][0] for i in range(len(per))], label = "NN", marker='v')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+4][0] for i in range(len(per))], label = "RF", marker='s')
	plt.legend(loc=1)
	plt.savefig(save_dir + 'ACCURACY_'+str(percent[j]*100)+'_.png')
	#plt.show()
	
	plt.figure(2)
	plt.ylabel("Precision")
	plt.xlabel("PER")
	plt.title('Precision vs PER')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+0][1] for i in range(len(per))], label = "CAMP", marker='8')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+1][1] for i in range(len(per))], label = "LR", marker='d')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+2][1] for i in range(len(per))], label = "DT", marker='>')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+3][1] for i in range(len(per))], label = "NN", marker='v')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+4][1] for i in range(len(per))], label = "RF", marker='s')
	plt.legend(loc=1)
	plt.savefig(save_dir + 'PRECISION_'+str(percent[j]*100)+'_.png')
	#plt.show()

	plt.figure(3)
	plt.ylabel("True Positive")
	plt.xlabel("PER")
	plt.title('True positive vs PER')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+0][2] for i in range(len(per))], label = "CAMP", marker='8')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+1][2] for i in range(len(per))], label = "LR", marker='d')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+2][2] for i in range(len(per))], label = "DT", marker='>')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+3][2] for i in range(len(per))], label = "NN", marker='v')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+4][2] for i in range(len(per))], label = "RF", marker='s')
	plt.legend(loc=1)
	plt.savefig(save_dir + 'TPR_'+str(percent[j]*100)+'_.png')
	#plt.show()

	plt.figure(4)
	plt.ylabel("False Negative")
	plt.xlabel("PER")
	plt.title('False negative vs PER')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+0][3] for i in range(len(per))], label = "CAMP", marker='8')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+1][3] for i in range(len(per))], label = "LR", marker='d')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+2][3] for i in range(len(per))], label = "DT", marker='>')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+3][3] for i in range(len(per))], label = "NN", marker='v')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+4][3] for i in range(len(per))], label = "RF", marker='s')
	plt.legend(loc=1)
	plt.savefig(save_dir + 'FNR_'+str(percent[j]*100)+'_.png')
	#plt.show()
	plt.figure(5)	

	plt.ylabel("True Negative")	
	plt.xlabel("PER")	
	plt.title('True negative vs PER')	
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+0][4] for i in range(len(per))], label = "CAMP", marker='8')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+1][4] for i in range(len(per))], label = "LR", marker='d')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+2][4] for i in range(len(per))], label = "DT", marker='>')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+3][4] for i in range(len(per))], label = "NN", marker='v')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+4][4] for i in range(len(per))], label = "RF", marker='s')
	plt.legend(loc=1)
	plt.savefig(save_dir + 'TNR_'+str(percent[j]*100)+'_.png')
	#plt.show()

	plt.figure(6)
	plt.ylabel("False Positive")
	plt.xlabel("PER")
	plt.title('False positive vs PER')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+0][5] for i in range(len(per))], label = "CAMP", marker='8')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+1][5] for i in range(len(per))], label = "LR", marker='d')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+2][5] for i in range(len(per))], label = "DT", marker='>')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+3][5] for i in range(len(per))], label = "NN", marker='v')
	plt.plot([per[i] for i in range(len(per))], [resultsData[(i*5)+4][5] for i in range(len(per))], label = "RF", marker='s')
	plt.legend(loc=1)
	plt.savefig(save_dir + 'FPR_'+str(percent[j]*100)+'_.png')
	#plt.show()


# uncomment the below block of code to visualize warning range and separation distance (for a particular scenario)
'''
file = '9123.txt' #for example

with open('./cleanData/PER_0/'+file) as f:
	data = []
	reader = csv.reader(f)
	for line in reader:
		data.append(map(float, line))
	plt.figure(10)
	plt.plot([i for i in range(len(data))], [data[i][4] for i in range(len(data))], label = "Separation distance", color="b")
	plt.plot([i for i in range(len(data))], [data[i][5] for i in range(len(data))], label = "Warning Range", color="r")
	plt.xlabel("Time step")
	plt.ylabel("Distance")
	plt.legend(loc=1)
	plt.savefig(save_CDFdir+file+'.png')
	#plt.show()
'''
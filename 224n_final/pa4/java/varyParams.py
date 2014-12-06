import os

num_iters = 1;
window_size = 3;
h_size = 100;
learning_rate = 0.001
regularizeParam = 0;

#///////////////////////////////////////////////////////////////////////////////#

'''
#Vary number epochs (1 -> 20)
os.popen("rm -f test1.out");
print os.popen("ant").read();
print "NUMBER EPOCHS"
for i in range(1, 21):
	arr = [str(i), str(window_size), str(h_size), str(learning_rate), str(regularizeParam)]
	x = os.popen("java -Xmx1g -cp \"extlib/*:classes\" cs224n.deep.NER ../data/train ../data/test " + " ".join(arr) + " -print > test1.out ; ../conlleval -r -d \"\t\" < test1.out").read(); 

	#Get f1 score
	score = x.split('\n')[1].split(';')[-1].split(":")[1].strip()

	print "\t".join([str(i), str(score)])
'''

#///////////////////////////////////////////////////////////////////////////////#

'''
# #Vary Window Size (1 -> 9)
os.popen("rm -f test2.out");
print os.popen("ant").read();
print "WINDOW_SIZE"
for i in range(0, 5):
	w = 1 + (i * 2)
	arr = [str(num_iters), str(w), str(h_size), str(learning_rate), str(regularizeParam)]
	x = os.popen("java -Xmx1g -cp \"extlib/*:classes\" cs224n.deep.NER ../data/train ../data/test " + " ".join(arr) + " -print > test2.out ; ../conlleval -r -d \"\t\" < test2.out").read(); 

	#Get f1 score
	score = x.split('\n')[1].split(';')[-1].split(":")[1].strip()

	print "\t".join([str(w), str(score)])
'''

#///////////////////////////////////////////////////////////////////////////////#

'''
#Vary Window Size (10 -> 200 factor of 10)
os.popen("rm -f test3.out");
print os.popen("ant").read();
print "HIDDEN_SIZE"
for i in range(1, 21):
	h = i * 10
	arr = [str(num_iters), str(window_size), str(h), str(learning_rate), str(regularizeParam)]
	x = os.popen("java -Xmx1g -cp \"extlib/*:classes\" cs224n.deep.NER ../data/train ../data/test " + " ".join(arr) + " -print > test3.out ; ../conlleval -r -d \"\t\" < test3.out").read(); 

	#Get f1 score
	score = x.split('\n')[1].split(';')[-1].split(":")[1].strip()

	print "\t".join([str(h), str(score)])
'''


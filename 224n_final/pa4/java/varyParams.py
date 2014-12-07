import os

num_iters = 4;
window_size = 3;
h_size = 100;
learning_rate = 0.001
regularizeParam = 0;

#///////////////////////////////////////////////////////////////////////////////#


# #Vary number epochs (1 -> 20)
# os.popen("rm -f test11111.out");
# print os.popen("ant").read();
# print "NUMBER EPOCHS"
# for i in range(20, 21):
# 	arr = [str(i), str(window_size), str(h_size), str(learning_rate), str(regularizeParam)]
# 	x = os.popen("java -Xmx1g -cp \"extlib/*:classes\" cs224n.deep.NER ../data/train ../data/dev " + " ".join(arr) + " -print > test11111.out ; ../conlleval -r -d \"\t\" < test11111.out").read(); 

# 	#Get f1 score
# 	score = x.split('\n')[1].split(';')[-1].split(":")[1].strip()

# 	print "\t".join([str(i), str(score)])


#///////////////////////////////////////////////////////////////////////////////#


# # #Vary Window Size (1 -> 9)
# os.popen("rm -f test2.out");
# print os.popen("ant").read();
# print "WINDOW_SIZE"
# for i in range(0, 5):
# 	w = 1 + (i * 2)
# 	arr = [str(num_iters), str(w), str(h_size), str(learning_rate), str(regularizeParam)]
# 	x = os.popen("java -Xmx1g -cp \"extlib/*:classes\" cs224n.deep.NER ../data/train ../data/dev " + " ".join(arr) + " -print > test2.out ; ../conlleval -r -d \"\t\" < test2.out").read(); 

# 	#Get f1 score
# 	score = x.split('\n')[1].split(';')[-1].split(":")[1].strip()

# 	print "\t".join([str(w), str(score)])


#///////////////////////////////////////////////////////////////////////////////#


word = "test2.out"
#Vary Hidden Size (10 -> 200 factor of 10)
os.popen("rm -f " + word)
print os.popen("ant").read();
print "HIDDEN_SIZE"
for i in range(16, 21):
	h = i * 10
	arr = [str(num_iters), str(window_size), str(h), str(learning_rate), str(regularizeParam)]
	x = os.popen("java -Xmx1g -cp \"extlib/*:classes\" cs224n.deep.NER ../data/train ../data/dev " + " ".join(arr) + " -print > " + word + " ; ../conlleval -r -d \"\t\" < " + word).read(); 

	#Get f1 score
	score = x.split('\n')[1].split(';')[-1].split(":")[1].strip()

	print "\t".join([str(h), str(score)])


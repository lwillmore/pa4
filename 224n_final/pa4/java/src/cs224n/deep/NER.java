package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {
    public static void main(String[] args) throws IOException {
	if (args.length < 2) {
	    System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
	    return;
	}	    

	// this reads in the train and test datasets
	List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
	List<Datum> testData = FeatureFactory.readTestData(args[1]);	

	int NUM_ITERATIONS = 1;
	int WINDOW_SIZE = 3;
	int HIDDEN_NODES = 100;
	double LEARNING_RATE = 0.001;
	double REGULARIZE = 0.00;

	if (args.length > 3){
		NUM_ITERATIONS = Integer.parseInt(args[2]);
		WINDOW_SIZE = Integer.parseInt(args[3]);
		HIDDEN_NODES = Integer.parseInt(args[4]);
		LEARNING_RATE = Double.parseDouble(args[5]);
		REGULARIZE = Double.parseDouble(args[6]);
	}
	
	//	read the train and test data
	FeatureFactory.initializeVocab("../data/vocab.txt");
	
	// Initialize model 
	WindowModel model = new WindowModel(WINDOW_SIZE, HIDDEN_NODES, LEARNING_RATE, REGULARIZE, NUM_ITERATIONS);
	model.initWeights();

	//Initialization of Word Vectors, either random or not
	FeatureFactory.readWordVectors("../data/wordVectors.txt");
	//FeatureFactory.initRandomWordVectors();

	
	//Baseline
	// model.baseLineTrain(trainData);
	// model.baseLineTest(testData);
	
	//Regular
	model.train(trainData);
	model.test(testData);
    }
}
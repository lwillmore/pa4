package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;
import java.util.Random;

import java.text.*;


public class WindowModel {

	protected SimpleMatrix L, W, Wout, U;
	//
	public int windowSize,wordSize, hiddenSize;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		//TODO
	}
	
	public static final int H = 100; //Hidden Layer elements
	public static final int C_N = 150;
	public static final int NUM_FEATURES = 5;
	public static final int N = 50;

	public void initWeights(){
		// initialize with bias inside as the last column
		
		double fanIn = C_N;
		double fanOut = H;
		double epsilon = Math.sqrt(6) / Math.sqrt(fanIn + fanOut);

		Random rand = new Random();
		W = SimpleMatrix.random(H, C_N + 1, -1 * epsilon, epsilon, rand);
		U = new SimpleMatrix(NUM_FEATURES, H + 1);


		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix...
	}

	public SimpleMatrix feedForward(SimpleMatrix x_1, SimpleMatrix x_2, SimpleMatrix x_3){
		//Create x vector, 151 x 1
		SimpleMatrix newX = new SimpleMatrix(C_N + 1, 1);
		int count = 0;
		for (int i = 0; i < N; i++){
			newX.set(count, 0, x_1.get(i, 0));
			newX.set(count+N, 0, x_2.get(i, 0));
			newX.set(count+N * 2, 0, x_3.get(i, 0));
			count++;
		}		
		newX.set(C_N, 0, 1);

		SimpleMatrix m = W.mult(newX);
		SimpleMatrix newM = new SimpleMatrix(H + 1, 1);
		for (int i = 0; i < H; i++){
			newM.set(i, 0, Math.tanh(m.get(i, 0)));
		}
		newM.set(H, 0, 1);
		SimpleMatrix finalMatrix = U.mult(newM);

		return SoftMaxScore(finalMatrix);
	}

	public SimpleMatrix SoftMaxScore(SimpleMatrix v){
		double denominator = 0.0;
		for (int i = 0; i < v.numRows(); i++){
			denominator += Math.exp(v.get(i,0));
		}
		SimpleMatrix g = new SimpleMatrix(v.numRows(), v.numCols());
		for (int i = 0; i < v.numRows(); i++){
			g.set(i, 0, Math.exp(v.get(i,0)) / denominator);
		}
		return g;
	}


	/*
	* Baseline Implementaiton
	*/
	HashMap<String, HashMap<String, Integer>> baseLineMap = new HashMap<String, HashMap<String, Integer>>();

	public void baseLineTrain(List<Datum> _trainData){
		for (Datum d : _trainData){
			String word = d.word;
			String label = d.label;
			if (baseLineMap.get(word) != null){
				HashMap<String, Integer> tagMap = baseLineMap.get(word);
				if(tagMap.get(label) != null){
					int count = tagMap.get(label);
					tagMap.put(label, count + 1);
				}
				else{
					tagMap.put(label, 1);
				}
			}
			else{
				HashMap<String, Integer> tagMap = new HashMap<String, Integer>();
				tagMap.put(label, 1);
				baseLineMap.put(word, tagMap);
			}
		}
	}
	public void baseLineTest(List<Datum> testData){
		for (Datum d : testData){
			String word = d.word;
			String label = d.label;
			if(baseLineMap.keySet().contains(d.word)){
				//Object has been seen

				Set<String> nerTags = baseLineMap.get(d.word).keySet();
				int largest = -1;
				String finalTag = "";
				for (String tag : nerTags){
					if (baseLineMap.get(d.word).get(tag) > largest){
						largest = baseLineMap.get(d.word).get(tag);
						finalTag = tag;
					}
				}
				System.out.println(word + "\t" + label + "\t" + finalTag);
			}
			else{
				//Object hasn't been seen, use O
				System.out.println(word + "\t" + label + "\t" + "O");
			}
		}
	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData ){
		SimpleMatrix x = new SimpleMatrix(50, 1);
		x.set(0, 1);
		feedForward(x, x, x);
	}

	
	public void test(List<Datum> testData){
		// TODO
	}
	
}

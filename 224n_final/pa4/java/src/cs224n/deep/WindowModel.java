package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;


public class WindowModel {

	protected SimpleMatrix L, W, Wout;
	//
	public int windowSize,wordSize, hiddenSize;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		//TODO
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		//TODO
		// initialize with bias inside as the last column
		// W = SimpleMatrix...
		// U for the score
		// U = SimpleMatrix...
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
	}

	
	public void test(List<Datum> testData){
		// TODO
		}
	
}

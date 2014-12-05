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
		WINDOW_SIZE = _windowSize;
		HIDDEN_ELEMENTS = _hiddenSize;
		alpha = _lr; //Learning Rate
	}
	
	public static int HIDDEN_ELEMENTS;
	public static int C_N;
	public static int WINDOW_SIZE;

	public static final int NUM_FEATURES = 5;
	public static final int N = 50;

	public static double alpha;
	public static final double epsilon = Math.pow(10, -4);
	

	public void initWeights(){
		// initialize with bias inside as the last column
		C_N = N * WINDOW_SIZE;
		double fanIn = C_N;
		double fanOut = HIDDEN_ELEMENTS;
		double epsilon = Math.sqrt(6) / Math.sqrt(fanIn + fanOut);

		Random rand = new Random();
		W = SimpleMatrix.random(HIDDEN_ELEMENTS, C_N + 1, -1 * epsilon, epsilon, rand);
		U = new SimpleMatrix(NUM_FEATURES, HIDDEN_ELEMENTS + 1);
	}

	public void feedForwardAndBackward(SimpleMatrix newX, SimpleMatrix labelVector, List<Integer> wordListIndex){
		//Forward Propagation
		SimpleMatrix m = W.mult(newX);
		SimpleMatrix newM = new SimpleMatrix(HIDDEN_ELEMENTS + 1, 1);
		
		for (int i = 0; i < HIDDEN_ELEMENTS; i++){
			newM.set(i, 0, Math.tanh(m.get(i, 0)));
		}
		newM.set(HIDDEN_ELEMENTS, 0, 1);

		SimpleMatrix finalMatrix = U.mult(newM);
		SimpleMatrix sigmoid = SoftMaxScore(finalMatrix);

		//Back propagation
		SimpleMatrix gradU = gradientU(sigmoid, newM, labelVector);
		SimpleMatrix gradW = gradientW(sigmoid, newM, newX, m, labelVector);
		SimpleMatrix gradL = updateL(finalMatrix, newM, m, labelVector);

		// gradientCheckU(gradU, newX, labelVector);
		gradientCheckW(gradW, newX, labelVector);

		U = U.minus(gradU.scale(alpha));
		W = W.minus(gradW.scale(alpha));

		//Update L
		for (int i = 0; i < wordListIndex.size(); i++){
			int column = wordListIndex.get(i);
			SimpleMatrix gradL_x = gradL.extractMatrix(i * N, (i+1) * N, 0, 1);
			SimpleMatrix rowVector = FeatureFactory.allVecs.extractVector(false, column);
			rowVector = rowVector.minus(gradL_x);

			for (int j = 0; j < rowVector.numRows(); j++){
				FeatureFactory.allVecs.set(j, column, rowVector.get(j, 0));
			}
		}
	}

	//Grad Check U
	public void gradientCheckU(SimpleMatrix gradU, SimpleMatrix newX, SimpleMatrix labelVector){
		SimpleMatrix left = gradU;
		
		int goodCount = 0;
		int badCount = 0;
		for (int row = 0; row < U.numRows(); row++){
			for (int col = 0; col < U.numCols(); col++){
				SimpleMatrix newUPlus = new SimpleMatrix(U);
				newUPlus.set(row, col, newUPlus.get(row, col) + epsilon);

				SimpleMatrix newUMinus = new SimpleMatrix(U);
				newUMinus.set(row, col, newUMinus.get(row, col) - epsilon);

				double plus = gradientHelper(newX, newUPlus, W, labelVector).get(0, 0);
				double minus = gradientHelper(newX, newUMinus, W, labelVector).get(0, 0);

				double right = (plus - minus) / (2 * epsilon);
				double num = Math.abs(left.get(row, col) - right);

				if (num > Math.pow(10, -7)){
					badCount++;
				}
				else{
					goodCount++;
				}
			}
		}
		System.out.println("GOODCOUNT (Below 10^-7): " + goodCount);
		System.out.println("BADCOUNT: " + badCount);
		System.out.println("");
	}

	//Grad Check W
	public void gradientCheckW(SimpleMatrix gradW, SimpleMatrix newX, SimpleMatrix labelVector){
		SimpleMatrix left = gradW;
		
		int goodCount = 0;
		int badCount = 0;
		for (int row = 0; row < W.numRows(); row++){
			for (int col = 0; col < W.numCols(); col++){
				SimpleMatrix newWPlus = new SimpleMatrix(W);
				newWPlus.set(row, col, newWPlus.get(row, col) + epsilon);

				SimpleMatrix newWMinus = new SimpleMatrix(W);
				newWMinus.set(row, col, newWMinus.get(row, col) - epsilon);

				double plus = gradientHelper(newX, U, newWPlus, labelVector).get(0, 0);
				double minus = gradientHelper(newX, U, newWMinus, labelVector).get(0, 0);

				double right = (plus - minus) / (2 * epsilon);
				double num = Math.abs(left.get(row, col) - right);

				if (num > Math.pow(10, -7)){
					badCount++;
				}
				else{
					goodCount++;
				}
			}
		}
		System.out.println("GOODCOUNT (Below 10^-7): " + goodCount);
		System.out.println("BADCOUNT: " + badCount);
		System.out.println("");
	}



	//Prediction for the gradient check
	public SimpleMatrix gradientHelper(SimpleMatrix newX, SimpleMatrix tempU, SimpleMatrix tempW, SimpleMatrix labelVector){
		//Forward Propagation
		SimpleMatrix m = tempW.mult(newX);
		SimpleMatrix newM = new SimpleMatrix(HIDDEN_ELEMENTS + 1, 1);
		
		for (int i = 0; i < HIDDEN_ELEMENTS; i++){
			newM.set(i, 0, Math.tanh(m.get(i, 0)));
		}
		newM.set(HIDDEN_ELEMENTS, 0, 1);

		SimpleMatrix finalMatrix = SoftMaxScoreWithLog(tempU.mult(newM));
		return labelVector.transpose().mult(finalMatrix);
	}



	//Prediction for testing
	public SimpleMatrix predict(SimpleMatrix newX){
		//Forward Propagation
		SimpleMatrix m = W.mult(newX);
		SimpleMatrix newM = new SimpleMatrix(HIDDEN_ELEMENTS + 1, 1);
		
		for (int i = 0; i < HIDDEN_ELEMENTS; i++){
			newM.set(i, 0, Math.tanh(m.get(i, 0)));
		}
		newM.set(HIDDEN_ELEMENTS, 0, 1);

		SimpleMatrix finalMatrix = U.mult(newM);
		return SoftMaxScore(finalMatrix);
	}

	//Calculates softmax score of the matrix V with log
	public SimpleMatrix SoftMaxScoreWithLog(SimpleMatrix v){
		double denominator = 0.0;
		for (int i = 0; i < v.numRows(); i++){
			denominator += Math.exp(v.get(i,0));
		}
		SimpleMatrix g = new SimpleMatrix(v.numRows(), v.numCols());
		for (int i = 0; i < v.numRows(); i++){
			g.set(i, 0, -1 * Math.log(Math.exp(v.get(i,0)) / denominator));
		}
		return g;
	}

	//Calculates softmax score of the matrix V
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

	//Calculates the gradient of U
	public SimpleMatrix gradientU(SimpleMatrix p, SimpleMatrix h, SimpleMatrix labelVector){
		SimpleMatrix du = (p.minus(labelVector)).mult(h.transpose());
		return du;
	}

	//Calculates the gradient of W
	public SimpleMatrix gradientW(SimpleMatrix p, SimpleMatrix h, SimpleMatrix x, SimpleMatrix m, SimpleMatrix labelVector){
		SimpleMatrix a = U.extractMatrix(0, NUM_FEATURES, 0, U.numCols() - 1).transpose().mult(p.minus(labelVector));

		SimpleMatrix b = new SimpleMatrix(HIDDEN_ELEMENTS, 1);
		
		for (int i = 0; i < a.numRows() - 1; i++){
			double temp = 1 - Math.pow(Math.tanh(m.get(i, 0)), 2);
			b.set(i, 0, temp);
		}
		SimpleMatrix finalMatrix = new SimpleMatrix(HIDDEN_ELEMENTS, 1);
		
		//Element-wise multiplication
		for (int i = 0; i < a.numRows(); i++){
			finalMatrix.set(a.get(i, 0) * b.get(i, 0));
		}

		return finalMatrix.mult(x.transpose());
	}

	//TODO: Confirm this is correct
	public SimpleMatrix updateL(SimpleMatrix p, SimpleMatrix h, SimpleMatrix m, SimpleMatrix labelVector){
		SimpleMatrix a = U.transpose().mult(p.minus(labelVector));
		SimpleMatrix b = new SimpleMatrix(HIDDEN_ELEMENTS+1, 1);
		
		for (int i = 0; i < a.numRows() - 1; i++){
			double temp = 1 - Math.pow(Math.tanh(m.get(i, 0)), 2);
			b.set(i, 0, temp);
		}

		double temp = 1 - Math.pow(Math.tanh(1), 2);
		b.set(a.numRows()-1, 0, temp);

		SimpleMatrix finalMatrix = new SimpleMatrix(HIDDEN_ELEMENTS+1, 1);
		//Element-wise multiplication
		for (int i = 0; i < a.numRows(); i++){
			finalMatrix.set(a.get(i, 0) * b.get(i, 0));
		}

		SimpleMatrix newW = W.extractMatrix(0, W.numRows(), 0, W.numCols() - 1);
		finalMatrix = finalMatrix.extractMatrix(0, finalMatrix.numRows() - 1, 0, finalMatrix.numCols());

		return newW.transpose().mult(finalMatrix).scale(alpha);
	}

	/**
	* Creates dictionary used for creating the label vector
	*/
	public void createDicts(){
		dict = new HashMap<String, Integer>();
		reverseDict = new HashMap<Integer, String>();
		String[] s = {"O", "LOC", "MISC", "ORG", "PER"};
		int count = 0;
		for (String x : s){
			dict.put(x, count);
			reverseDict.put(count, x);
			count++;
		}
	}

	/**
	 * Simplest SGD training 
	 */
	HashMap<String, Integer> dict; 
	HashMap<Integer, String> reverseDict;
	public void train(List<Datum> _trainData){
		createDicts();
		
		for (int i = WINDOW_SIZE / 2; i < _trainData.size()-(WINDOW_SIZE / 2); i++){
			//Don't check if beginning or end of sentence
			if (_trainData.get(i).equals("<s>") || _trainData.get(i).equals("</s>") ) continue;
			
			// System.out.println("" + i + " / " + (_trainData.size() - (WINDOW_SIZE / 2)) + "done");
			SimpleMatrix newX = new SimpleMatrix(C_N + 1, 1);
			
			int count = 0;
			List<Integer> wordListIndex = new ArrayList<Integer>();

			int startIndex = i - (WINDOW_SIZE / 2);
			for (int j = startIndex; j <= i + (WINDOW_SIZE / 2); j++){
				Datum elem = _trainData.get(j);
				
				//Get word vector
				String word = elem.word.toLowerCase();
				int columnIndex = 0;
				if (FeatureFactory.wordToNum.keySet().contains(word)){
					columnIndex = FeatureFactory.wordToNum.get(word);
				}
				else{
					//Word not in vocabulary
					word = word.replaceAll("\\d", "DG".toLowerCase());
					if (FeatureFactory.wordToNum.keySet().contains(word)){
						columnIndex = FeatureFactory.wordToNum.get(word);
					}
					else{
						columnIndex = FeatureFactory.wordToNum.get("UUUNKKK".toLowerCase()); //TODO: Check that this works, converting to lower case

					}
				}

				wordListIndex.add(columnIndex);
				SimpleMatrix a = FeatureFactory.allVecs.extractMatrix(0, N, columnIndex, columnIndex + 1);

				for (int x = 0; x < N; x++){
					newX.set(count, 0, a.get(x, 0));
					count++;
				}
			}
			//Set final element to be 1
			newX.set(C_N, 0, 1);

			//Get correct label
			SimpleMatrix labelVector = new SimpleMatrix(NUM_FEATURES, 1);
			String label = _trainData.get(i).label;
			labelVector.set(dict.get(label),0, 1);

			feedForwardAndBackward(newX, labelVector, wordListIndex);
		}
	}
	
	public void test(List<Datum> testData){
		for (int i = WINDOW_SIZE / 2; i < testData.size()-(WINDOW_SIZE / 2); i++){
			SimpleMatrix newX = new SimpleMatrix(C_N + 1, 1);
			
			int count = 0;
			int startIndex = i - (WINDOW_SIZE / 2);
			for (int j = startIndex; j < i + (WINDOW_SIZE / 2); j++){
				Datum elem = testData.get(j);
				//Get word vector
				String word = elem.word.toLowerCase();
				int columnIndex = 0;
				if (FeatureFactory.wordToNum.keySet().contains(word)){
					columnIndex = FeatureFactory.wordToNum.get(word);
				}
				else{
					//Word not in vocabulary
					word = word.replaceAll("\\d", "DG".toLowerCase());
					if (FeatureFactory.wordToNum.keySet().contains(word)){
						columnIndex = FeatureFactory.wordToNum.get(word);
					}
					else{
						columnIndex = FeatureFactory.wordToNum.get("UUUNKKK".toLowerCase()); //TODO: Check that this works, converting to lower case

					}
				}
				SimpleMatrix a = FeatureFactory.allVecs.extractMatrix(0, N, columnIndex, columnIndex + 1);

				for (int x = 0; x < N; x++){
					newX.set(count, 0, a.get(x, 0));
					count++;
				}
			}
			//Set final element to be 1
			newX.set(C_N, 0, 1);

			//Get correct label
			SimpleMatrix labelVector = new SimpleMatrix(NUM_FEATURES, 1);
			String word = testData.get(i).word;
			String correctLabel = testData.get(i).label;

			SimpleMatrix prediction = predict(newX);

			int highestIndex = -1;
			double highestScore = -1.0;
			for (int q = 0; q < prediction.numRows(); q++){
				double score = prediction.get(q, 0);
				if (score > highestScore){
					highestScore = score;
					highestIndex = q;
				}
			}
			System.out.println(word + "\t" + correctLabel + "\t" + reverseDict.get(highestIndex));
		}
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
}

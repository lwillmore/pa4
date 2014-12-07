package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;
import java.util.Random;

import java.text.*;

public class WindowModel {
	public static final boolean GRADIENT_CHECK_ON = false;

	protected SimpleMatrix L, W, U, b1, b2;
	public static int HIDDEN_ELEMENTS;
	public static int C_N;
	public static int WINDOW_SIZE;
	public static boolean regularize = false;
	public static int NUM_ITERS;
	public static final int NUM_FEATURES = 5;
	public static final int N = 50;
	public static double alpha;
	public static final double epsilon = Math.pow(10, -4);
	public static double lambda = 0.0;

	public WindowModel(int _windowSize, int _hiddenSize, double _lr, double regularize, int NUM_ITERATIONS) {
		WINDOW_SIZE = _windowSize;
		HIDDEN_ELEMENTS = _hiddenSize;
		alpha = _lr; 
		if (regularize > 0.0){
			this.regularize = true;
			lambda = regularize;
		}
		NUM_ITERS = NUM_ITERATIONS;
	}

	public void initWeights(){
		/* initialize with bias inside as the last column*/
		C_N = N * WINDOW_SIZE;
		double fanIn = C_N;
		double fanOut = HIDDEN_ELEMENTS;
		double e = Math.sqrt(6) / Math.sqrt(fanIn + fanOut);

		/*Random initialization*/
		Random rand = new Random();
		W = SimpleMatrix.random(HIDDEN_ELEMENTS, C_N, -1 * e, e, rand);
		U = SimpleMatrix.random(NUM_FEATURES, HIDDEN_ELEMENTS, -1 * e, 1 * e, rand);
		b1 = new SimpleMatrix(HIDDEN_ELEMENTS, 1);
		b2 = new SimpleMatrix(NUM_FEATURES, 1);
	}

	public void feedForwardAndBackward(SimpleMatrix newX, SimpleMatrix labelVector, List<Integer> wordListIndex){
		/*Forward Propagation*/
		SimpleMatrix m = W.mult(newX).plus(b1);
		SimpleMatrix newM = new SimpleMatrix(HIDDEN_ELEMENTS, 1);

		for (int i = 0; i < HIDDEN_ELEMENTS; i++){
			newM.set(i, 0, Math.tanh(m.get(i, 0)));
		}
		SimpleMatrix finalMatrix = U.mult(newM).plus(b2);
		SimpleMatrix sigmoid = SoftMaxScore(finalMatrix);


		/*Back propagation*/
		SimpleMatrix gradU = gradientU(sigmoid, newM, labelVector);
		SimpleMatrix gradb2 = gradientB2(sigmoid, labelVector);


		SimpleMatrix gradW = gradientW(sigmoid, newX, newM, labelVector);
		SimpleMatrix gradb1 = gradientB1(sigmoid, m, labelVector);
		SimpleMatrix gradL = gradientL(sigmoid, newM, labelVector);

		/*Gradient Checks*/
		if (GRADIENT_CHECK_ON){
			gradientCheckU(gradU, newX, labelVector);
			gradientCheckB2(gradb2, newX, labelVector);
			gradientCheckW(gradW, newX, labelVector); 
			gradientCheckB1(gradb1, newX, labelVector);
			gradientCheckL(gradL, newX, labelVector);
		}

		/*Apply Gradients */
		U = U.minus(gradU.scale(alpha));
		b2 = b2.minus(gradb2.scale(alpha));
		W = W.minus(gradW.scale(alpha));
		b1 = b1.minus(gradb1.scale(alpha));

		/*Update L*/
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

	/*Calculates the gradient of U*/
	public SimpleMatrix gradientU(SimpleMatrix p, SimpleMatrix h, SimpleMatrix labelVector){
		SimpleMatrix du = (p.minus(labelVector)).mult(h.transpose());
		if (regularize){
			return du.plus(U.scale(lambda));
		}
		else{
			return du;
		}
	}

	/*Calculates the gradient of b2 */
	public SimpleMatrix gradientB2(SimpleMatrix p, SimpleMatrix labelVector){
		SimpleMatrix du = p.minus(labelVector);
		return du;
	}

	/*Calculates the gradient of W */
	public SimpleMatrix gradientW(SimpleMatrix p, SimpleMatrix x, SimpleMatrix newM, SimpleMatrix labelVector){
		SimpleMatrix a = U.transpose().mult(p.minus(labelVector));
		SimpleMatrix b = new SimpleMatrix(HIDDEN_ELEMENTS, 1);

		for (int i = 0; i < a.numRows(); i++){
			double temp = 1 - Math.pow(newM.get(i,0),2);
			b.set(i, 0, temp);
		}
		SimpleMatrix finalMatrix = new SimpleMatrix(HIDDEN_ELEMENTS, 1);

		/*Element-wise multiplication */
		for (int i = 0; i < a.numRows(); i++){
			finalMatrix.set(i,0,a.get(i, 0) * b.get(i, 0));
		}

		if (regularize){
			return finalMatrix.mult(x.transpose()).plus(W.scale(lambda));
		}
		else{
			return finalMatrix.mult(x.transpose());
		}
	}

	/*Calculates the gradient of B1 */
	public SimpleMatrix gradientB1(SimpleMatrix p, SimpleMatrix m, SimpleMatrix labelVector){
		SimpleMatrix a = U.transpose().mult(p.minus(labelVector));
		SimpleMatrix b = new SimpleMatrix(HIDDEN_ELEMENTS, 1);

		for (int i = 0; i < b.numRows(); i++){
			double temp = 1 - Math.pow(Math.tanh(m.get(i, 0)), 2);
			b.set(i, 0, temp);
		}
		SimpleMatrix finalMatrix = new SimpleMatrix(HIDDEN_ELEMENTS, 1);

		/*Element-wise multiplication */
		for (int i = 0; i < a.numRows(); i++){
			finalMatrix.set(i,0,a.get(i, 0) * b.get(i, 0));
		}
		return finalMatrix;
	}

	/*Calculate Gradient of L */
	public SimpleMatrix gradientL(SimpleMatrix p, SimpleMatrix newM, SimpleMatrix labelVector){
		SimpleMatrix a = U.transpose().mult(p.minus(labelVector));
		SimpleMatrix b = new SimpleMatrix(HIDDEN_ELEMENTS, 1);

		for (int i = 0; i < a.numRows(); i++){
			double temp = 1 - Math.pow(newM.get(i,0),2);
			b.set(i, 0, temp);
		}
		SimpleMatrix finalMatrix = new SimpleMatrix(HIDDEN_ELEMENTS, 1);

		/*Element-wise multiplication */
		for (int i = 0; i < a.numRows(); i++){
			finalMatrix.set(i,0,a.get(i, 0) * b.get(i, 0));
		}

		return W.transpose().mult(finalMatrix);
	}

	/*Grad Check U*/
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

				double plus = 0.0;
				double minus = 0.0;
				if (regularize){
					plus = gradientHelperRegularized(newX, newUPlus, W, labelVector, b1, b2);
					minus = gradientHelperRegularized(newX, newUMinus, W, labelVector, b1, b2);
				}
				else{
					plus = gradientHelper(newX, newUPlus, W, labelVector, b1, b2).get(0, 0);
					minus = gradientHelper(newX, newUMinus, W, labelVector, b1, b2).get(0, 0);
				}

				double right = (plus - minus) / (2 * epsilon);
				double num = Math.abs(left.get(row, col) - right);

				double threshold = Math.pow(10, -7);
				if (regularize){
					threshold = 5 * Math.pow(10, -7);
				}
				if (num > threshold){
					badCount++;
				}
				else{
					goodCount++;
				}
			}
		}
		System.out.println("U");
		System.out.println("GOODCOUNT: " + goodCount);
		System.out.println("BADCOUNT: " + badCount);
		System.out.println("");
	}

	/*Grad Check B2*/
	public void gradientCheckB2(SimpleMatrix gradB2, SimpleMatrix newX, SimpleMatrix labelVector){
		SimpleMatrix left = gradB2;

		int goodCount = 0;
		int badCount = 0;

		for (int row = 0; row < b2.numRows(); row++){
			for (int col = 0; col < b2.numCols(); col++){
				SimpleMatrix newB2Plus = new SimpleMatrix(b2);
				newB2Plus.set(row, col, newB2Plus.get(row, col) + epsilon);

				SimpleMatrix newB2Minus = new SimpleMatrix(b2);
				newB2Minus.set(row, col, newB2Minus.get(row, col) - epsilon);

				double plus = gradientHelper(newX, U, W, labelVector, b1, newB2Plus).get(0, 0);
				double minus = gradientHelper(newX, U, W, labelVector, b1, newB2Minus).get(0, 0);

				double right = (plus - minus) / (2 * epsilon);
				double num = Math.abs(left.get(row, col) - right);

				double threshold = Math.pow(10, -7);
				if (regularize){
					threshold = Math.pow(10, -6);
				}

				if (num > threshold){
					badCount++;
				}
				else{
					goodCount++;
				}
			}
		}
		System.out.println("B2");
		System.out.println("GOODCOUNT: " + goodCount);
		System.out.println("BADCOUNT: " + badCount);
		System.out.println("");
	}

	/*Grad Check W*/
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

				double plus = 0.0;
				double minus = 0.0;
				if (regularize){
					plus = gradientHelperRegularized(newX, U, newWPlus, labelVector, b1, b2);
					minus = gradientHelperRegularized(newX, U, newWMinus, labelVector, b1, b2);
				}
				else{
					plus = gradientHelper(newX, U, newWPlus, labelVector, b1, b2).get(0, 0);
					minus = gradientHelper(newX, U, newWMinus, labelVector, b1, b2).get(0, 0);
				}

				double right = (plus - minus) / (2 * epsilon);
				double num = Math.abs(left.get(row, col) - right);

				double threshold = Math.pow(10, -7);
				if (regularize){
					threshold = 5 * Math.pow(10, -7);
				}
				if (num > threshold){
					badCount++;
				}
				else{
					goodCount++;
				}
			}
		}
		System.out.println("W");
		System.out.println("GOODCOUNT: " + goodCount);
		System.out.println("BADCOUNT: " + badCount);
		System.out.println("");
	}

	/*Grad Check B1*/
	public void gradientCheckB1(SimpleMatrix gradB1, SimpleMatrix newX, SimpleMatrix labelVector){
		int goodCount = 0;
		int badCount = 0;

		for (int row = 0; row < b1.numRows(); row++){
			for (int col = 0; col < b1.numCols(); col++){
				SimpleMatrix newB1Plus = new SimpleMatrix(b1);
				newB1Plus.set(row, col, newB1Plus.get(row, col) + epsilon);

				SimpleMatrix newB1Minus = new SimpleMatrix(b1);
				newB1Minus.set(row, col, newB1Minus.get(row, col) - epsilon);

				double plus = gradientHelper(newX, U, W, labelVector, newB1Plus, b2).get(0, 0);
				double minus = gradientHelper(newX, U, W, labelVector, newB1Minus, b2).get(0, 0);

				double right = (plus - minus) / (2 * epsilon);
				double num = Math.abs(gradB1.get(row, col) - right);

				if (num > Math.pow(10, -7)){
					badCount++;
				}
				else{
					goodCount++;
				}
			}
		}
		System.out.println("B1");
		System.out.println("GOODCOUNT: " + goodCount);
		System.out.println("BADCOUNT: " + badCount);
		System.out.println("");
	}

	/*Grad Check B1*/
	public void gradientCheckL(SimpleMatrix gradL, SimpleMatrix newX, SimpleMatrix labelVector){
		int goodCount = 0;
		int badCount = 0;

		for (int row = 0; row < newX.numRows(); row++){
			for (int col = 0; col < newX.numCols(); col++){
				SimpleMatrix newXPlus = new SimpleMatrix(newX);
				newXPlus.set(row, col, newXPlus.get(row, col) + epsilon);

				SimpleMatrix newXMinus = new SimpleMatrix(newX);
				newXMinus.set(row, col, newXMinus.get(row, col) - epsilon);

				double plus = gradientHelper(newXPlus, U, W, labelVector, b1, b2).get(0, 0);
				double minus = gradientHelper(newXMinus, U, W, labelVector, b1, b2).get(0, 0);

				double right = (plus - minus) / (2 * epsilon);
				double num = Math.abs(gradL.get(row, col) - right);

				if (num > Math.pow(10, -7)){
					badCount++;
				}
				else{
					goodCount++;
				}
			}
		}
		System.out.println("L");
		System.out.println("GOODCOUNT: " + goodCount);
		System.out.println("BADCOUNT: " + badCount);
		System.out.println("");
	}

	/*Prediction for the gradient check*/
	public SimpleMatrix gradientHelper(SimpleMatrix newX, SimpleMatrix tempU, SimpleMatrix tempW, SimpleMatrix labelVector, SimpleMatrix newB1, SimpleMatrix newB2){
		SimpleMatrix m = tempW.mult(newX).plus(newB1);
		SimpleMatrix newM = new SimpleMatrix(HIDDEN_ELEMENTS, 1);

		for (int i = 0; i < HIDDEN_ELEMENTS; i++){
			newM.set(i, 0, Math.tanh(m.get(i, 0)));
		}

		SimpleMatrix finalMatrix = SoftMaxScoreWithLog(tempU.mult(newM).plus(newB2));
		return labelVector.transpose().mult(finalMatrix);
	}

	/*Prediction for the gradient check (Regularized)*/
	public double gradientHelperRegularized(SimpleMatrix newX, SimpleMatrix tempU, SimpleMatrix tempW, SimpleMatrix labelVector, SimpleMatrix newB1, SimpleMatrix newB2){

		SimpleMatrix m = tempW.mult(newX).plus(newB1);
		SimpleMatrix newM = new SimpleMatrix(HIDDEN_ELEMENTS, 1);

		for (int i = 0; i < HIDDEN_ELEMENTS; i++){
			newM.set(i, 0, Math.tanh(m.get(i, 0)));
		}

		SimpleMatrix finalMatrix = SoftMaxScoreWithLog(tempU.mult(newM).plus(newB2));
		return labelVector.transpose().mult(finalMatrix).get(0, 0) + (lambda/2) * (W.normF() + U.normF());
	}

	/*Calculates softmax score of the matrix V with log*/
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

	/*Calculates softmax score of the matrix V */
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

	/*Creates dictionarys used for creating the label vector */
	HashMap<String, Integer> dict; 
	HashMap<Integer, String> reverseDict;
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

	/*Train method*/
	public void train(List<Datum> _trainData){
		for (int iters = 0; iters < NUM_ITERS; iters++){
			createDicts();

			for (int i = WINDOW_SIZE / 2; i < _trainData.size()-(WINDOW_SIZE / 2); i++){

				if (_trainData.get(i).word.equals("<s>") || _trainData.get(i).word.equals("</s>") ) continue;
				SimpleMatrix newX = new SimpleMatrix(C_N, 1);

				int count = 0;
				List<Integer> wordListIndex = new ArrayList<Integer>();

				int startIndex = i - (WINDOW_SIZE / 2);
				for (int j = startIndex; j <= i + (WINDOW_SIZE / 2); j++){
					Datum elem = _trainData.get(j);


					String word = elem.word.toLowerCase();
					int columnIndex = 0;
					if (FeatureFactory.wordToNum.keySet().contains(word)){
						columnIndex = FeatureFactory.wordToNum.get(word);
					}
					else{
						/*Word not in vocabulary, but may contain digits*/
						word = word.replaceAll("\\d", "DG".toLowerCase());
						if (FeatureFactory.wordToNum.keySet().contains(word)){
							columnIndex = FeatureFactory.wordToNum.get(word);
						}
						else{
							/*Doesn't contain digits or isn't known with replacement*/
							columnIndex = FeatureFactory.wordToNum.get("UUUNKKK".toLowerCase());
						}
					}

					wordListIndex.add(columnIndex);
					SimpleMatrix a = FeatureFactory.allVecs.extractVector(false, columnIndex);

					for (int x = 0; x < N; x++){
						newX.set(count, 0, a.get(x, 0));
						count++;
					}
				}
				SimpleMatrix labelVector = new SimpleMatrix(NUM_FEATURES, 1);
				String label = _trainData.get(i).label;
				labelVector.set(dict.get(label), 0, 1);

				feedForwardAndBackward(newX, labelVector, wordListIndex);
			}
		}
	}
	public void test(List<Datum> testData){
		for (int i = WINDOW_SIZE / 2; i < testData.size()-(WINDOW_SIZE / 2); i++){
			if (testData.get(i).word.equals("<s>") || testData.get(i).word.equals("</s>") ) continue;

			SimpleMatrix newX = new SimpleMatrix(C_N, 1);

			int count = 0;
			List<Integer> wordListIndex = new ArrayList<Integer>();

			int startIndex = i - (WINDOW_SIZE / 2);
			for (int j = startIndex; j <= i + (WINDOW_SIZE / 2); j++){
				Datum elem = testData.get(j);

				String word = elem.word.toLowerCase();
				int columnIndex = 0;
				if (FeatureFactory.wordToNum.keySet().contains(word)){
					columnIndex = FeatureFactory.wordToNum.get(word);
				}
				else{
					/*Word not in vocabulary, but may contain digits*/
					word = word.replaceAll("\\d", "DG".toLowerCase());
					if (FeatureFactory.wordToNum.keySet().contains(word)){
						columnIndex = FeatureFactory.wordToNum.get(word);
					}
					else{
						/*Doesn't contain digits or isn't known with replacement*/
						columnIndex = FeatureFactory.wordToNum.get("UUUNKKK".toLowerCase());
					}
				}

				wordListIndex.add(columnIndex);
				SimpleMatrix a = FeatureFactory.allVecs.extractVector(false, columnIndex);

				for (int x = 0; x < N; x++){
					newX.set(count, 0, a.get(x, 0));
					count++;
				}
			}

			/*Get correct label*/
			SimpleMatrix labelVector = new SimpleMatrix(NUM_FEATURES, 1);
			String correctLabel = testData.get(i).label;
			String word = testData.get(i).word;

			labelVector.set(dict.get(correctLabel), 0, 1);

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

	/*Prediction for test function*/
	public SimpleMatrix predict(SimpleMatrix newX){
		/*Forward Propagation*/
		SimpleMatrix m = W.mult(newX).plus(b1);
		SimpleMatrix newM = new SimpleMatrix(HIDDEN_ELEMENTS, 1);

		for (int i = 0; i < HIDDEN_ELEMENTS; i++){
			newM.set(i, 0, Math.tanh(m.get(i, 0)));
		}

		SimpleMatrix finalMatrix = U.mult(newM).plus(b2);
		return SoftMaxScore(finalMatrix);
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
			/*Object has been seen*/
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
			/*Object hasn't been seen, use O*/
			System.out.println(word + "\t" + label + "\t" + "O");
		}
	}
}
}
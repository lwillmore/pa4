package cs224n.deep;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.ejml.simple.*;


public class FeatureFactory {
	private static final String START_TOKEN = "<s>";
	private static final String END_TOKEN = "</s>";

	private FeatureFactory() {

	}

	static List<Datum> trainData;
	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
		if (trainData==null) trainData = read(filename);
		return trainData;
	}

	static List<Datum> testData;
	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
		if (testData==null) testData= read(filename);
		return testData;
	}


	/*Method modified to include sentence boundaries*/
	private static List<Datum> read(String filename)
	throws FileNotFoundException, IOException {
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));

		for (int i = 0; i <(WindowModel.WINDOW_SIZE/2); i++){
			Datum datum = new Datum(START_TOKEN, "O");
			data.add(datum);
		}

		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				for (int i = 0; i <(WindowModel.WINDOW_SIZE/2); i++){
					Datum datum = new Datum(END_TOKEN, "O");
					data.add(datum);
				}	
				for (int i = 0; i <(WindowModel.WINDOW_SIZE/2); i++){
					Datum datum = new Datum(START_TOKEN, "O");
					data.add(datum);
				}
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
			String label = bits[1];

			Datum datum = new Datum(word, label);
			data.add(datum);
		}
		data.remove(data.size() - 1);
		return data;
	}

	static final int DIMENSIONALITY = 50;
	/*Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V| */
	static SimpleMatrix allVecs; 

	public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
		if (allVecs!=null) return allVecs;

		allVecs = new SimpleMatrix(DIMENSIONALITY, wordToNum.keySet().size());

		BufferedReader br = new BufferedReader(new FileReader(vecFilename));
		String line;
		int colCount = 0;
		while ((line = br.readLine()) != null) {
			String[] arr = line.split(" ");
			for (int i = 0; i < arr.length; i++){
				allVecs.set(i, colCount, Double.parseDouble(arr[i]));
			}
			colCount++;
		}
		return allVecs;
	}

	public static void initRandomWordVectors() throws IOException {
		if (allVecs!=null) return;

		Random rand = new Random();

		double fanIn = WindowModel.C_N;
		double fanOut = WindowModel.HIDDEN_ELEMENTS;
		double e = Math.sqrt(6) / Math.sqrt(fanIn + fanOut);

		allVecs = SimpleMatrix.random(DIMENSIONALITY, wordToNum.keySet().size(), -1 * e, e, rand);
	}

	public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>(); 
	public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

	public static HashMap<String, Integer> initializeVocab(String vocabFilename) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(vocabFilename));
		String line;
		int count = 0;
		while ((line = br.readLine()) != null) {
			wordToNum.put(line.toLowerCase(), count);
			numToWord.put(count, line.toLowerCase());
			count++;
		}
		return wordToNum;
	}
}

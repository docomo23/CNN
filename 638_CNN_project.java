
/**
 * @Author: Yuting Liu and Jude Shavlik.  
 * 
 * Copyright 2017.  Free for educational and basic-research use.
 * 
 * The main class for Lab3 of cs638/838.
 * 
 * Reads in the image files and stores BufferedImage's for every example.  Converts to fixed-length
 * feature vectors (of doubles).  Can use RGB (plus grey-scale) or use grey scale.
 * 
 * You might want to debug and experiment with your Deep ANN code using a separate class, but when you turn in Lab3.java, insert that class here to simplify grading.
 * 
 * Some snippets from Jude's code left in here - feel free to use or discard.
 *
 */

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Hashtable;
import java.util.Random;
import java.util.Vector;
import javax.imageio.ImageIO;



public class Lab3 {

	private static int imageSize = 32; // Images are imageSize x imageSize. The
	// provided data is 128x128, but this
	// can be resized by setting this value
	// (or passing in an argument).
	// You might want to resize to 8x8,
	// 16x16, 32x32, or 64x64; this can
	// reduce your network size and speed up
	// debugging runs.
	// ALL IMAGES IN A TRAINING RUN SHOULD
	// BE THE *SAME* SIZE.
	private static boolean createExtraTrainingExamples = true;
	protected static final double shiftProbNumerator = 6.0; // 6.0 is the
															// 'default.'
	protected static final double probOfKeepingShiftedTrainsetImage = (shiftProbNumerator
			/ 48.0); // This 48 is also embedded elsewhere!
	protected static final boolean perturbPerturbedImages = false;

	private static enum Category {
		airplanes, butterfly, flower, grand_piano, starfish, watch
	}; // We'll hardwire these in, but more robust code would not do so.

	private static final Boolean useRGB = true; // If true, FOUR units are used
	// per pixel: red, green,
	// blue, and grey. If false,
	// only ONE (the grey-scale
	// value).
	private static int unitsPerPixel = (useRGB ? 4 : 1); // If using RGB, use
	// red+blue+green+grey.
	// Otherwise just
	// use the grey
	// value.

	private static String modelToUse = "deep"; // Should be one of {
	// "perceptrons",
	// "oneLayer", "deep" }; You
	// might want to use this if
	// you are trying approaches
	// other than a Deep ANN.
	private static int inputVectorSize; // The provided code uses a 1D vector of
	// input features. You might want to
	// create a 2D version for your Deep ANN
	// code.
	// Or use the get2DfeatureValue()
	// 'accessor function' that maps 2D
	// coordinates into the 1D vector.
	// The last element in this vector holds
	// the 'teacher-provided' label of the
	// example.

	private static double eta = 0.1, fractionOfTrainingToUse = 1,
			dropoutRate = 0.00; // To turn off drop out, set dropoutRate to 0.0
	// (or a neg number).
	private static int maxEpochs = 1000; // Feel free to set to a different
	static Dataset trainsetExtras = new Dataset();

	public static void main(String[] args) {
		String trainDirectory = "images/trainset/";
		String tuneDirectory = "images/tuneset/";
		String testDirectory = "images/testset/";

		if (args.length > 5) {
			System.err.println(
					"Usage error: java Lab3 <train_set_folder_path> <tune_set_folder_path> <test_set_folder_path> <imageSize>");
			System.exit(1);
		}
		if (args.length >= 1) {
			trainDirectory = args[0];
		}
		if (args.length >= 2) {
			tuneDirectory = args[1];
		}
		if (args.length >= 3) {
			testDirectory = args[2];
		}
		if (args.length >= 4) {
			imageSize = Integer.parseInt(args[3]);
		}

		// Here are statements with the absolute path to open images folder
		File trainsetDir = new File(trainDirectory);
		File tunesetDir = new File(tuneDirectory);
		File testsetDir = new File(testDirectory);

		// create three datasets
		Dataset trainset = new Dataset();
		Dataset tuneset = new Dataset();
		Dataset testset = new Dataset();

		// Load in images into datasets.
		long start = System.currentTimeMillis();
		loadDataset(trainset, trainsetDir);
		System.out
		.println(
				"The trainset contains " + comma(trainset.getSize())
				+ " examples.  Took "
				+ convertMillisecondsToTimeSpan(
						System.currentTimeMillis() - start)
				+ ".");

		// ####################### Add shifting #######################
		if (createExtraTrainingExamples) {
			int count_trainsetExtrasKept = 0;

			start = System.currentTimeMillis();

			// Flipping watches will mess up the digits on the watch faces, but that probably is ok.
			for (Instance origTrainImage : trainset.getImages()) {
				createMoreImagesFromThisImage(origTrainImage, 1.00);
			}
			if (perturbPerturbedImages) {
				Dataset copyOfExtras = new Dataset(); // Need (I think) to copy before doing the FOR loop since will add to this!
				for (Instance perturbedTrainImage : trainsetExtras.getImages()) {
					copyOfExtras.add(perturbedTrainImage);
				}
				for (Instance perturbedTrainImage : copyOfExtras.getImages()) {
					createMoreImagesFromThisImage(perturbedTrainImage, 
							((perturbedTrainImage.getProvenance() == Instance.HowCreated.FlippedLeftToRight ||
							perturbedTrainImage.getProvenance() == Instance.HowCreated.FlippedTopToBottom)
									? 3.33  // Increase the odds of perturbing flipped images a bit, since fewer of those.
											: 0.66) // Aim to create about one more perturbed image per originally perturbed image.
							/ (0.5 + 6.0 + shiftProbNumerator)); // The 0.5 is for the chance of flip-flopping. The 6.0 is from rotations.
				}
			}	

			int[] countOfCreatedTrainingImages = new int[Category.values().length];
			for (Instance createdTrainImage : trainsetExtras.getImages()) {
				// Keep more of the less common categories?
				double probOfKeeping = 1.0;

				// Trainset counts: airplanes=127, butterfly=55, flower=114, piano=61, starfish=51, watch=146
				if      ("airplanes".equals(  createdTrainImage.getLabel())) probOfKeeping = 0.66; // No flips, so fewer created.
				else if ("butterfly".equals(  createdTrainImage.getLabel())) probOfKeeping = 1.00; // No top-bottom flips, so fewer created.
				else if ("flower".equals(     createdTrainImage.getLabel())) probOfKeeping = 0.66; // No top-bottom flips, so fewer created.
				else if ("grand_piano".equals(createdTrainImage.getLabel())) probOfKeeping = 1.00; // No flips, so fewer created.
				else if ("starfish".equals(   createdTrainImage.getLabel())) probOfKeeping = 1.00; // No top-bottom flips, so fewer created.
				else if ("watch".equals(      createdTrainImage.getLabel())) probOfKeeping = 0.50; // Already have a lot of these.
				else System.out.println("Unknown label: " + createdTrainImage.getLabel());

				if (random() <= probOfKeeping) {
					countOfCreatedTrainingImages[convertCategoryStringToEnum(createdTrainImage.getLabel()).ordinal()]++;
					count_trainsetExtrasKept++;
					trainset.add(createdTrainImage);//	println("The trainset NOW contains " + comma(trainset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
				}
			}
			for (Category cat : Category.values()) {
				System.out.println(" Kept " + padLeft(comma(countOfCreatedTrainingImages[cat.ordinal()]), 5) + " 'tweaked' images of " + cat + ".");
			}
			System.out.println("Created a total of " + comma(trainsetExtras.getSize()) + " new training examples and kept " + comma(count_trainsetExtrasKept) + ".  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
			System.out.println("The trainset NOW contains " + comma(trainset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");
		}

		// ############## End Shifting ###########################
		start = System.currentTimeMillis();
		loadDataset(tuneset, tunesetDir);
		System.out
		.println(
				"The  testset contains " + comma(tuneset.getSize())
				+ " examples.  Took "
				+ convertMillisecondsToTimeSpan(
						System.currentTimeMillis() - start)
				+ ".");

		start = System.currentTimeMillis();
		loadDataset(testset, testsetDir);
		System.out
		.println(
				"The  tuneset contains " + comma(testset.getSize())
				+ " examples.  Took "
				+ convertMillisecondsToTimeSpan(
						System.currentTimeMillis() - start)
				+ ".");

		// Now train a Deep ANN. You might wish to first use your Lab 2 code
		// here and see how one layer of HUs does. Maybe even try your
		// perceptron code.
		// We are providing code that converts images to feature vectors. Feel
		// free to discard or modify.
		start = System.currentTimeMillis();
		trainANN(trainset, tuneset, testset);
		System.out
		.println("\nTook "
				+ convertMillisecondsToTimeSpan(
						System.currentTimeMillis() - start)
				+ " to train.");

	}


	private static void createMoreImagesFromThisImage(Instance trainImage, double probOfKeeping){
		if (!"airplanes".equals(  trainImage.getLabel()) &&  // Airplanes all 'face' right and up, so don't flip left-to-right or top-to-bottom.
				!"grand_piano".equals(trainImage.getLabel())) {  // Ditto for pianos.

			if (trainImage.getProvenance() != Instance.HowCreated.FlippedLeftToRight && random() <= probOfKeeping) trainsetExtras.add(trainImage.flipImageLeftToRight());

			if (!"butterfly".equals(trainImage.getLabel()) &&  // Butterflies all have the heads at the top, so don't flip to-to-bottom.
					!"flower".equals(   trainImage.getLabel()) &&  // Ditto for flowers.
					!"starfish".equals( trainImage.getLabel())) {  // Star fish are standardized to 'point up.
				if (trainImage.getProvenance() != Instance.HowCreated.FlippedTopToBottom && random() <= probOfKeeping) trainsetExtras.add(trainImage.flipImageTopToBottom());
			}
		}
		boolean rotateImages = true;
		if (rotateImages && trainImage.getProvenance() != Instance.HowCreated.Rotated) {
			//    Instance rotated = origTrainImage.rotateImageThisManyDegrees(3);
			//    origTrainImage.display2D(origTrainImage.getGrayImage());
			//    rotated.display2D(              rotated.getGrayImage()); waitForEnter();

			if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(  3));
			if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees( -3));
			if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(  4));
			if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees( -4));
			if (!"butterfly".equals(trainImage.getLabel()) &&  // Butterflies all have the heads at the top, so don't rotate too much.
					!"flower".equals(   trainImage.getLabel()) &&  // Ditto for flowers and starfish.
					!"starfish".equals( trainImage.getLabel())) {
				if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(  5));
				if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees( -5));    
			} else {
				if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(  2));
				if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees( -2));                            
			}
		}
		// Would be good to also shift and rotate the flipped examples, but more complex code needed.
		if (trainImage.getProvenance() != Instance.HowCreated.Shifted) {
			for (    int shiftX = -3; shiftX <= 3; shiftX++) {
				for (int shiftY = -3; shiftY <= 3; shiftY++) {
					// Only keep some of these, so these don't overwhelm the flipped and rotated examples when down sampling below.
					if ((shiftX != 0 || shiftY != 0) && random() <= probOfKeepingShiftedTrainsetImage * probOfKeeping) trainsetExtras.add(trainImage.shiftImage(shiftX, shiftY));
				}
			}
		}        
	}
	
	public static void loadDataset(Dataset dataset, File dir) {
		for (File file : dir.listFiles()) {
			// check all files
			if (!file.isFile() || !file.getName().endsWith(".jpg")) {
				continue;
			}
			// String path = file.getAbsolutePath();
			BufferedImage img = null, scaledBI = null;
			try {
				// load in all images
				img = ImageIO.read(file);
				// every image's name is in such format:
				// label_image_XXXX(4 digits) though this code could handle more
				// than 4 digits.
				String name = file.getName();
				int locationOfUnderscoreImage = name.indexOf("_image");

				// Resize the image if requested. Any resizing allowed, but
				// should really be one of 8x8, 16x16, 32x32, or 64x64 (original
				// data is 128x128).
				if (imageSize != 128) {
					scaledBI = new BufferedImage(imageSize, imageSize,
							BufferedImage.TYPE_INT_RGB);
					Graphics2D g = scaledBI.createGraphics();
					g.drawImage(img, 0, 0, imageSize, imageSize, null);
					g.dispose();
				}

				Instance instance = new Instance(
						scaledBI == null ? img : scaledBI, name, 
						name.substring(0, locationOfUnderscoreImage));

				dataset.add(instance);
			} catch (IOException e) {
				System.err.println("Error: cannot load in the image file");
				System.exit(1);
			}
		}
	}
	///////////////////////////////////////////////////////////////////////////////////////////////

	private static Category convertCategoryStringToEnum(String name) {
		if ("airplanes".equals(name))
			return Category.airplanes; // Should have been the singular
		// 'airplane' but we'll live with this
		// minor error.
		if ("butterfly".equals(name))
			return Category.butterfly;
		if ("flower".equals(name))
			return Category.flower;
		if ("grand_piano".equals(name))
			return Category.grand_piano;
		if ("starfish".equals(name))
			return Category.starfish;
		if ("watch".equals(name))
			return Category.watch;
		throw new Error("Unknown category: " + name);
	}

	private static double getRandomWeight(int fanin, int fanout,
			boolean isaReLU) { // This is one, 'rule of thumb' for initializing
		// weights.
		double range = Math.max(10 * Double.MIN_VALUE,
				isaReLU ? 2.0 / Math.sqrt(fanin + fanout) // From paper by
						// Glorot & Bengio.
						// See
						// http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
						: 4.0 * Math.sqrt(6.0 / (fanin + fanout))); // See
		// http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
		return (2.0 * random() - 1.0) * range;
	}

	// Map from 2D coordinates (in pixels) to the 1D fixed-length feature
	// vector.
	/*
	 * private static double get2DfeatureValue(Vector<Double> ex, int x, int y,
	 * int offset) { // If only using GREY, then offset = 0; Else offset = // 0
	 * for RED, 1 for GREEN, 2 for BLUE, and 3 for // GREY. return
	 * ex.get(unitsPerPixel * (y * imageSize + x) + offset); // Jude: I // have
	 * // not // used // this, // so // might // need // debugging. }
	 */

	///////////////////////////////////////////////////////////////////////////////////////////////

	// Return the count of TESTSET errors for the chosen model.
	private static int trainANN(Dataset trainset, Dataset tuneset,
			Dataset testset) {
		Instance sampleImage = trainset.getImages().get(0); // Assume there is
		// at least one
		// train image!
		inputVectorSize = sampleImage.getWidth() * sampleImage.getHeight()
				* unitsPerPixel + 1; // The '-1' for the bias is not explicitly
		// added to all examples (instead code
		// should implicitly handle it). The
		// final 1 is for the CATEGORY.

		// For RGB, we use FOUR input units per pixel: red, green, blue, plus
		// grey. Otherwise we only use GREY scale.
		// Pixel values are integers in [0,255], which we convert to a double in
		// [0.0, 1.0].
		// The last item in a feature vector is the CATEGORY, encoded as a
		// double in 0 to the size on the Category enum.
		// We do not explicitly store the '-1' that is used for the bias.
		// Instead code (to be written) will need to implicitly handle that
		// extra feature.
		System.out.println("\nThe input vector size is "
				+ comma(inputVectorSize - 1) + ".\n");

		Vector<Vector<Double>> trainFeatureVectors = new Vector<Vector<Double>>(
				trainset.getSize());
		Vector<Vector<Double>> tuneFeatureVectors = new Vector<Vector<Double>>(
				tuneset.getSize());
		Vector<Vector<Double>> testFeatureVectors = new Vector<Vector<Double>>(
				testset.getSize());

		long start = System.currentTimeMillis();
		fillFeatureVectors(trainFeatureVectors, trainset);
		System.out
				.println(
						"Converted " + trainFeatureVectors.size()
								+ " TRAIN examples to feature vectors. Took "
								+ convertMillisecondsToTimeSpan(
										System.currentTimeMillis() - start)
								+ ".");

		start = System.currentTimeMillis();
		fillFeatureVectors(tuneFeatureVectors, tuneset);
		System.out
				.println(
						"Converted " + tuneFeatureVectors.size()
								+ " TUNE  examples to feature vectors. Took "
								+ convertMillisecondsToTimeSpan(
										System.currentTimeMillis() - start)
								+ ".");

		start = System.currentTimeMillis();
		fillFeatureVectors(testFeatureVectors, testset);
		System.out
				.println(
						"Converted " + testFeatureVectors.size()
								+ " TEST  examples to feature vectors. Took "
								+ convertMillisecondsToTimeSpan(
										System.currentTimeMillis() - start)
								+ ".");

		System.out.println("\nTime to start learning!");

		// Call your Deep ANN here. We recommend you create a separate class
		// file for that during testing and debugging, but before submitting
		// your code cut-and-paste that code here.

		if ("perceptrons".equals(modelToUse))
			return trainPerceptrons(trainFeatureVectors, tuneFeatureVectors,
					testFeatureVectors); // This is optional. Either comment out
		// this line or just right a 'dummy'
		// function.
		else if ("oneLayer".equals(modelToUse))
			return trainOneHU(trainFeatureVectors, tuneFeatureVectors,
					testFeatureVectors); // This is optional. Ditto.
		else if ("deep".equals(modelToUse))
			return trainDeep(trainFeatureVectors, tuneFeatureVectors,
					testFeatureVectors);
		return -1;
	}

	private static void fillFeatureVectors(
			Vector<Vector<Double>> featureVectors, Dataset dataset) {
		for (Instance image : dataset.getImages()) {
			featureVectors.addElement(convertToFeatureVector(image));
		}
	}

	private static Vector<Double> convertToFeatureVector(Instance image) {
		Vector<Double> result = new Vector<Double>(inputVectorSize);

		for (int index = 0; index < inputVectorSize - 1; index++) {
			if (useRGB) {
				int xValue = (index / unitsPerPixel) % image.getWidth();
				int yValue = (index / unitsPerPixel) / image.getWidth();
				// System.out.println(" xValue = " + xValue + " and yValue = " +
				// yValue + " for index = " + index);
				if (index % unitsPerPixel == 0)
					result.add(image.getRedChannel()[xValue][yValue] / 255.0);
				else if (index % unitsPerPixel == 1)
					result.add(image.getGreenChannel()[xValue][yValue] / 255.0);
				else if (index % unitsPerPixel == 2)
					result.add(image.getBlueChannel()[xValue][yValue] / 255.0);
				else
					result.add(image.getGrayImage()[xValue][yValue] / 255.0);
			} else {
				int xValue = index % image.getWidth();
				int yValue = index / image.getWidth();
				result.add(image.getGrayImage()[xValue][yValue] / 255.0);
			}
		}
		result.add((double) convertCategoryStringToEnum(image.getLabel())
				.ordinal()); // The last item is the CATEGORY, representing as
		// an integer starting at 0 (and that int is
		// then coerced to double).

		return result;
	}

	//////////////////// Some utility methods (cut-and-pasted from JWS'
	//////////////////// Utils.java file).
	//////////////////// ///////////////////////////////////////////////////

	private static final long millisecInMinute = 60000;
	private static final long millisecInHour = 60 * millisecInMinute;
	private static final long millisecInDay = 24 * millisecInHour;

	public static String convertMillisecondsToTimeSpan(long millisec) {
		return convertMillisecondsToTimeSpan(millisec, 0);
	}

	public static String convertMillisecondsToTimeSpan(long millisec,
			int digits) {
		if (millisec == 0) {
			return "0 seconds";
		} // Handle these cases this way rather than saying "0 milliseconds."
		if (millisec < 1000) {
			return comma(millisec) + " milliseconds";
		} // Or just comment out these two lines?
		if (millisec > millisecInDay) {
			return comma(millisec / millisecInDay) + " days and "
					+ convertMillisecondsToTimeSpan(millisec % millisecInDay,
							digits);
		}
		if (millisec > millisecInHour) {
			return comma(millisec / millisecInHour) + " hours and "
					+ convertMillisecondsToTimeSpan(millisec % millisecInHour,
							digits);
		}
		if (millisec > millisecInMinute) {
			return comma(millisec / millisecInMinute) + " minutes and "
					+ convertMillisecondsToTimeSpan(millisec % millisecInMinute,
							digits);
		}

		return truncate(millisec / 1000.0, digits) + " seconds";
	}

	public static String comma(int value) { // Always use separators (e.g.,
		// "100,000").
		return String.format("%,d", value);
	}

	public static String comma(long value) { // Always use separators (e.g.,
		// "100,000").
		return String.format("%,d", value);
	}

	public static String comma(double value) { // Always use separators (e.g.,
		// "100,000").
		return String.format("%,f", value);
	}

	public static String padLeft(String value, int width) {
		String spec = "%" + width + "s";
		return String.format(spec, value);
	}

	/**
	 * Format the given floating point number by truncating it to the specified
	 * number of decimal places.
	 * 
	 * @param d
	 *            A number.
	 * @param decimals
	 *            How many decimal places the number should have when displayed.
	 * @return A string containing the given number formatted to the specified
	 *         number of decimal places.
	 */
	public static String truncate(double d, int decimals) {
		double abs = Math.abs(d);
		if (abs > 1e13) {
			return String.format("%." + (decimals + 4) + "g", d);
		} else if (abs > 0 && abs < Math.pow(10, -decimals)) {
			return String.format("%." + decimals + "g", d);
		}
		return String.format("%,." + decimals + "f", d);
	}

	/**
	 * Randomly permute vector in place.
	 *
	 * @param <T>
	 *            Type of vector to permute.
	 * @param vector
	 *            Vector to permute in place.
	 */
	public static <T> void permute(Vector<T> vector) {
		if (vector != null) { // NOTE from JWS (2/2/12): not sure this is an
			// unbiased permute; I prefer (1) assigning
			// random number to each element, (2) sorting,
			// (3) removing random numbers.
			// But also see
			// "http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle" which
			// justifies this.
			/*
			 * To shuffle an array a of n elements (indices 0..n-1): for i from
			 * n - 1 downto 1 do j <- random integer with 0 <= j <= i exchange
			 * a[j] and a[i]
			 */

			for (int i = vector.size() - 1; i >= 1; i--) { // Note from JWS
				// (2/2/12): to
				// match the above I
				// reversed the FOR
				// loop that Trevor
				// wrote, though I
				// don't think it
				// matters.
				int j = random0toNminus1(i + 1);
				if (j != i) {
					T swap = vector.get(i);
					vector.set(i, vector.get(j));
					vector.set(j, swap);
				}
			}
		}
	}

	public static Random randomInstance = new Random();

	/**
	 * @return The next random double.
	 */
	public static double random() {
		return randomInstance.nextDouble();
	}

	/**
	 * @param lower
	 *            The lower end of the interval.
	 * @param upper
	 *            The upper end of the interval. It is not possible for the
	 *            returned random number to equal this number.
	 * @return Returns a random integer in the given interval [lower, upper).
	 */
	public static int randomInInterval(int lower, int upper) {
		return lower + (int) Math.floor(random() * (upper - lower));
	}

	/**
	 * @param upper
	 *            The upper bound on the interval.
	 * @return A random number in the interval [0, upper).
	 */
	public static int random0toNminus1(int upper) {
		return randomInInterval(0, upper);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////// Write
	/////////////////////////////////////////////////////////////////////////////////////////////// your
	/////////////////////////////////////////////////////////////////////////////////////////////// own
	/////////////////////////////////////////////////////////////////////////////////////////////// code
	/////////////////////////////////////////////////////////////////////////////////////////////// below
	/////////////////////////////////////////////////////////////////////////////////////////////// here.
	/////////////////////////////////////////////////////////////////////////////////////////////// Feel
	/////////////////////////////////////////////////////////////////////////////////////////////// free
	/////////////////////////////////////////////////////////////////////////////////////////////// to
	/////////////////////////////////////////////////////////////////////////////////////////////// use
	/////////////////////////////////////////////////////////////////////////////////////////////// or
	/////////////////////////////////////////////////////////////////////////////////////////////// discard
	/////////////////////////////////////////////////////////////////////////////////////////////// what
	/////////////////////////////////////////////////////////////////////////////////////////////// is
	/////////////////////////////////////////////////////////////////////////////////////////////// provided.

	private static int trainPerceptrons(
			Vector<Vector<Double>> trainFeatureVectors,
			Vector<Vector<Double>> tuneFeatureVectors,
			Vector<Vector<Double>> testFeatureVectors) {
		Vector<Vector<Double>> perceptrons = new Vector<Vector<Double>>(
				Category.values().length); // One perceptron per category.

		for (int i = 0; i < Category.values().length; i++) {
			Vector<Double> perceptron = new Vector<Double>(inputVectorSize);
			perceptrons.add(perceptron);
			for (int indexWgt = 0; indexWgt < inputVectorSize; indexWgt++)
				perceptron.add(getRandomWeight(inputVectorSize, 1, false)); // Initialize
			// weights.
		}

		if (fractionOfTrainingToUse < 1.0) { // Randomize list, then get the
			// first N of them.
			int numberToKeep = (int) (fractionOfTrainingToUse
					* trainFeatureVectors.size());
			Vector<Vector<Double>> trainFeatureVectors_temp = new Vector<Vector<Double>>(
					numberToKeep);

			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute,
			// but that is OK.
			for (int i = 0; i < numberToKeep; i++) {
				trainFeatureVectors_temp.add(trainFeatureVectors.get(i));
			}
			trainFeatureVectors = trainFeatureVectors_temp;
		}

		int /*
			 * trainSetErrors = Integer.MAX_VALUE, tuneSetErrors =
			 * Integer.MAX_VALUE,
			 */
		best_tuneSetErrors = Integer.MAX_VALUE,
				/* testSetErrors = Integer.MAX_VALUE, best_epoch = -1, */
				testSetErrorsAtBestTune = Integer.MAX_VALUE, best_epoch = -1;
		long overallStart = System.currentTimeMillis(), start = overallStart;

		for (int epoch = 1; epoch <= maxEpochs /* && trainSetErrors > 0 */; epoch++) {
			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute,
			// but that is OK.

			// CODE NEEDED HERE!

			System.out.println("Done with Epoch # " + comma(epoch) + ".  Took "
					+ convertMillisecondsToTimeSpan(
							System.currentTimeMillis() - start)
					+ " ("
					+ convertMillisecondsToTimeSpan(
							System.currentTimeMillis() - overallStart)
					+ " overall).");
			reportPerceptronConfig(); // Print out some info after epoch, so you
			// can see what experiment is running in
			// a given console.
			start = System.currentTimeMillis();
		}
		System.out
				.println(
						"\n***** Best tuneset errors = "
								+ comma(best_tuneSetErrors) + " of "
								+ comma(tuneFeatureVectors.size()) + " ("
								+ truncate((100.0 * best_tuneSetErrors)
										/ tuneFeatureVectors.size(), 2)
								+ "%) at epoch = " + comma(best_epoch)
								+ " (testset errors = "
								+ comma(testSetErrorsAtBestTune) + " of "
								+ comma(testFeatureVectors.size()) + ", "
								+ truncate((100.0 * testSetErrorsAtBestTune)
										/ testFeatureVectors.size(), 2)
								+ "%).\n");
		return testSetErrorsAtBestTune;
	}

	private static void reportPerceptronConfig() {
		System.out.println("***** PERCEPTRON: UseRGB = " + useRGB
				+ ", imageSize = " + imageSize + "x" + imageSize
				+ ", fraction of training examples used = "
				+ truncate(fractionOfTrainingToUse, 2) + ", eta = "
				+ truncate(eta, 2) + ", dropout rate = "
				+ truncate(dropoutRate, 2));
	}

	//////////////////////////////////////////////////////////////////////////////////////////////// ONE
	//////////////////////////////////////////////////////////////////////////////////////////////// HIDDEN
	//////////////////////////////////////////////////////////////////////////////////////////////// LAYER

	/*
	 * private static boolean debugOneLayer = false; // If set true, more things
	 */
	// checked and/or printed
	// (which does slow down the
	// code).
	private static int numberOfHiddenUnits = 250;

	private static int trainOneHU(Vector<Vector<Double>> trainFeatureVectors,
			Vector<Vector<Double>> tuneFeatureVectors,
			Vector<Vector<Double>> testFeatureVectors) {
		long overallStart = System.currentTimeMillis(), start = overallStart;
		int /*
			 * trainSetErrors = Integer.MAX_VALUE, tuneSetErrors =
			 * Integer.MAX_VALUE,
			 */
		best_tuneSetErrors = Integer.MAX_VALUE,
				/* testSetErrors = Integer.MAX_VALUE, */ best_epoch = -1,
				testSetErrorsAtBestTune = Integer.MAX_VALUE;

		for (int epoch = 1; epoch <= maxEpochs /* && trainSetErrors > 0 */; epoch++) {
			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute,
			// but that is OK.

			// CODE NEEDED HERE!

			System.out.println("Done with Epoch # " + comma(epoch) + ".  Took "
					+ convertMillisecondsToTimeSpan(
							System.currentTimeMillis() - start)
					+ " ("
					+ convertMillisecondsToTimeSpan(
							System.currentTimeMillis() - overallStart)
					+ " overall).");
			reportOneLayerConfig(); // Print out some info after epoch, so you
			// can see what experiment is running in a
			// given console.
			start = System.currentTimeMillis();
		}

		System.out
				.println(
						"\n***** Best tuneset errors = "
								+ comma(best_tuneSetErrors) + " of "
								+ comma(tuneFeatureVectors.size()) + " ("
								+ truncate((100.0 * best_tuneSetErrors)
										/ tuneFeatureVectors.size(), 2)
								+ "%) at epoch = " + comma(best_epoch)
								+ " (testset errors = "
								+ comma(testSetErrorsAtBestTune) + " of "
								+ comma(testFeatureVectors.size()) + ", "
								+ truncate((100.0 * testSetErrorsAtBestTune)
										/ testFeatureVectors.size(), 2)
								+ "%).\n");
		return testSetErrorsAtBestTune;
	}

	private static void reportOneLayerConfig() {
		System.out.println("***** ONE-LAYER: UseRGB = " + useRGB
				+ ", imageSize = " + imageSize + "x" + imageSize
				+ ", fraction of training examples used = "
				+ truncate(fractionOfTrainingToUse, 2) + ", eta = "
				+ truncate(eta, 2) + ", dropout rate = "
				+ truncate(dropoutRate, 2) + ", number HUs = "
				+ numberOfHiddenUnits
		// + ", activationFunctionForHUs = " + activationFunctionForHUs + ",
		// activationFunctionForOutputs = " + activationFunctionForOutputs
		// + ", # forward props = " + comma(forwardPropCounter)
		);
		// for (Category cat : Category.values()) { // Report the output unit
		// biases.
		// int catIndex = cat.ordinal();
		//
		// System.out.print(" bias(" + cat + ") = " +
		// truncate(weightsToOutputUnits[numberOfHiddenUnits][catIndex], 6));
		// } System.out.println();
	}

	// private static long forwardPropCounter = 0; // Count the number of
	// forward propagations performed.

	//////////////////////////////////////////////////////////////////////////////////////////////// DEEP
	//////////////////////////////////////////////////////////////////////////////////////////////// ANN
	//////////////////////////////////////////////////////////////////////////////////////////////// Code
	private static int KERNEL_SIZE = 5;
	private static int POOL_KERNEL_SIZE = 2;
	static int INPUT_SIZE = imageSize * imageSize * unitsPerPixel + 1;
	static int PLATE_NUM = 20;
	static int FLAT_PLATE_NUM = 300;
	static double EPSILON = 1e-4;
	static double LEAKY = 1e-2;
	private static int OUTPUT_SIZE = Category.values().length;
	private static int PATIENCE = 10;
	private static String METHOD = "sigmoid";

	static int C1_PLATE_SIZE = imageSize - KERNEL_SIZE + 1;
	static int P1_PLATE_SIZE = C1_PLATE_SIZE / 2;
	static int C2_PLATE_SIZE = P1_PLATE_SIZE - KERNEL_SIZE + 1;
	static int P2_PLATE_SIZE = C2_PLATE_SIZE / 2;

	private static double sigmoidGradient(double num) {
		return sigmoid(num) * (1 - sigmoid(num));
	}

	private static double sigmoid(double num) {
		return 1 / (Math.pow(Math.E, (-1) * num) + 1.0);
	}

	private static double rectifier(double num) {
		return (num > 0 ? num : 0);
	}

	private static double rectifierGradient(double num) {
		return (num > 0 ? 1 : 0);
	}

	private static double leakyRectifier(double num) {
		return (num > 0 ? num : num * LEAKY);
	}

	private static double leakyRectifierGradient(double num) {
		return (num > 0 ? 1 : LEAKY);
	}

	private static double activate(double num, String method) {
		if (method.equals("sigmoid")) {
			return sigmoid(num);
		} else if (method.equals("rectifier")) {
			return rectifier(num);
		} else if (method.equals("leaky")) {
			return leakyRectifier(num);
		} else {
			return -1.0;
		}
	}

	private static double gradient(double num, String method) {
		if (method.equals("sigmoid")) {
			return sigmoidGradient(num);
		} else if (method.equals("rectifier")) {
			return rectifierGradient(num);
		} else if (method.equals("leaky")) {
			return leakyRectifierGradient(num);
		} else {
			return -1.0;
		}
	}

	private static class Coordinate {
		private int row;
		private int col;

		private Coordinate(int x, int y) {
			this.row = x;
			this.col = y;
		}

		@Override
		public String toString() {
			return '(' + String.valueOf(this.row) + ", "
					+ String.valueOf(this.col) + ')';
		}
	}

	private static class Network {
		// Training info
		private Vector<Vector<Double>> trainFeatureVectors;

		// Weights
		private double[][][] theta1;
		private double[][][][] theta2;
		private double[][][][] theta3;
		private double[][] theta4;
		private double[] theta1Bias;
		private double[] theta2Bias;
		private double[] theta3Bias;
		private double[] theta4Bias;

		// Network structure
		private double[][][] c1;
		private double[][][] p1;
		private double[][][] c2;
		private double[][][] p2;
		private double[] fc;
		private double[] out;

		// Activated layers
		private double[][][] actP1;
		private double[][][] actP2;
		private double[] actFc;
		private double[] actOut;

		// BP structure
		// private double[][] deltaIn;
		private double[][][] deltaC1;
		private double[][][] deltaP1;
		private double[][][] deltaC2;
		private double[][][] deltaP2;
		private double[] deltaFc;
		private double[] deltaOut;

		// BP record
		private boolean[][][] pool1Indeces;
		private boolean[][][] pool2Indeces;

		// #################### Initialization #########################

		private Network(Vector<Vector<Double>> trainFeatureVectors) {

			this.trainFeatureVectors = trainFeatureVectors;
			init();
		}

		private void init() {
			this.c1 = new double[PLATE_NUM][C1_PLATE_SIZE][C1_PLATE_SIZE];
			this.p1 = new double[PLATE_NUM][P1_PLATE_SIZE][P1_PLATE_SIZE];
			this.c2 = new double[PLATE_NUM][C2_PLATE_SIZE][C2_PLATE_SIZE];
			this.p2 = new double[PLATE_NUM][P2_PLATE_SIZE][P2_PLATE_SIZE];
			this.fc = new double[FLAT_PLATE_NUM];
			this.out = new double[OUTPUT_SIZE];

			this.actP1 = new double[PLATE_NUM][P1_PLATE_SIZE][P1_PLATE_SIZE];
			this.actP2 = new double[PLATE_NUM][P2_PLATE_SIZE][P2_PLATE_SIZE];
			this.actFc = new double[FLAT_PLATE_NUM];
			this.actOut = new double[OUTPUT_SIZE];

			this.deltaC1 = new double[PLATE_NUM][C1_PLATE_SIZE][C1_PLATE_SIZE];
			this.deltaP1 = new double[PLATE_NUM][P1_PLATE_SIZE][P1_PLATE_SIZE];
			this.deltaC2 = new double[PLATE_NUM][C2_PLATE_SIZE][C2_PLATE_SIZE];
			this.deltaP2 = new double[PLATE_NUM][P2_PLATE_SIZE][P2_PLATE_SIZE];
			this.deltaFc = new double[FLAT_PLATE_NUM];
			this.deltaOut = new double[OUTPUT_SIZE];

			this.pool1Indeces = new boolean[PLATE_NUM][C1_PLATE_SIZE][C1_PLATE_SIZE];
			this.pool2Indeces = new boolean[PLATE_NUM][C2_PLATE_SIZE][C2_PLATE_SIZE];

			initWeights();
		}

		// All random first
		private void initWeights() {

			this.theta1 = new double[PLATE_NUM][KERNEL_SIZE][KERNEL_SIZE * 4];
			this.theta2 = new double[PLATE_NUM][PLATE_NUM][KERNEL_SIZE][KERNEL_SIZE];
			this.theta3 = new double[FLAT_PLATE_NUM][PLATE_NUM][P2_PLATE_SIZE][P2_PLATE_SIZE];
			this.theta4 = new double[OUTPUT_SIZE][FLAT_PLATE_NUM];

			this.theta1Bias = new double[PLATE_NUM];
			this.theta2Bias = new double[PLATE_NUM];
			this.theta3Bias = new double[FLAT_PLATE_NUM];
			this.theta4Bias = new double[OUTPUT_SIZE];

			for (int i = 0; i < this.theta1.length; i++) {
				for (int j = 0; j < this.theta1[0].length; j++) {
					for (int k = 0; k < this.theta1[0][0].length; k++) {
						this.theta1[i][j][k] = (2.0 * random() - 1.0) * 0.12;
					}
				}
				this.theta1Bias[i] = (2.0 * random() - 1.0) * 0.12;
			}

			for (int i = 0; i < this.theta2.length; i++) {
				for (int j = 0; j < this.theta2[0].length; j++) {
					for (int k = 0; k < this.theta2[0][0].length; k++) {
						for (int l = 0; l < this.theta2[0][0][0].length; l++) {
							this.theta2[i][j][k][l] = (2.0 * random() - 1.0)
									* 0.12;
						}
					}
				}
				this.theta2Bias[i] = (2.0 * random() - 1.0) * 0.12;
			}

			for (int i = 0; i < this.theta3.length; i++) {
				for (int j = 0; j < this.theta3[0].length; j++) {
					for (int k = 0; k < this.theta3[0][0].length; k++) {
						for (int l = 0; l < this.theta3[0][0][0].length; l++) {
							this.theta3[i][j][k][l] = (2.0 * random() - 1.0)
									* 0.12;
						}
					}
				}
				this.theta3Bias[i] = (2.0 * random() - 1.0) * 0.12;
			}

			for (int i = 0; i < this.theta4.length; i++) {
				for (int j = 0; j < this.theta4[0].length; j++) {
					this.theta4[i][j] = (2.0 * random() - 1.0) * 0.12;
				}
				this.theta4Bias[i] = (2.0 * random() - 1.0) * 0.12;
			}
		}

		private void shuffle() {
			Collections.shuffle(this.trainFeatureVectors);
		}

		private void inputToC1(Vector<Double> input,String mode) {
			// Make 2D image from input
			double[][] input2d = new double[imageSize][imageSize * 4];
			double rate = dropoutRate;
			if(mode.equals("train")){
				rate = 0.0;
			}
			// -1 for the calssification
			for (int i = 0; i < input.size() - 1; i++) {
				input2d[i / (imageSize * 4)][i % (imageSize * 4)] = input
						.get(i);
			}
			if(mode.equals("train")&&dropoutRate!=0.0){
				dropoutArray(input2d,dropoutRate);
			}
			// Iterate through all c plates
			for (int t = 0; t < this.c1.length; t++) {
				// Here row and col are the coordinates of start pixel
				for (int row = 0; row <= imageSize - KERNEL_SIZE; row++) {
					int c1Col = 0;
					for (int col = 0; col <= 4
							* (imageSize - KERNEL_SIZE); col += 4) {
						// Here we have the row and col for the kernel
						double total = 0;
						for (int r = 0; r < KERNEL_SIZE; r++) {
							for (int c = 0; c < KERNEL_SIZE * 4; c++) {
								total += input2d[row + r][col + c]
										* this.theta1[t][r][c]*(1.0-rate);
							}
						}
						this.c1[t][row][c1Col] = total + this.theta1Bias[t];
						c1Col++;
					}
				}
			}
		}

		private void pool1(String mode) {
			// Initialize the index book keeping
			this.pool1Indeces = new boolean[PLATE_NUM][C1_PLATE_SIZE][C1_PLATE_SIZE];
			for (int p = 0; p < this.p1.length; p++) {
				Coordinate curP1 = new Coordinate(0, 0);
				// Here row and col are the coordinates of start pixel
				for (int row = 0; row <= C1_PLATE_SIZE
						- POOL_KERNEL_SIZE; row += POOL_KERNEL_SIZE) {
					curP1.col = 0;
					for (int col = 0; col <= C1_PLATE_SIZE
							- POOL_KERNEL_SIZE; col += POOL_KERNEL_SIZE) {

						// Here we have the row and col for the kernel
						double max = -Double.MAX_VALUE;
						Coordinate maxCoor = new Coordinate(-1, -1);

						for (int r = 0; r < POOL_KERNEL_SIZE; r++) {
							for (int c = 0; c < POOL_KERNEL_SIZE; c++) {

								double value = c1[p][row + r][col + c];
								// We want the max value
								if (value > max) {
									max = value;
									maxCoor.row = row + r;
									maxCoor.col = col + c;
								}
							}
						}
						// Update p1 and bookkeeping the max index
						this.p1[p][curP1.row][curP1.col] = max;
						// Activate p1
						this.actP1[p][curP1.row][curP1.col] = activate(max,
								METHOD);
						
						this.pool1Indeces[p][maxCoor.row][maxCoor.col] = true;
						curP1.col++;
					}
					curP1.row++;
				}
			}
			
			if(mode.equals("train")&&dropoutRate!=0.0){
				dropoutArray(this.actP1,dropoutRate);
			}
		}

		private void pool2(String mode) {
			// Initialize the index book keeping
			this.pool2Indeces = new boolean[PLATE_NUM][C2_PLATE_SIZE][C2_PLATE_SIZE];

			for (int p = 0; p < this.p2.length; p++) {
				Coordinate curP2 = new Coordinate(0, 0);
				// Here row and col are the coordinates of start pixel
				for (int row = 0; row <= C2_PLATE_SIZE
						- POOL_KERNEL_SIZE; row += POOL_KERNEL_SIZE) {
					curP2.col = 0;
					for (int col = 0; col <= C2_PLATE_SIZE
							- POOL_KERNEL_SIZE; col += POOL_KERNEL_SIZE) {

						// Here we have the row and col for the kernel
						double max = -Double.MAX_VALUE;
						Coordinate maxCoor = new Coordinate(-1, -1);

						for (int r = 0; r < POOL_KERNEL_SIZE; r++) {
							for (int c = 0; c < POOL_KERNEL_SIZE; c++) {

								double value = c2[p][row + r][col + c];
								// We want the max value
								if (value > max) {
									max = value;
									maxCoor.row = row + r;
									maxCoor.col = col + c;
								}
							}
						}
						// Update p2 and bookkeeping the max index
						this.p2[p][curP2.row][curP2.col] = max;
						this.actP2[p][curP2.row][curP2.col] = activate(max,
								METHOD);
						this.pool2Indeces[p][maxCoor.row][maxCoor.col] = true;
						curP2.col++;
					}
					curP2.row++;
				}
			}
			
			if(mode.equals("train")&&dropoutRate!=0.0){
				dropoutArray(this.actP2,dropoutRate);
			}
		}

		private void p1ToC2(String mode) {
			double rate = dropoutRate;
			if(mode.equals("train")){
				rate = 0.0;
			}
			// Iterate through all c2 plates
			for (int t = 0; t < this.c2.length; t++) {
				// Here row and col are the coordinates of start pixel
				for (int row = 0; row <= P1_PLATE_SIZE - KERNEL_SIZE; row++) {
					for (int col = 0; col <= P1_PLATE_SIZE
							- KERNEL_SIZE; col++) {
						// One plate in c2 connects to all plates in p1
						double total = 0;
						for (int p = 0; p < this.p1.length; p++) {
							// Here we have the row and col for the kernel
							for (int r = 0; r < KERNEL_SIZE; r++) {
								for (int c = 0; c < KERNEL_SIZE; c++) {
									total += this.actP1[p][row + r][col + c]
											* theta2[t][p][r][c]*(1.0-rate);
								}
							}
						}
						this.c2[t][row][col] = total + this.theta2Bias[t];
					}
				}

			}
		}

		private void p2ToFc(String mode) {
			double rate = dropoutRate;
			if(mode.equals("train")){
				rate = 0.0;
			}
			// Iterate through Fc
			for (int f = 0; f < this.fc.length; f++) {
				// Iterate through p2
				double total = 0;
				for (int p = 0; p < this.p2.length; p++) {
					for (int r = 0; r < this.p2[0].length; r++) {
						for (int c = 0; c < this.p2[0][0].length; c++) {
							total += this.actP2[p][r][c]
									* this.theta3[f][p][r][c]*(1.0-rate);
						}
					}
				}
				this.fc[f] = total + this.theta3Bias[f];
				// Activate fc
				this.actFc[f] = activate(this.fc[f], METHOD);
			}
			
			if(mode.equals("train")&&dropoutRate!=0.0){
				dropoutArray(this.actFc,dropoutRate);
			}
		}

		private void fcToOut(String mode) {
			double rate = dropoutRate;
			if(mode.equals("train")){
				rate = 0.0;
			}
			// Iterate through out
			for (int o = 0; o < this.out.length; o++) {
				// Iterate through fc
				double total = 0;
				for (int f = 0; f < this.fc.length; f++) {
					total += fc[f] * theta4[o][f]*(1.0-rate);
				}
				this.out[o] = total + theta4Bias[o];
				// Activate out
				this.actOut[o] = activate(this.out[o], METHOD);
			}
			if(mode.equals("train")&&dropoutRate!=0.0){
				dropoutArray(this.actOut,dropoutRate);
			}
			
		}

		private void forward(Vector<Double> input,String mode) {
			this.inputToC1(input,mode);
			this.pool1(mode);
			this.p1ToC2(mode);
			this.pool2(mode);
			this.p2ToFc(mode);
			this.fcToOut(mode);
			// System.out.println(Arrays.toString(fc));
		}

		private void outToFc(Vector<Double> input) {
			// Get the label vector
			double[] teacher = makeOutput(input.get(input.size() - 1));

			// Compute delta out
			for (int o = 0; o < teacher.length; o++) {
				this.deltaOut[o] = gradient(this.out[o], METHOD)
						* (teacher[o] - this.actOut[o]);
			}

			// BP to fc
			for (int f = 0; f < this.fc.length; f++) {
				double error = 0;
				for (int o = 0; o < this.out.length; o++) {
					error += this.theta4[o][f] * this.deltaOut[o];
				}
				this.deltaFc[f] = error * gradient(this.fc[f], METHOD);
			}
		}

		private void fcToP2() {
			for (int p = 0; p < p2.length; p++) {
				for (int r = 0; r < p2[0].length; r++) {
					for (int c = 0; c < p2[0][0].length; c++) {
						double errorSum = 0;
						for (int f = 0; f < fc.length; f++) {
							errorSum += theta3[f][p][r][c] * deltaFc[f];
						}
						deltaP2[p][r][c] = gradient(p2[p][r][c], METHOD)
								* errorSum;
					}
				}
			}
		}

		private void p2ToC2() {
			// Reinitialize the deltaC2
			this.deltaC2 = new double[PLATE_NUM][C2_PLATE_SIZE][C2_PLATE_SIZE];

			for (int p = 0; p < this.p2.length; p++) {
				Coordinate curP2 = new Coordinate(0, 0);
				// Here row and col are the coordinates of start pixel
				for (int row = 0; row <= C2_PLATE_SIZE
						- POOL_KERNEL_SIZE; row += POOL_KERNEL_SIZE) {
					curP2.col = 0;
					for (int col = 0; col <= C2_PLATE_SIZE
							- POOL_KERNEL_SIZE; col += POOL_KERNEL_SIZE) {
						// Here we have the row and col for the kernel
						for (int r = 0; r < POOL_KERNEL_SIZE; r++) {
							for (int c = 0; c < POOL_KERNEL_SIZE; c++) {
								if (this.pool2Indeces[p][row + r][col + c]) {
									this.deltaC2[p][row + r][col
											+ c] = this.deltaP2[p][curP2.row][curP2.col];
								}
							}
						}
						curP2.col++;
					}
					curP2.row++;
				}
			}
		}

		private void c2ToP1() {
			double[][][] errorSum = new double[PLATE_NUM][p1[0].length][p1[0][0].length];
			// Iterate through all c2 plates
			for (int t = 0; t < this.c2.length; t++) {
				// Here row and col are the coordinates of start pixel
				for (int row = 0; row <= P1_PLATE_SIZE - KERNEL_SIZE; row++) {
					for (int col = 0; col <= P1_PLATE_SIZE
							- KERNEL_SIZE; col++) {
						// One plate in c2 connects to all plates in p1
						for (int p = 0; p < this.p1.length; p++) {
							// Here we have the row and col for the kernel
							for (int r = 0; r < KERNEL_SIZE; r++) {
								for (int c = 0; c < KERNEL_SIZE; c++) {
									errorSum[p][row + r][col
											+ c] += theta2[t][p][r][c]
													* this.deltaC2[t][row][col];

								}
							}
						}
					}
				}
			}
			// Update delta
			for (int p = 0; p < this.p1.length; p++) {
				for (int r = 0; r < this.p1[0].length; r++) {
					for (int c = 0; c < this.p1[0][0].length; c++) {
						this.deltaP1[p][r][c] = gradient(this.p1[p][r][c],
								METHOD) * errorSum[p][r][c];
					}
				}
			}
		}

		private void p1ToC1() {
			this.deltaC1 = new double[PLATE_NUM][C1_PLATE_SIZE][C1_PLATE_SIZE];
			for (int p = 0; p < this.p1.length; p++) {
				Coordinate curP1 = new Coordinate(0, 0);
				// Here row and col are the coordinates of start pixel
				for (int row = 0; row <= C1_PLATE_SIZE
						- POOL_KERNEL_SIZE; row += POOL_KERNEL_SIZE) {
					curP1.col = 0;
					for (int col = 0; col <= C1_PLATE_SIZE
							- POOL_KERNEL_SIZE; col += POOL_KERNEL_SIZE) {

						// Here we have the row and col for the kernel
						for (int r = 0; r < POOL_KERNEL_SIZE; r++) {
							for (int c = 0; c < POOL_KERNEL_SIZE; c++) {
								if (this.pool1Indeces[p][row + r][col + c]) {
									this.deltaC1[p][row + r][col
											+ c] = this.deltaP1[p][curP1.row][curP1.col];
								}
							}
						}
						curP1.col++;
					}
					curP1.row++;
				}
			}
		}

		private void updateTheta1(Vector<Double> input) {
			// Make 2D image from input
			double[][] input2d = new double[imageSize][imageSize * 4];

			// -1 for the calssification
			for (int i = 0; i < input.size() - 1; i++) {
				input2d[i / (imageSize * 4)][i % (imageSize * 4)] = input
						.get(i);
			}

			// TODO
			// double smallEta = eta / (P1_PLATE_SIZE * P1_PLATE_SIZE);
			double smallEta = eta;

			// Iterate through all c plates
			for (int t = 0; t < this.c1.length; t++) {
				// Here row and col are the coordinates of start pixel
				for (int row = 0; row <= imageSize - KERNEL_SIZE; row++) {
					int c1Col = 0;
					for (int col = 0; col <= 4
							* (imageSize - KERNEL_SIZE); col += 4) {
						// Here we have the row and col for the kernel
						for (int r = 0; r < KERNEL_SIZE; r++) {
							for (int c = 0; c < KERNEL_SIZE * 4; c++) {
								this.theta1[t][r][c] += smallEta
										* input2d[row + r][col + c]
										* this.deltaC1[t][row][c1Col];
							}
						}
						this.theta1Bias[t] += smallEta
								* this.deltaC1[t][row][c1Col];
						c1Col++;
					}
				}
			}
		}

		private void updateTheta2() {
			double smallEta = eta;
			// double smallEta = eta / (P2_PLATE_SIZE * P2_PLATE_SIZE);

			// Iterate through all c2 plates
			for (int t = 0; t < this.c2.length; t++) {
				// One plate in c2 connects to all plates in p1

				// Here row and col are the coordinates of start pixel
				for (int row = 0; row <= P1_PLATE_SIZE - KERNEL_SIZE; row++) {
					for (int col = 0; col <= P1_PLATE_SIZE
							- KERNEL_SIZE; col++) {
						for (int p = 0; p < this.p1.length; p++) {
							// Here we have the row and col for the kernel
							for (int r = 0; r < KERNEL_SIZE; r++) {
								for (int c = 0; c < KERNEL_SIZE; c++) {
									theta2[t][p][r][c] += smallEta
											* this.deltaC2[t][row][col]
											* this.actP1[p][row + r][col + c];
								}
							}
						}
						this.theta2Bias[t] += smallEta
								* this.deltaC2[t][row][col];
					}
				}
			}
		}

		private void updateTheta3() {
			// Iterate through Fc
			for (int f = 0; f < this.fc.length; f++) {
				// Iterate through p2
				for (int p = 0; p < this.p2.length; p++) {
					for (int r = 0; r < this.p2[0].length; r++) {
						for (int c = 0; c < this.p2[0][0].length; c++) {
							this.theta3[f][p][r][c] += this.actP2[p][r][c] * eta
									* this.deltaFc[f];
						}
					}
				}
				this.theta3Bias[f] += eta * this.deltaFc[f];
			}
		}

		private void updateTheta4() {
			// Iterate through out
			for (int o = 0; o < this.out.length; o++) {
				// Iterate through fc
				for (int f = 0; f < this.fc.length; f++) {
					theta4[o][f] += fc[f] * eta * this.deltaOut[o];
				}
				this.theta4Bias[o] += eta * this.deltaOut[o];
			}
		}

		private void backward(Vector<Double> input) {
			outToFc(input);
			// gradientCheckTheta4();
			// System.exit(0);
			fcToP2();
			p2ToC2();
			c2ToP1();
			p1ToC1();
			updateTheta1(input);
			updateTheta2();
			updateTheta3();
			updateTheta4();
		}

		private void train() {
			String mode = "train";
			for (int t = 0; t < this.trainFeatureVectors.size(); t++) {
				forward(this.trainFeatureVectors.get(t),mode);
				backward(this.trainFeatureVectors.get(t));
				// System.exit(0);;
				// System.out.println(Arrays.toString(this.theta1[5][0]));
				// System.out.println(Arrays.toString(this.theta1[0][0]));
				/*
				 * double correct = this.trainFeatureVectors.get(t)
				 * .get(this.trainFeatureVectors.get(t).size() - 1);
				 * System.out.println(String.valueOf(predict()) + " : " +
				 * String.valueOf(correct));
				 */
				// if (t == 10){
				// System.exit(0);
				// }
			}
		}

		private double predict() {
			int maxIndex = -1;
			double max = -Double.MAX_VALUE;
			for (int i = 0; i < this.actOut.length; i++) {
				if (this.actOut[i] > max) {
					max = this.actOut[i];
					maxIndex = i;
				}
			}
			return (double) maxIndex;
		}

		public int countError(Vector<Vector<Double>> sets, String... adding) {
			String add = "";
			String mode = "test";
			if (adding.length > 0) {
				add = adding[0];
			}

			int error = 0;
			// Iterate through the feature set
			for (int i = 0; i < sets.size(); i++) {
				double correct = sets.get(i).get(sets.get(i).size() - 1);
				this.forward(sets.get(i),mode);
				double prediction = this.predict();
				// System.out.println(add + String.valueOf(correct) + " : " +
				// String.valueOf(prediction));
				if (correct != prediction) {
					// System.out.println("NO");
					error++;
				}
			}
			return error;
		}

		private void gradientCheckTheta4() {
			double[][] gradient = new double[this.theta4.length][this.theta4[0].length];

			for (int i = 0; i < this.theta4.length; i++) {
				for (int j = 0; j < this.theta4[0].length; j++) {
					gradient[i][j] = /* this.theta4[i][j] + */ this.actFc[j]
							* /* eta * */ this.deltaOut[i];
				}
			}

			int CHECK_TIME = 10;
			double[] diff = new double[CHECK_TIME];
			Random rand = new Random();

			for (int t = 0; t < CHECK_TIME; t++) {
				int row = rand.nextInt(theta4.length);
				int col = rand.nextInt(theta4[0].length);
				int inp = rand.nextInt(this.trainFeatureVectors.size());

				// Add epsilon
				this.theta4[row][col] += EPSILON;
				this.forward(this.trainFeatureVectors.get(inp), "test");
				double firstLoss = this
						.computeLoss(this.trainFeatureVectors.get(inp));

				// Minus epsilon
				this.theta4[row][col] -= 2 * EPSILON;
				this.forward(this.trainFeatureVectors.get(inp), "test");
				double secondLoss = this
						.computeLoss(this.trainFeatureVectors.get(inp));

				// Restore the weight
				this.theta4[row][col] += EPSILON;

				// Compare the result
				double predictGradient = (firstLoss - secondLoss)
						/ (2 * EPSILON);
				double relative = Math.abs(predictGradient - gradient[row][col])
						/ Math.max(
								Math.max(predictGradient, gradient[row][col]),
								EPSILON);

				diff[t] = relative;
			}
			System.out.println(Arrays.toString(diff));
		}

		private double computeLoss(Vector<Double> input) {
			double loss = 0;
			int correctCat = (int) ((double) input.get(input.size() - 1));
			for (Category cat : Category.values()) { // Visit the output units.
				int catIndex = cat.ordinal();
				loss -= (catIndex == correctCat
						? Math.log(this.actOut[catIndex])
						: Math.log(1 - this.actOut[catIndex]));
			}
			return loss;
		}

		private double computeLossSig(Vector<Double> input) {
			double loss = 0;
			double[] correct = makeOutput(input.get(input.size() - 1));
			for (int o = 0; o < correct.length; o++) {
				loss += Math.pow(correct[o] - actOut[o], 2);
			}
			return loss / 2;
		}
	

		private static double[][][] dropoutArray(double[][][] aIn, double dropoutRate) {
			double[][][] aOut = new double[aIn.length][aIn[0].length][aIn[0][0].length];
			for (int i = 0; i < aIn.length; i++) {
				for (int j = 0; j < aIn[i].length; j++) {
					for(int k = 0; k<aIn[i][j].length;k++){
					if (random() > (1.0 - dropoutRate))
						aOut[i][j][k] = 0.0;
					else
						aOut[i][j][k] = aIn[i][j][k];
					}

				}
			}

			return aOut;
		}
		
		private static double[][] dropoutArray(double[][] aIn, double dropoutRate) {
			double[][] aOut = new double[aIn.length][aIn[0].length];
			for (int i = 0; i < aIn.length; i++) {
				for (int j = 0; j < aIn[i].length; j++) {
					if (random() > (1.0 - dropoutRate))
						aOut[i][j] = 0.0;
					else
						aOut[i][j] = aIn[i][j];

				}
			}

			return aOut;
		}
		
		private static double[] dropoutArray(double[] aIn, double dropoutRate) {
			double[] aOut = new double[aIn.length];
			for (int i = 0; i < aIn.length; i++) {
					if (random() > (1.0 - dropoutRate))
						aOut[i] = 0.0;
					else
						aOut[i] = aIn[i];

				}
			

			return aOut;
		}
	
	
	
	}

	private static double[] makeOutput(double category) {
		double[] output = new double[OUTPUT_SIZE];
		// Initialize to category number size
		for (int i = 0; i < OUTPUT_SIZE; i++) {
			output[i] = (int) category == i ? 1.0 : 0.0;
		}
		return output;
	}

	private static String percent(int errorCount, int total) {
		double num = ((double) errorCount) / total;
		double percent = ((int) (num * 10000)) / 100.0;
		return String.valueOf(percent) + "%";
	}

	private static int trainDeep(Vector<Vector<Double>> trainFeatureVectors,
			Vector<Vector<Double>> tuneFeatureVectors,
			Vector<Vector<Double>> testFeatureVectors) {
		//System.out.print(trainFeatureVectors.size());
		//System.exit(0);

		// Choose a portion of training set to use
		if (fractionOfTrainingToUse < 1.0) { // Randomize list, then get the
			// first N of them.
			int numberToKeep = (int) (fractionOfTrainingToUse
					* trainFeatureVectors.size());
			Vector<Vector<Double>> trainFeatureVectors_temp = new Vector<Vector<Double>>(
					numberToKeep);

			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute,
			// but that is OK.
			for (int i = 0; i < numberToKeep; i++) {
				trainFeatureVectors_temp.add(trainFeatureVectors.get(i));
			}
			trainFeatureVectors = trainFeatureVectors_temp;
		}

		// Values for early stopping
		int tuneError = Integer.MAX_VALUE, trainError = Integer.MAX_VALUE,
				testError = Integer.MAX_VALUE,
				bestTuneError = Integer.MAX_VALUE,
				bestTestError = Integer.MAX_VALUE,
				bestTrainError = Integer.MAX_VALUE, bestEpoch = -1,
				localPatience = PATIENCE;
        Object[] bestMatrix = new Object[3];

		// Output the error rates for analysis
		FileWriter fw = null;
		try {
			fw = new FileWriter("out.txt", true);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		BufferedWriter bw = new BufferedWriter(fw);

		Network cnn = new Network(trainFeatureVectors);
		for (int e = 1; e < maxEpochs; e++) {
			cnn.shuffle();
			cnn.train();
			testError = cnn.countError(testFeatureVectors);
			tuneError = cnn.countError(tuneFeatureVectors);
			trainError = cnn.countError(trainFeatureVectors);

			System.out.println(String.valueOf(e) + ": "
					+ String.valueOf(tuneError) + ", "
					+ percent(tuneError, tuneFeatureVectors.size()) + ", "
					+ percent(trainError, trainFeatureVectors.size())
					+ ", test errors: " + String.valueOf(testError));

			// Write to the file
			try {
				bw.write(String.valueOf(e) + ", "
						+ String.valueOf(((double) trainError)
								/ trainFeatureVectors.size())
						+ ", "
						+ String.valueOf(((double) tuneError)
								/ tuneFeatureVectors.size())
						+ ", " + String.valueOf(((double) testError)
								/ testFeatureVectors.size())
						+ "\n");
			} catch (IOException e1) {
				e1.printStackTrace();
			}

			// Record the best error count
			if (tuneError < bestTuneError) {
				bestEpoch = e;
				bestTuneError = tuneError;
				bestTestError = testError;
				bestTrainError = trainError;
				localPatience = PATIENCE;

				System.out.println("Find a better weights at "
						+ String.valueOf(e) + ", train error: "
						+ percent(bestTrainError, trainFeatureVectors.size())
						+ ", tune error: "
						+ percent(bestTuneError, tuneFeatureVectors.size())
						+ ", test error: "
						+ percent(bestTestError, testFeatureVectors.size()));
				bestMatrix = printConfusionMatrix(cnn,testFeatureVectors,"test",e);
			}

			if (localPatience-- == 0) {
				System.out.println("Best weights at "
						+ String.valueOf(bestEpoch) + ", train error: "
						+ percent(bestTrainError, trainFeatureVectors.size())
						+ ", tune error: "
						+ percent(bestTuneError, tuneFeatureVectors.size())
						+ ", test error: "
						+ percent(bestTestError, testFeatureVectors.size()));
                showConfusionMatrix(bestMatrix, bestEpoch);
				break;
			}
		}

		try {
			bw.close();
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return -1;
	}
	
	
	
	public static int findMaxIndex(double[] outputValues) {
		double max = -Double.MAX_VALUE;
		int maxIndex = 0;
		for (int i = 0; i < outputValues.length; i++) {
			if (outputValues[i] > max) {
				max = outputValues[i];
				maxIndex = i;
			}
		}

		return maxIndex;
	}
	
    private static Object[] printConfusionMatrix(Network cnn, Vector<Vector<Double>> tuneFeatureVectors,String mode,int epoch) {
		ArrayList<String> category = new ArrayList<String>(Category.values().length);
		for(Category i : Category.values()){
			category.add(i.name());
		}
		int[][] confusionMatrix = new int[Category.values().length][Category.values().length];
		int[] columnTotal = new int[Category.values().length];
		int[] rowTotal = new int[Category.values().length];

		for (int examp = 0; examp < tuneFeatureVectors.size(); examp++) {
			Vector<Double> example = tuneFeatureVectors.get(examp);
			cnn.forward(example, "test");
			int outputIndex = findMaxIndex(cnn.actOut);
			int exampleIndex = example.get(example.size()-1).intValue();
			confusionMatrix[outputIndex][exampleIndex] ++;
		}
		
		for(int i =0; i<Category.values().length; i++) {
			for(int j = 0; j< Category.values().length; j++) {
				rowTotal[i]+=confusionMatrix[i][j];
				columnTotal[j]+=confusionMatrix[i][j];
			}
		}
		
		System.out.println(epoch+"th epoch "+mode);
		System.out.print("              ");
		for(int i = 0; i<Category.values().length;i++){
			System.out.printf("%14s",category.get(i));
		}
		
		System.out.printf("%14s","pridicted sum");
		System.out.println("");
		for(int i = 0; i<Category.values().length;i++) {
			System.out.printf("%14s",category.get(i));
			for(int j = 0; j<Category.values().length;j++) {
				System.out.printf("%14d",confusionMatrix[i][j]);
			}
			System.out.printf("%14s",rowTotal[i]);
			System.out.println("");
		}
		
		System.out.printf("%14s","actual sum");
		for(int j = 0; j<Category.values().length;j++) {
			System.out.printf("%14s",columnTotal[j]);
		}
		System.out.println("\n\n");
		// return matrix
		return new Object[]{confusionMatrix, columnTotal, rowTotal};
	}
	
	public static void showConfusionMatrix(Object[] matrix, int epoch){
		
		ArrayList<String> category = new ArrayList<String>(Category.values().length);
		for(Category i : Category.values()){
			category.add(i.name());
		}
		
		int[][] confusionMatrix = (int[][]) matrix[0];
		int[] columnTotal = (int[]) matrix[1];
		int[] rowTotal = (int[]) matrix[2];
		
		System.out.println("At best epoch : " + epoch + "th epoch ");
		System.out.print("              ");
		for(int i = 0; i<Category.values().length;i++){
			System.out.printf("%14s",category.get(i));
		}
		
		System.out.printf("%14s","pridicted sum");
		System.out.println("");
		for(int i = 0; i<Category.values().length;i++) {
			System.out.printf("%14s",category.get(i));
			for(int j = 0; j<Category.values().length;j++) {
				System.out.printf("%14d",confusionMatrix[i][j]);
			}
			System.out.printf("%14s",rowTotal[i]);
			System.out.println("");
		}
		
		System.out.printf("%14s","actual sum");
		for(int j = 0; j<Category.values().length;j++) {
			System.out.printf("%14s",columnTotal[j]);
		}
		System.out.println("\n\n");
	}
	////////////////////////////////////////////////////////////////////////////////////////////////
}

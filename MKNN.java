package weka.classifiers.lazy;

import java.util.*;

import weka.classifiers.AbstractClassifier;
import weka.core.*;

/**
 <!-- globalinfo-start -->
 * Modified K-nearest neighbours (MKNN) classifier. Default value of K is 1.
 modified knn with Validty value and Weight Voting.<br/>
 * <br/>
 <!-- globalinfo-end -->
 *
 */

public class MKNN extends AbstractClassifier implements OptionHandler {

    protected Instances Trainset;
    protected int NumClasses;
    protected int ClassAttribute;
    protected int m_knn;
    protected boolean m_knnValid;
    protected double NumAttributesUsed;

    public MKNN() {
        init();
    }

    public String globalInfo() {

        return  "Modified K-nearest neighbours (MKNN) classifier. Can "
                + "default value of K is 3."
                + "modified knn with Validty value and Weight Voting.\n\n";
    }

    /**
     * k the number of neighbours.
     */
    public void setKNN(int k) {
        m_knn = k;
        m_knnValid = false;
    }

    public int getKNN() {

        return m_knn;
    }

    public int getNumTraining() {

        return Trainset.numInstances();
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return  the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    /**
     * Parses a given list of options. <p/>
     *
     <!-- options-start -->
     * Valid options are: <p/>
     *
     * <pre> -V &lt;value of validity;
     *  Validity Value comparing class trainingset with nearest neighbors
     *  (use when k &gt; 1)</pre>
     *
     * <pre> -K &lt;number of neighbors&gt;
     *  Number of nearest neighbours (k) used in classification.
     *  (Default = 3)</pre>
     *
     * <pre> -W &lt;weight voting;
     *  Weight Voting dividing the validity value by the eucledian distance
     *  (validty / (eucledian + 0.5))
     *
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        String knnString = Utils.getOption('K', options);
        if (knnString.length() != 0) {
            setKNN(Integer.parseInt(knnString));
        } else setKNN(3);

        super.setOptions(options);
    }

    public String [] getOptions() {

        Vector<String> options = new Vector<>();
        options.add("-K"); options.add("" + getKNN());

        Collections.addAll(options, super.getOptions());

        return options.toArray(new String[0]);
    }

    public Enumeration<Option> listOptions() {

        Vector<Option> newVector = new Vector<>(7);
        newVector.addElement(new Option(
                "\tValidty Value\n"+
                          "\tValue comparing class trainingset with nearest neighbors ",
                    "V", 1,"-V <Validty Value>"));
        newVector.addElement(new Option(
                "\tNumber of nearest neighbours (k) used in classification.\n"+
                          "\t(Default = 1)",
                    "K", 3,"-K <number of neighbors>"));
        newVector.addElement(new Option(
                "\tvalue of Weight Voting.\n"+
                          "\tDividing validity value with eucledian distance + 0.5)",
                    "W", 1,"-W <Weight Voting>"));

        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }

    public void buildClassifier(Instances data) throws Exception {

        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove data with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        NumClasses = data.numClasses();
        ClassAttribute = data.classAttribute().type();
        Trainset = new Instances(data, 0, data.numInstances());

        NumAttributesUsed = 0.0;
        for (int i = 0; i < Trainset.numAttributes(); i++) {
            if ((i != Trainset.classIndex()) &&
                    (Trainset.attribute(i).isNominal() ||
                            Trainset.attribute(i).isNumeric())) {
                NumAttributesUsed += 1.0;
            }
        }
    }

    /**
     * Initialise scheme variables.
     */
    protected void init() {

        setKNN(1);
    }

    protected void makeDistribution(Instances neighbours, double[] distances) {

        double total = 0;
        double [] distribution = new double [NumClasses];

        // Set up a correction to the estimator
        if (ClassAttribute == Attribute.NOMINAL) {
            for(int i = 0; i <NumClasses; i++) {
                distribution[i] = 1.0 / Math.max(1,Trainset.numInstances());
            }
            total = (double)NumClasses / Math.max(1,Trainset.numInstances());
        }

        for(int i=0; i < neighbours.numInstances(); i++) {
            distances[i] = distances[i]*distances[i];
            distances[i] = Math.sqrt(distances[i]/NumAttributesUsed);

            if (total > 0) {
                Utils.normalize(distribution, total);
            }
        }
    }

    public static void main(String [] argv) {
        runClassifier(new MKNN(), argv);
    }
}
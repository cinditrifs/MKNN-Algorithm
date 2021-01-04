package weka.classifiers.lazy;
import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.core.pmml.jaxbbindings.ClassLabels;

import java.util.Collections;
import java.util.Vector;
import java.util.Set;

public class ModifKNN extends AbstractClassifier implements OptionHandler {
    protected Instances m_Train;
    protected int m_NumInstances;
    protected int m_NumClasses;
    protected int m_NumAttributes;
    protected int m_ClassType;
    protected int m_kNN;

    /** Begin **/
    private double K;
    private double Euclid;
    private double alpha;
    private double validitas;
    private double weight;


    public ModifKNN() {
        alpha=0.5;
        K=3;
    }
    public String globalInfo() {
        return "Class for using Modified KNN algorithm for classification";
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.DATE_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }
    @Override
    public double[][] distributionsForInstances(Instances batch) throws Exception {
        return super.distributionsForInstances(batch);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return super.classifyInstance(instance);
    }

    @Override
    public void setOptions(String[] options) throws Exception {

        String knnString = Utils.getOption('K', options);
        if (knnString.length() != 0) {
            setKNN(Integer.parseInt(knnString));
        } else {
            setKNN(1);
        }
        super.setOptions(options);
    }

    private void setKNN(int parseInt) {
    }

    @Override
    public String[] getOptions() {
        Vector<String> options = new Vector<String>();
        options.add("-K"); options.add("" + getKNN());
        Collections.addAll(options, super.getOptions());

        return options.toArray(new String[0]);
    }

    /** main **/
    public static void main(String [] argv) {
        runClassifier(new ModifKNN(), argv);
    }

    @Override
    public void buildClassifier(Instances weight) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(weight);

        // remove instances with missing class
        weight = new Instances(weight);
        weight.deleteWithMissingClass();

        m_Train = new Instances(weight, 0, weight.numInstances());

        // initializes class attributes ** java-speaking! :-) **
        init_m_Attributes();
    }

    private void init_m_Attributes() {
    }


    /** Distance **/
    public static double getEuclid (Instance instance ) {
        double squareDistance = 0;

        Set<Integer> aFeature = a.getFeatures();
        for (int feature : a.Features) {
            if (b.getFeature(feature) != null)
                squareDistance += Math.pow((a.getFeature(feature) - b.getFeature(feature)), 2);
            else
                squareDistance += Math.pow(a.getFeature(feature), 2);
        }
        Set<Integer> bFeature = b.getFeatures();
        for (int feature : b.Features) {
            if (a.getFeature(feature) == null)
                squareDistance += Math.pow(b.getFeature(feature), 2);
        }
        return Math.sqrt(squareDistance);
    }
    /** Validitas **/
    private double getValiditas([]) {
        validitas = new double[];
        for (int i = 0; i < this.K; i++)
            if (b.getnum_class(m_NumClasses)!= a.getnumclass)
                validitas=0;
            else
                validitas=1;
        return validitas;
    }

    /** Weigth **/
    private void weight(double validitas, double alpha, double Euclid){
        double weight = validitas*(1/Euclid+alpha);
    }
    public double getweight() {
        return weight;
    }


}





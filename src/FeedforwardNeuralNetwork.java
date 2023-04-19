import org.w3c.dom.ls.LSOutput;

import java.util.Random;

public class FeedforwardNeuralNetwork {

    private int inputDim;
    private int hiddenDim;
    private int outputDim;
    private double[][] inputWeights;
    private double[] hiddenBiases;
    private double[][] outputWeights;
    private double[] outputBiases;

    public FeedforwardNeuralNetwork(int inputDim, int hiddenDim, int outputDim){
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        inputWeights = new double[hiddenDim][inputDim];
        hiddenBiases = new double[hiddenDim];
        outputWeights = new double[outputDim][hiddenDim];
        outputBiases = new double[outputDim];
        initializeWeights();
    }

    private void initializeWeights(){
        Random rand = new Random();

        // inicialização dos pesos de cada camada
        inputWeights = new double[hiddenDim][inputDim];
        for(int i = 0; i < hiddenDim; i++){
            for(int j = 0; j < inputDim; j++){
                inputWeights[i][j] = rand.nextDouble() - 0.5;
            }
        }

        //inicialização das biases
        hiddenBiases = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++){
            hiddenBiases[i] = rand.nextDouble() - 0.5;
        }

        // Inicialize os pesos da camada oculta para a camada de saída
        outputWeights = new double[outputDim][hiddenDim];
        for (int i = 0; i < outputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                outputWeights[i][j] = rand.nextDouble() - 0.5; // valores aleatórios no intervalo [-0.5, 0.5]
            }
        }

        // Inicialize os biases da camada de saída
        outputBiases = new double[outputDim];
        for (int i = 0; i < outputDim; i++) {
            outputBiases[i] = rand.nextDouble() - 0.5; // valores aleatórios no intervalo [-0.5, 0.5]
        }
    }

    public double[] forward(double[] input){
        // Calcule a ativação da camada oculta
        double[] hiddenActivations = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            double activation = 0.0;
            for (int j = 0; j < inputDim; j++) {
                activation += inputWeights[i][j] * input[j];
            }
            activation += hiddenBiases[i];
            hiddenActivations[i] = activation;
        }

        // Calcule a ativação da camada de saída
        double[] outputActivations = new double[outputDim];
        for (int i = 0; i < outputDim; i++) {
            double activation = 0.0;
            for (int j = 0; j < hiddenDim; j++) {
                activation += outputWeights[i][j] * sigmoid(hiddenActivations[j]);
            }
            activation += outputBiases[i];
            outputActivations[i] = activation;
        }

        // Aplique a função de ativação final (no caso, a identidade)
        double[] output = new double[outputDim];
        for (int i = 0; i < outputDim; i++) {
            output[i] = outputActivations[i];
        }

        return output;
    }

    // Função sigmóide
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }


    public double[] getNode(){
        int totalParams = inputDim * hiddenDim + hiddenDim + hiddenDim * outputDim + outputDim;
        double[] params = new double[totalParams];

        int index = 0;
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                params[index++] = inputWeights[j][i];
            }
        }

        for (int i = 0; i < hiddenDim; i++) {
            params[index++] = hiddenBiases[i];
        }

        for (int i = 0; i < hiddenDim; i++) {
            for (int j = 0; j < outputDim; j++) {
                params[index++] = outputWeights[j][i];
            }
        }

        for (int i = 0; i < outputDim; i++) {
            params[index++] = outputBiases[i];
        }

        return params;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Input layer size: ").append(inputDim).append("\n");
        sb.append("Hidden layer size: ").append(hiddenDim).append("\n");
        sb.append("Output layer size: ").append(outputDim).append("\n");
        sb.append("\n");

        sb.append("Input weights:\n");
        for (int i = 0; i < inputWeights.length; i++) {
            for (int j = 0; j < inputWeights[i].length; j++) {
                sb.append(inputWeights[i][j]).append(" ");
            }
            sb.append("\n");
        }
        sb.append("\n");

        sb.append("Hidden biases:\n");
        for (int i = 0; i < hiddenBiases.length; i++) {
            sb.append(hiddenBiases[i]).append("\n");
        }
        sb.append("\n");

        sb.append("Output weights:\n");
        for (int i = 0; i < outputWeights.length; i++) {
            for (int j = 0; j < outputWeights[i].length; j++) {
                sb.append(outputWeights[i][j]).append(" ");
            }
            sb.append("\n");
        }
        sb.append("\n");

        sb.append("Output biases:\n");
        for (int i = 0; i < outputBiases.length; i++) {
            sb.append(outputBiases[i]).append("\n");
        }
        sb.append("\n");

        return sb.toString();
    }

    public static void main(String[] args) {
        FeedforwardNeuralNetwork fs = new FeedforwardNeuralNetwork(2, 2, 1);
        System.out.println(fs);


    }
}

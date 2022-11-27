#include <bits/stdc++.h>
using namespace std;

// MLP Classifer trained on iris dataset
// Using sigmoid function as the activation function

// Class 0 === Iris-setosa
// Class 1 === Iris-versicolor
// Class 2 === Iris-virginica

// Returns a random float

#define MAX_EPOCHS 1000
float randomFloat()
{
    return ((rand() % 50) / 100.0);
}

class mlp
{
public:
    int inputLayer = 4;
    int hiddenLayer = 5;
    int outputLayer = 3;
    float learningRate = 0.005;
    int max_epochs = MAX_EPOCHS;
    int biasHiddenValue = -1;
    int biasOutputValue = -1;
    int classesNumber = 3;

    // Using 74% of dataset as training data
    // and rest 26% as test data
    int train_percent = 74;
    int test_percent = 26;

    vector<vector<float>> train_X;
    vector<vector<float>> test_X;
    vector<float> train_Y;
    vector<float> test_Y;

    vector<float> output;
    vector<vector<float>> output_l1;
    vector<vector<float>> output_l2;

    vector<vector<float>> hiddenInputWeight = vector<vector<float>>(4, vector<float>(5, 0));
    vector<vector<float>> hiddenOutputWeight = vector<vector<float>>(5, vector<float>(3, 0));
    vector<vector<float>> hiddenBias = vector<vector<float>>(1, vector<float>(5, -1));
    vector<vector<float>> outputBias = vector<vector<float>>(1, vector<float>(3, -1));
    mlp()
    {
        srand(time(0));
        vector<vector<float>> dataset;
        vector<vector<float>> train;
        vector<vector<float>> test;
        readFromCSV(dataset, "./iris.data");

        // Splitting the dataset for each of the three classes
        // according to the training and test %
        train.insert(train.end(), dataset.begin(), dataset.begin() + 37);
        train.insert(train.end(), dataset.begin() + 50, dataset.begin() + 87);
        train.insert(train.end(), dataset.begin() + 100, dataset.begin() + 137);

        test.insert(test.end(), dataset.begin() + 37, dataset.begin() + 50);
        test.insert(test.end(), dataset.begin() + 87, dataset.begin() + 100);
        test.insert(test.end(), dataset.begin() + 137, dataset.begin() + 150);

        random_shuffle(train.begin(), train.end());
        random_shuffle(test.begin(), test.end());

        for (auto it : train)
        {
            vector<float> x;
            for (int i = 0; i < 4; i++)
            {
                x.push_back(it[i]);
            }
            train_X.push_back(x);
            train_Y.push_back(it.back());
        }
        for (auto it : test)
        {
            vector<float> x;
            for (int i = 0; i < 4; i++)
            {
                x.push_back(it[i]);
            }
            test_X.push_back(x);
            test_Y.push_back(it.back());
        }

        for (auto &i : hiddenInputWeight)
        {
            for (auto &j : i)
            {
                j = randomFloat();
            }
        }

        for (auto &i : hiddenOutputWeight)
        {
            for (auto &j : i)
            {
                j = randomFloat();
            }
        }
    }

    void readFromCSV(vector<vector<float>> &dataset, string filename)
    {
        fstream fin;
        fin.open(filename, ios::in);
        string line, word, temp;
        vector<float> row;
        int ct = 0;
        while (getline(fin, line))
        {
            ct++;
            row.clear();
            stringstream s(line);
            int count = 0;
            while (getline(s, word, ','))
            {
                float num;
                if (count == 4)
                {
                    num = getClassNumberFromClass(word);
                }
                else
                {
                    num = stof(word);
                }
                row.push_back(num);
                count++;
            }
            dataset.push_back(row);
        }
    }

    // Function to multiply two given matrices
    vector<vector<float>> matrix_mul(vector<vector<float>> mat1, vector<vector<float>> mat2)
    {
        int row1 = mat1.size(), col1 = mat1[0].size(), col2 = mat2[0].size();
        vector<vector<float>> ans(row1, vector<float>(col2, 0));
        for (int i = 0; i < row1; i++)
        {
            for (int j = 0; j < col2; j++)
            {
                for (int k = 0; k < col1; k++)
                {
                    ans[i][j] += (mat1[i][k] * mat2[k][j]);
                }
            }
        }
        return ans;
    }
    
    // Function to add two given matrices
    vector<vector<float>> add_matrix(vector<vector<float>> mat1, vector<vector<float>> mat2)
    {
        int row = mat1.size(), col = mat1[0].size();
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
                mat1[i][j] += mat2[i][j];
        }
        return mat1;
    }

    // Function to return the transpose matrix of two given matrices
    vector<vector<float>> transpose_matrix(vector<vector<float>> mat1)
    {
        vector<vector<float>> transp = vector<vector<float>>(mat1[0].size(), vector<float>(mat1.size()));
        int row = mat1.size(), col = mat1[0].size();
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                transp[j][i] = mat1[i][j];
            }
        }
        return transp;
    }

    int getClassNumberFromClass(string word)
    {
        if (word == "Iris-setosa")
        {
            return 0;
        }
        else if (word == "Iris-versicolor")
        {
            return 1;
        }
        else
        {
            return 2;
        }
    }
    
    // Using the sigmoid function as the activation Function
    vector<vector<float>> activationFuncn(vector<vector<float>> x)
    {
        for (int i = 0; i < x.size(); i++)
        {
            for (int j = 0; j < x[0].size(); j++)
            {
                x[i][j] = (1.0 / (1.0 + exp(-x[i][j])));
            }
        }
        return x;
    }
    float derivativeFuncn(float x)
    {
        return (x * (1 - x));
    }

    void predictions()
    {
        vector <vector<float>> forward = matrix_mul(test_X,hiddenInputWeight);
        for(auto x:forward){
            for(int i=0;i<hiddenLayer;i++){
                x[i]+=hiddenBias[0][i];
            }
        }
        forward = matrix_mul(forward, hiddenOutputWeight);
        for (auto x : forward)
        {
            for (int i = 0; i < outputLayer; i++)
            {
                x[i] += outputBias[0][i];
            }
        }
        vector <int> predictions;
        for(int i=0;i<forward.size();i++){
            float mx = max(max(forward[i][0], forward[i][1]), forward[i][2]);
            if (mx == forward[i][0])
            {
                predictions.push_back(0);
            }
            else if (mx == forward[i][1])
            {
                predictions.push_back(1);
            }
            else
            {
                predictions.push_back(2);
            }
        }

        cout << endl << endl;
        float ct = 0;
        cout << "Actual Value  Predicted Value" << endl << endl;
        for (int i = 0; i < test_Y.size(); i++)
        {
            cout.width(12);
            cout << test_Y[i] << "  ";
            cout.width(12);
            cout << predictions[i] << endl;
            if (predictions[i] == test_Y[i])
                ct++;
        }
        cout << endl << "Prediction Accuracy: " << (ct / 39) * 100 << " %" << endl << endl;
    }

    void backpropogation(vector<vector<float>> inputs)
    {
        vector<vector<float>> delta_output;
        vector<float> temp;
        vector<float> error_output;
        for (int i = 0; i < output.size(); i++)
            error_output.push_back(output[i] - output_l2[0][i]);
        for (int i = 0; i < output_l2[0].size(); i++)
        {
            float val = derivativeFuncn(output_l2[0][i]) * error_output[i];
            temp.push_back(-val);
        }
        delta_output.push_back(temp);
        vector<float> arrayStore;
        for (int i = 0; i < hiddenLayer; i++)
        {
            for (int j = 0; j < outputLayer; j++)
            {
                hiddenOutputWeight[i][j] -= (learningRate * (delta_output[0][j] * output_l1[0][i]));
                outputBias[0][j] -= (learningRate * delta_output[0][j]);
            }
        }
        vector<vector<float>> delta_hidden, tem2, mul2 = output_l1;
        tem2 = matrix_mul(delta_output, transpose_matrix(hiddenOutputWeight));
        for (int i = 0; i < output_l1.size(); i++)
        {
            for (int j = 0; j < output_l1[0].size(); j++)
            {
                mul2[i][j] = derivativeFuncn(output_l1[i][j]);
            }
        }
        delta_hidden = mul2;
        for (int i = 0; i < output_l1.size(); i++)
        {
            for (int j = 0; j < output_l1[0].size(); j++)
            {
                delta_hidden[i][j] = mul2[i][j] * tem2[i][j];
            }
        }
        for (int i = 0; i < inputLayer; i++)
        {
            for (int j = 0; j < hiddenLayer; j++)
            {
                hiddenInputWeight[i][j] -= learningRate * (delta_hidden[0][j]) * (inputs[0][i]);
                hiddenBias[0][j] -= learningRate * delta_hidden[0][j];
            }
        }
    }

    void fit()
    {
        int epoch_ct = 1;
        float tot_error = 0;
        int sz = train_X.size();

        vector<float> epoch;
        vector<float> error;
        vector<vector<vector<float>>> w0;
        vector<vector<vector<float>>> w1;

        while (epoch_ct <= max_epochs)
        {
            for (int i = 0; i < sz; i++)
            {
                output = vector<float>(3, 0.0);
                vector<vector<float>> inputs;
                inputs.push_back(train_X[i]);

                output_l1 = activationFuncn(add_matrix(matrix_mul(inputs, hiddenInputWeight), (hiddenBias)));
                output_l2 = activationFuncn(add_matrix(matrix_mul(output_l1, hiddenOutputWeight), (outputBias)));

                if (train_Y[i] == 0)
                {
                    output = {1, 0, 0};
                }
                else if (train_Y[i] == 1)
                {
                    output = {0, 1, 0};
                }
                else
                {
                    output = {0, 0, 1};
                }

                float square_error = 0;
                for (int i = 0; i < outputLayer; i++)
                {
                    float error = (pow(output[i] - output_l2[0][i], 2));
                    square_error += 0.05 * error;
                    tot_error += square_error;
                }
                backpropogation(inputs);
            }
            tot_error /= sz;
            if (epoch_ct % 50 == 0 || epoch_ct == 1)
            {
                cout << "Epoch: " << epoch_ct << " : Total Error: " << tot_error << endl;
                epoch.push_back(epoch_ct);
                error.push_back(tot_error);
            }

            epoch_ct++;
            w0.push_back(hiddenInputWeight);
            w1.push_back(hiddenOutputWeight);
        }
    }
};

int main()
{
    mlp mlpClassifier;
    mlpClassifier.fit();
    mlpClassifier.predictions();
    return 0;
}

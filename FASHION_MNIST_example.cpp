#include "NN_LIB.h"

class MNIST_Data {
    private:

        string clothing_piece(int i){
            vector<string> v = {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};
            return v[i];
        }

    public:
        Data * train_data;
        Data * test_data;
        Model * model;

        void prepare_MNIST(){
            vector<string> leftout = {"path"};
            this->train_data = new Data("Fashion_train.csv","label", leftout);
            this->test_data = new Data("Fashion_test.csv","label", leftout);
            this->model = new Model(this->train_data, true , true, false, false, 1.0, 0.0, 0.0);
            this->test_data->results_one_hot_ecoded_matrix();

            this->train_data->training_data = this->train_data->training_data.return_operatorwhise_calc('+', -255/2);
            this->train_data->training_data = this->train_data->training_data.return_operatorwhise_calc('/', 255);
            this->test_data->total_data = this->test_data->total_data.return_operatorwhise_calc('+', -255/2);
            this->test_data->total_data = this->test_data->total_data.return_operatorwhise_calc('/', 255);
        }

        void predict_image(string path, int a = 0){
            int index = 0;
            for (size_t i = 0; i < this->test_data->left_out[a].data.size(); i++) {
                if (this->test_data->left_out[a].data[i].find(path) != string::npos) {
                    index = i;
                    break;
                }  
            }
            Matrix  inp, y_hat, output; 
            inp.append_row(this->test_data->give_total_data()->pop_row(false, index));
            y_hat.append_row(this->test_data->give_total_resuts()->pop_row(false, index));
            output = this->model->predict(&inp, &y_hat);

            cout << "\nThe testing picture to be predicted: " << path << endl
            << "Predicted Label: " << this->clothing_piece(output.index_max().item()) << " , Actual Label: " << this->clothing_piece(y_hat.index_max().item());
        }

        void delete_MNIST_data(){
            delete this->train_data;
            delete this->test_data;
            delete this->model;
        }
};


int main(){

    MNIST_Data mnist;
    mnist.prepare_MNIST();
    Dense_Layer dense1(784 , 100, 0.0001,0,5e-4,0,5e-4);
    mnist.model->push_sequential(&dense1);
    Activation_ReLU relu;
    mnist.model->push_sequential(&relu);
    Dense_Layer dense2(100,10, 0.0001);
    mnist.model->push_sequential(&dense2);
    Activation_Softmax_Loss_Crossentropy slc;
    mnist.model->push_sequential(&slc);

    Optimizer_Adam opt(0.001, 5e-7);

    mnist.model->train(&opt, 0, 100,10);

    mnist.predict_image("7/0000.png");
    mnist.predict_image("3/0000.png");
    mnist.predict_image("1/0000.png");
    mnist.predict_image("9/0000.png");
}
#include <cstdlib>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <time.h> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <experimental/random>
#include <math.h>
#include <algorithm>
#include <typeinfo>

using namespace std;
class Layer;
class Dense_Layer;

int rand_num(int min ,int max ){   // returns a random number in the interval of min and max
    return std::experimental::randint(min,max);
}

template<class T>
void print_vector_as_row(vector<T> vec){
    cout << "\n";
    for (T s : vec){
        cout << s << " ";
    }
    cout << "\n";
}

class Matrix{   // numpy array style matrix
    
    private:
        
        int rows=0;
        int columns=0;
        vector<vector<double>> vec;
    
    public:
        
        int ret_cols(){
            return this->columns;
        }

        int ret_rows(){
            return this->rows;
        }

        void put_matrix(vector<vector<double>> input_vec){  // pass an entire matrix which the becomes vec
            if (input_vec.size() == 0){
                cout << "\nThis Matrix is empty\n";
                exit(EXIT_FAILURE);
            }
            
            for (size_t a = 0; a < input_vec.size(); a++){
                if (input_vec[0].size() != input_vec[a].size()){
                    cout << "\nThis Matrix is not an even Array\n";
                    exit(EXIT_FAILURE);
                }
            }

            this->rows = input_vec.size();
            this->columns = input_vec[0].size();
            this->vec = input_vec;
        }
        
        void append_row(vector<double> input_vec){        // Append a row
            if (this->rows == 0){
                this->vec.push_back(input_vec);
                this->rows = 1;
                this->columns = input_vec.size();
                return;
            }

            if (input_vec.size() == this->columns){
                this->rows++;
                this->vec.push_back(input_vec);
                return;
            }

            cout << "\nThis row can not be appended due to this Matrix's columns being: " << this->columns << " and the columns of the row to be appended are: " << input_vec.size() << endl;
            exit(EXIT_FAILURE);
        }

        void delete_row(int index){
            if (index > this->rows || this->rows == 0){
                cout << "\nYou are trying to delete row: " << index + 1 <<", however, this matrix has only: " << this->rows << " rows" << endl;
                exit(EXIT_FAILURE);
            }
            this->vec[index].clear();
            this->vec.erase(this->vec.begin() + index, this->vec.begin() + index + 1);
            this->rows--;
        }

        void null_init(int rows, int columns){
            this->rows = rows;
            this->columns = columns;
            for (size_t i = 0; i < rows; i++){
                vector<double> pushed (columns);
                this->vec.push_back(pushed);
            }
        }
        void rand_init(int rows, int columns, float depth=0.5){  // initializes the weights of a rows*columns matrix randomly in a range of depth + (depth * rand_num() * 0.01)
            this->rows = rows;
            this->columns = columns;
            for (size_t i = 0; i < rows; i++){
                vector<double> pushed;
                for (size_t i = 0; i < columns; i++){
                    double d = depth * rand_num(1,10);
                    pushed.push_back(d);
                }
                this->vec.push_back(pushed);
            }
        }
        void print_shape(){
            cout << "Rows: "<< this->rows << ", Columns: "<< this->columns;
        }   
        void print_Matrix(){
            cout << "\n";
            for (vector<double> row : this->vec){
                for (double column_num : row){
                    cout << " " << column_num;
                }
                cout << "\n";
            }
        }
        void custom_Matrix(int rows, int columns){
            this->rows = rows;
            this->columns = columns;
            vector<double> pushed(columns);
            for (size_t a = 0; a < rows; a++){
                for (size_t b = 0; b < columns; b++){
                    cout << "Enter the Element of the matrix in row: " << a  << ", column:" << b << ", here:  ";
                    cin >> pushed[b];
                }
                this->vec.push_back(pushed);
            }
        }

        void Vector(int i){
            this->rows = i;
            this->columns = 1;

            vector<double> intermed_vec;
            intermed_vec.push_back(0);
            for (size_t a = 0; a < i; a++){
                this->vec.push_back(intermed_vec);
            }
            
        }

        void Transpose(){   // Transpose Matrix
            vector<vector<double>> result;
            for (size_t a = 0; a < this->columns; a++){
                vector<double> intermed_vec(this->rows);
                for (size_t b = 0; b < this->rows; b++){
                    intermed_vec[b] = this->vec[b][a];
                }
                result.push_back(intermed_vec);
            }

            for (size_t a = 0; a < this->rows; a++){
                this->vec[a].clear();
            }

            this->vec.clear();
            this->put_matrix(result);
            //this->vec = result;   // you changes this line for the line above, previously the rows and columns did not switch
        }

        double item(){
            if (this->rows != 1 || this->columns != 1){
                cout << "This item is a vector or a matrix";
                exit(EXIT_FAILURE);
            }
            return this->vec[0][0];
        }

        Matrix index_max(){
            int index = 0;
            double max = this->vec[0][0];
            Matrix ret;
            for (size_t a = 0; a < this->rows; a++) {
                vector<double> v = {};
                for (size_t b = 0; b < this->columns; b++) {
                    if (max < this->vec[a][b]){
                        index = b;
                        max = this->vec[a][b];
                    }   
                }
                v.push_back(index); 
                ret.append_row(v);      
            }
            return ret;
        }

        Matrix sum_self(bool total=true, int dim=0){
            Matrix intermed;
            double item = 0;
            if (total == true){ 
                vector<double> d;
                vector<vector<double>> dd;
                for (size_t a = 0; a < this->rows; a++){
                    for (size_t b = 0; b < this->columns; b++){
                        item  += this->vec[a][b]; 
                    }
                }
                d.push_back(item);
                dd.push_back(d);
                intermed.put_matrix(dd);
                return intermed;
            } 
            if (dim < 0 || dim > 1){
                cout << "Summation can only happen with a 2-dimensional Matrix" << endl;
                exit(EXIT_FAILURE);
            }
            if (dim == 0){
                intermed.Vector(this->rows);
                for (size_t a = 0; a < this->rows; a++){
                    for (size_t b = 0;b < this->columns; b++){
                        item += this->vec[a][b];
                    }
                    intermed.vec[a][0] = item;
                    item = 0;
                } 
                return intermed;
            }
            intermed.Vector(this->columns);
            intermed.Transpose();
            for (size_t a = 0; a < this->columns; a++){
                for (size_t b = 0; b < this->rows; b++){
                    item += this->vec[b][a];
                }
                intermed.vec[0][a] = item;
                item = 0;
            }
            return intermed;         
        }

        Matrix operator * (Matrix * m_right){    // Matrix multiplication,              example: Matrix a; Matrix b, the matrix product of a and b is: Matrix c =  a*&b , b is passed as a pointer,hence the & before it 
            if (this->columns != m_right->rows){
                cout << "The shape of the Matrixes are not fit for the multiplication operation, i.e the columns of the left matrix are: " << this->columns << " and the rows of the right matrix are: " << m_right->rows; 
                exit(EXIT_FAILURE);
            }

            Matrix result;
            result.rows = this->rows;
            result.columns = m_right->columns;

            for (size_t b = 0; b < this->rows; b++){
                vector<double> intermed_vec(m_right->columns);
                for (size_t c = 0; c < m_right->columns; c++){
                    for (size_t d = 0; d < m_right->rows; d++){
                        intermed_vec[c] += this->vec[b][d] * m_right->vec[d][c];
                    }   
                }
            result.vec.push_back(intermed_vec);
            }
            return result;
        }

        Matrix operator + (Matrix * m_right){        // Matrix Addition ,  example: Matrix a; Matrix b, the matrix addition of a and b is: Matrix c =  a+&b , b is passed as a pointer,hence the & before it
            if (this->columns != m_right->columns || this->rows != m_right->rows){
                cout << "The shape of the Matrixes are not fit for the addition operation, i.e the columns of the left matrix are: " << this->columns << " and the columns of the right matrix are: " << m_right->columns << " and also the rows of the left matrix are: " << this->rows << " and the rows of the right matrix are: " << m_right->rows << endl;
                exit(EXIT_FAILURE);
            }

            Matrix result;
            result.rows = this->rows;
            result.columns = this->columns;

            for (size_t a = 0; a < this->rows; a++){
                vector<double> intermed_vec(this->columns);
                for (size_t b = 0; b < this->columns; b++){
                    intermed_vec[b] = this->vec[a][b] + m_right->vec[a][b];
                }
                result.vec.push_back(intermed_vec);
            }
            return result;
        }

        
        Matrix operator % (Matrix * bias_vector){  // add biases to each row of inputs*weights matrix
            if (this->columns != bias_vector->columns || bias_vector->rows != 1 ){
                cout << "The shape of the Matrixes are not fit for the row-wise addition operation, i.e the columns of the left matrix are: " << this->columns << " and the columns of the right matrix are: " << bias_vector->columns << " and also the  rows of the bias vector are: " << bias_vector->rows << endl;
                exit(EXIT_FAILURE);
            }

            for (size_t a = 0; a < this->rows; a++){
                for (size_t b = 0; b < this->columns; b++){
                    this->vec[a][b] += bias_vector->vec[0][b];
                } 
            }
            return *this;
        } 

        Matrix operator / (Matrix * dividend_matrix){ // the quotient is divided by the dividend, division happens row-whise
            if(this->equal_proportions(dividend_matrix) == false){
                cout << "Row-whise division can not be made" << endl;
                exit(EXIT_FAILURE);
            }
            Matrix ret_matrix;
            for (size_t a = 0; a < this->rows; a++){
                vector<double> d;
                for (size_t b = 0; b < this->columns; b++){
                    d.push_back(this->vec[a][b]/dividend_matrix->vec[a][b]);
                }
                ret_matrix.append_row(d);
            }
            return ret_matrix;
        }

        Matrix operator += (Matrix  adder){
            if (this->equal_proportions(&adder) == false){
                cout << "Matrix += : The matrix to be added does not have the same proportions as the matrix it is supposed to be added to" << endl;
                exit(EXIT_FAILURE);
            }

            for (size_t a = 0; a < this->rows; a++){
                for (size_t b = 0; b < this->columns; b++){
                    this->vec[a][b] += adder.vec[a][b];
                }   
            }
            return *this;  
        }

        Matrix operator *= (Matrix * multiplier){  // not a matrix multiplication but elementwhise multiplication
            if (this->equal_proportions(multiplier)==false){
                cout << "\nMatrix *= : The Matrixes do not have the same proportions" << endl;
                exit(EXIT_FAILURE);
            }
            
            for (size_t a = 0; a < this->rows; a++){
                for (size_t b = 0; b < this->columns; b++){
                    this->vec[a][b] *= multiplier->vec[a][b];
                } 
            }
            return *this;
        }

        double max_or_min(bool maximum=true){  // get maximum value of matrix
            double max = 0;
            double min = 0;
            if (maximum == false){
                min = this->max_or_min(true);
            }
            double container;
            for (size_t a = 0; a < this->rows; a++){
                for (size_t b = 0; b < this->columns; b++){
                    container = abs(this->vec[a][b]);
                    if (max < container){
                        max = container;
                    } 
                    if (min > container){
                        min = container;
                    }
                }
            } 
            if (maximum == false){
                return min;
            }
            return max;
        }

        double make_Even(){                  // divide every number in matrix by maximum of matrix
            double max = this->max_or_min(true);
            for (size_t a = 0; a < this->rows; a++){
                for (size_t b = 0; b < this->columns; b++){
                    this->vec[a][b] /= max;
                }
            } 
            return max;
        }

        bool equal_proportions(Matrix * input){
            if (this->rows == input->rows && this->columns == input->columns){
                return true;
            } 
            return false;
        }

        bool contains_element(double pot_element){
            for (size_t a = 0; a < this->rows; a++){
                for (size_t b = 0; b < this->columns; b++){
                    if (this->vec[a][b] == pot_element) {
                        return true;
                    } 
                }
            }
            return false;
        }

        void empty(){
            for (size_t a = 0; a < this->rows; a++) {
                this->vec[a].clear();
            }
            this->vec.clear();
            this->rows = 0;
            this->columns = 0;
        }

        bool is_one_hot_encoded(){
            if (this->columns == 1){
                return false;
            }
            int num_ones = 0;
            for (size_t a = 0; a < this->rows; a++){
                for (size_t b = 0; b < this->columns; b++){
                    if (this->vec[a][b]== 0 || this->vec[a][b]==1){
                        if (this->vec[a][b]==1){
                            if (num_ones > 0){
                                return false;
                            }
                            num_ones++;
                        }
                    } 
                    else{
                        return false;
                    }
                }
                if (num_ones != 1){
                    return false;
                } 
                num_ones = 0;
            }
            return true;
        }

        Matrix return_encoded_one_hot(){
            if (this->columns != 1){
                cout << "\nMatrix can not be one hot encoded since it is not a vector." << endl;
                exit(EXIT_FAILURE); 
            }
            int col_indx = 0;
            double max = this->max_or_min(true);
            Matrix one_hot;
            for (size_t a = 0; a < this->rows; a++){
                vector<double> indxs(max+1);
                col_indx = this->vec[a][0];
                indxs[col_indx] =  1;
                one_hot.append_row(indxs);
            }
            return one_hot;
        }

        Matrix mean(bool rowwhise=true){
            if (rowwhise == true){
                return this->sum_self(false, 0).return_operatorwhise_calc('/', this->columns);
            } else {
                return this->sum_self(false, 1).return_operatorwhise_calc('/', this->rows);
            }
        }

        Matrix stddev(Matrix * input, bool rowwhise=true){
            Matrix ret;
            double num = 0;
            if (rowwhise == true) {
                ret = input->mean(true);
                for (size_t a = 0; a < input->rows; a++){
                    num = 0;
                    for (size_t b = 0; b < input->columns; b++){
                        num += pow(input->vec[a][b]-ret.vec[a][0],2);
                    }
                    ret.vec[a][0] = num/input->columns;
                }   
            } else {
                ret = input->mean(false);
                for (size_t a = 0; a < input->columns; a++) {
                    num = 0;
                    for (size_t b = 0; b < input->rows; b++) {
                        num += pow(input->vec[b][a]-ret.vec[0][a],2);
                    }
                    ret.vec[0][a] = num/input->rows;
                }
            }
            return ret.return_operatorwhise_calc('r', 2);
        }

        Matrix return_operatorwhise_calc(char opera, double val=1){
            vector<char> ops = {'+','-','/','*', '^','r', '|'};
            int op = -1;
            for (size_t a = 0; a < ops.size(); a++){
                if (opera == ops[a]){
                    op = a;    
                }
            }
            Matrix ret_matrix = *this;
            for (size_t a = 0; a < ret_matrix.rows; a++){
                for (size_t b = 0; b < ret_matrix.columns; b++){
                    switch (op){
                    case 0:
                        ret_matrix.vec[a][b] += val;
                        break;
                    case 1:
                        ret_matrix.vec[a][b] -= val;
                        break;
                    case 2:
                        ret_matrix.vec[a][b] /= val;
                        break;
                    case 3:
                        ret_matrix.vec[a][b] *= val;
                        break;
                    case 4:
                        ret_matrix.vec[a][b] = pow(this->vec[a][b],val);
                        break;
                    case 5:
                        ret_matrix.vec[a][b] = pow(this->vec[a][b],1/val);
                        break;
                    case 6:
                        ret_matrix.vec[a][b] = abs(ret_matrix.vec[a][b]);
                        break;
                    default:
                        cout << "calculation operation is not defined" << endl;
                        exit(EXIT_FAILURE);
                        break;
                    }
                }
            }
            return ret_matrix;
        }

        void clip(double ceil=1., double divide=10.){
            for (size_t a = 0; a < this->rows; a++){
                for (size_t b = 0; b < this->columns; b++){
                    if (this->vec[a][b] > ceil){
                        this->vec[a][b] /= divide;
                    }  
                }
            }
        }

        void round_after_comma(int comma){
            double quotient = pow(10.0, comma);
            for (size_t a = 0; a < this->rows; a++){
                for (size_t b = 0; b < this->columns; b++){
                    this->vec[a][b] = round(this->vec[a][b]*quotient) / quotient;
                }
            }       
        }

        vector<double> pop_row(bool del = true, int i = 0){
            vector<double> ret = this->vec[i];
            if (del==true){
                this->rows--;
                this->vec.erase(this->vec.begin() + i, this->vec.begin()+ i + 1);
            }
            return ret;
        }
        
        friend class Dense_Layer;
        friend class Data;
        friend class Activation_ReLU;
        friend class Activation_Softmax;    
        friend class Loss;
        friend class Loss_Categorical_Cross_Entropy;
        friend class Activation_Softmax_Loss_Crossentropy;
        friend class Optimizer_AdaGrad;
        friend class Optimizer_RMSprop;
        friend class Optimizer_Adam;
        friend class Dropout_Layer;
        friend class Activation_Sigmoid;
        friend class Loss_BinaryCrossentropy;
        friend class Loss_Mean_Squared_Error;
        friend class Loss_Mean_Absolute_Error;
        friend double accuracy(Matrix * output, Matrix * results, bool categorization, bool binary, double regression_divisor);
        
        
        
        
        friend int main();
};

class left_outer {
            public:

                string label;
                int index;
                vector<string> data;

                left_outer(string lab, int ind) : label{lab}, index{ind} {}

                void show(string name, int i=-1){
                    if (i >= 0 ){
                        if (i >= this->data.size()){
                            cout << "\nLeftout data is out of bound" << endl;
                            exit(EXIT_FAILURE);
                        }
                        cout << this->data[i];
                        return;
                    }
                    cout << endl;
                    for (string v : this->data){
                        cout << v << endl;
                    }
                }
        };

class Data {
    
    private:
        
        ifstream data_file;

        template<class T>
        bool target_in_labels(T target , vector<T> labels){
            int occurence = 0;
            for (T s : labels){
                if(s == target) occurence++;
            }
            if (occurence != 1){
                return false;
            }
            return true;
        }
        
        template<class T>
        int find_index(T num ,vector<T> vec){
            for (size_t a = 0; a < vec.size(); a++){
                if (vec[a] == num){
                    return a;
                } 
            }
            return -1;
        }

        

        bool left_outer_list(int i , vector<left_outer>  vec){
            for (left_outer l : vec) {
                if(i == l.index) return true;
            }
            return false;
        }

    public:

        Matrix total_results;
        Matrix total_data;
        
        string dataset_name;
        vector<string> labels;
        vector<double> label_maxes;      //  max value per label - use for regression
        double data_max;                 //  max value of total dataset - use for categorization
        vector<double> results_encoded;
        
        Matrix testing_data;
        Matrix validation_data;
        Matrix training_data;
        
        Matrix testing_results;        
        Matrix validation_results;
        Matrix training_results;

        int batch_size = 0;
        bool batch_iterate = false;
        Matrix data_batch_slicer;
        Matrix result_batch_slicer;

        vector<left_outer> left_out = {};

        int ret_num_classes(){
            return labels.size();
        }

        Matrix * give_total_data(){
            return &this->total_data;
        }

        Matrix * give_total_resuts(){
            return &this->total_results;
        }
       
        Data(string file_name, string target_label , vector<string> leave_out={}){  // takes a big dataset and puts it into numbers      //  vector<string> * leave_out is a string vector of names of variables (being columns of the matrix of data needing to be taken out before)
            this->dataset_name = file_name;
            this->data_file.open(this->dataset_name);
            vector<double> result;
            string line = "";
            int target_index = 0;

            for (size_t a = 0;!this->data_file.eof(); a++){
                getline(this->data_file, line);
                stringstream lineStream(line);
                string cell;
                for (size_t b = 0; getline(lineStream, cell, ',') ; b++) {
                    if (a == 0){
                        this->labels.push_back(cell);
                    } else{
                        if(b == target_index) {
                            vector<double> intermed;
                            intermed.push_back(stod(cell));
                            this->total_results.append_row(intermed);
                            intermed.clear();
                        } else if(this->left_outer_list(b, this->left_out)){
                            for (size_t i = 0; i < this->left_out.size(); i++) {
                                if (this->left_out[i].index == b) {
                                    this->left_out[i].data.push_back(cell);
                                }   
                            }
                        } else {
                            result.push_back(stod(cell));
                        }
                    }
                }
                
                if (a == 0){ 
                    if (target_in_labels<string>(target_label, this->labels) == false ){
                        cout << "\nData_reading: The given dependent variable is not among the labels." << endl;
                        exit(EXIT_FAILURE);
                    }  
                    target_index = this->find_index<string>(target_label, this->labels);
                    this->labels.erase(this->labels.begin() + target_index , this->labels.begin() + target_index + 1);
                    for (string str : leave_out){
                        if (target_in_labels<string>(str , this->labels) == false ){
                            cout << "\nData_reading: One of the to be taken out variables are not contained in the datasheet in the first place" << endl;
                            exit(EXIT_FAILURE);
                        }
                        left_outer leftee(str,this->find_index<string>(str, this->labels));
                        this->left_out.push_back(leftee);
                        this->labels.erase(this->labels.begin()+leftee.index, this->labels.begin()+leftee.index+1);
                    }
                } else {
                    this->total_data.append_row(result);
                    result.clear();
                }

                if (!lineStream && cell.empty()){
                    break;
                }
            }
        }

        Data(vector<vector<double>> input_vec , vector<string> labels, string target_label){  // pass in labels and according matrix which becomes after the cropping of the target labels is split into testing, validation and training data 
            if (input_vec.size() == 0 || input_vec[0].size() == 0 || input_vec[0].size() != labels.size() || target_in_labels<string>(target_label, labels) == false){
                cout << "\nThe matrix proportions do not fit, recheck the matrixes and the labels. Either the matrix is empty, the columns of the input matrix do not match the ones of the label vector or the target label is not in the target vector" << endl;
                exit(EXIT_FAILURE);
            }
            size_t target_index = 0;
            for (; target_index < labels.size(); target_index++){  // get index of the column of dependent variable
                if (labels[target_index] == target_label)break;
            }

            this->total_results.Vector(input_vec.size());

            for (size_t a = 0; a < input_vec.size(); a++){
                this->total_results.vec[a][0] = input_vec[a][target_index];
                input_vec[a].erase(input_vec[a].begin()+target_index, input_vec[a].begin() + target_index +1);
            }

            this->total_data.put_matrix(input_vec);
            labels.erase(labels.begin() + target_index , labels.begin() + target_index + 1);
            this->labels = labels;
        }

        void results_one_hot_ecoded_matrix(){  
            vector<double> categories;
            for (size_t a = 0; a < this->total_results.rows; a++){
                if (target_in_labels<double>(this->total_results.vec[a][0] , categories)==false){
                    categories.push_back(this->total_results.vec[a][0]);
                }
            }

            sort(categories.begin(), categories.end());
            this->results_encoded = categories;

            Matrix intermed_results;
            intermed_results.null_init(total_results.rows ,categories.size());

            for (size_t a = 0; a < this->total_results.rows; a++){
                intermed_results.vec[a][this->find_index<double>(this->total_results.vec[a][0],categories)] = 1;
            }
            this->total_results.empty();
            this->total_results.put_matrix(intermed_results.vec);
            intermed_results.empty();
        }

        void max_standardization(bool columnwhise=true, bool result_normalize=false){
            if (columnwhise == false){
                this->data_max = this->total_data.make_Even();
                if (result_normalize == true){
                    for (size_t a = 0; a < this->total_results.rows; a++){
                        for (size_t b = 0; b < this->total_results.columns; b++){
                            this->total_results.vec[a][b] /= this->data_max;
                        }
                    }
                }
                 return; 
            }
            
            for (size_t i = 0; i < this->ret_num_classes(); i++) this->label_maxes.push_back(1e-7);
            for (size_t a = 0; a < this->total_data.rows; a++){   // iterate over columns
                for (size_t b = 0; b < this->total_data.columns; b++){
                    if (this->label_maxes[b] < abs(this->total_data.vec[a][b])) this->label_maxes[b] = abs(this->total_data.vec[a][b]); 
                }
                if (result_normalize==true && this->label_maxes[this->label_maxes.size()-1] < this->total_results.vec[a][0]){
                    this->label_maxes[this->label_maxes.size()-1] = this->total_results.vec[a][0];
                }    
            }
            for (size_t a = 0; a < this->total_data.rows; a++){   // iterate over columns
                for (size_t b = 0; b < this->total_data.columns; b++){
                    this->total_data.vec[a][b] /= this->label_maxes[b];
                }
                if (result_normalize == true) this->total_results.vec[a][0] /= this->label_maxes[this->label_maxes.size()-1];
            }
            
        }

        void data_random_shuffle(){        // shuffle total_data and total results randomly (corresponding to each other, obviously)
            Matrix intermed_total_data;
            Matrix intermed_total_results;
            int index = 0;

            int ceiling = this->total_data.rows;
            for (size_t a = 0; a < ceiling; a++){
                index = rand_num(0,this->total_data.vec.size() -1);
                intermed_total_data.append_row(this->total_data.vec[index]);
                intermed_total_results.append_row(this->total_results.vec[index]);
                this->total_data.delete_row(index);
                this->total_results.delete_row(index);
            }
            this->total_results = intermed_total_results;
            this->total_data = intermed_total_data;
            intermed_total_data.empty();
            intermed_total_results.empty();
        }

        void dataset_distribution(double training_proportion=0.8, double validation_proportion=0.1, double testing_proportion=0.1){  // distribute the total data and total results over training, validation and test, data based on the inputs, respectively.
            if ((training_proportion + validation_proportion + testing_proportion) != 1.0){
                cout << "\nThe proportions do not add up to 1" << endl;
                exit(EXIT_FAILURE);
            }
            int train_cases = round(training_proportion *this->total_data.rows); 
            int validation_cases = round(validation_proportion *this->total_data.rows);
            int test_cases = round(testing_proportion *this->total_data.rows);

            int ceiling = this->total_data.rows;
            for (size_t a = 0; a < ceiling; a++){

                if ( a < train_cases){
                    this->training_data.append_row(this->total_data.vec[0]);
                    this->training_results.append_row(this->total_results.vec[0]);
                    this->total_data.delete_row(0);
                    this->total_results.delete_row(0);
                    
                } else if(train_cases <= a && a < train_cases + validation_cases){
                    this->validation_data.append_row(this->total_data.vec[0]);
                    this->validation_results.append_row(this->total_results.vec[0]);
                    this->total_data.delete_row(0);
                    this->total_results.delete_row(0);
                } else {
                    break;
                }
            }
            this->testing_data = this->total_data;
            this->testing_results = this->total_results;
            this->total_data.empty();
            this->total_results.empty();
        }

        Matrix batch_return(bool data , Matrix * data_set=NULL, Matrix * result_set=NULL, int batchsize=0){
            if (this->batch_iterate == false && data_set != NULL){
                this->batch_iterate = true;
                if (batchsize > 0){
                    this->batch_size = batchsize;
                } else {
                    this->batch_size = data_set->rows;
                }
                this->data_batch_slicer = *data_set;
                this->result_batch_slicer = *result_set;
            } 
            Matrix batch;
            for (size_t a = 0; a < this->batch_size; a++){
                if (data == true){
                    batch.append_row(this->data_batch_slicer.pop_row());
                    if (this->data_batch_slicer.rows == 0) break;   
                } else {
                    batch.append_row(this->result_batch_slicer.pop_row());
                    if (this->result_batch_slicer.rows == 0) {
                        this->batch_iterate = false;
                        break;
                        }
                }
            }
            return batch;
        }
};

class Super_Layer {

    private:

        bool dense = false;
        bool activation_softmax_lc = false;
        bool categorization = true;
        bool binary = false;
        bool regression = false;
        double regression_divisor = 250;

    public:
        
        Matrix output;
        Matrix dinputs;

        virtual void forward(Matrix * input){};
        virtual void forward(Matrix * input, Matrix * y_hat){};
        virtual void backward(Matrix * dvalues){};
        virtual void backward(Matrix * dvalues, Matrix * y_true){};
        virtual double calculate(vector<Dense_Layer *> * layers = NULL){return 0;};

        friend class Dense_Layer;
        friend class Model;
        friend class Loss_BinaryCrossentropy;
        friend class Loss_Mean_Squared_Error;
        friend class Loss_Mean_Absolute_Error;
        friend class Activation_Softmax_Loss_Crossentropy;
};


class Layer : public Super_Layer{

    private:
        
        Matrix weights;
        Matrix biases;
        Matrix inputs; 

    public:
        
        Matrix dweights;
        Matrix dbiases;
        Matrix weight_momentums;
        Matrix bias_momentums;
        Matrix weight_cache;
        Matrix bias_cache;

        double weight_regularizer_l1;
        double weight_regularizer_l2;
        double bias_regularizer_l1;
        double bias_regularizer_l2;
    
    explicit Layer(int n_inputs, int n_neurons, float weight_magnitude=0.5, double weight_regularizer_l1=0, double weight_regularizer_l2=0, double bias_regularizer_l1=0, double bias_regularizer_l2=0){   // initialize layer
            
        this->weights.rand_init(n_inputs, n_neurons, weight_magnitude);
        this->biases.null_init(1,n_neurons);
            
        this->weight_regularizer_l1 = weight_regularizer_l1;
        this->weight_regularizer_l2 = weight_regularizer_l2;
        this->bias_regularizer_l1 = bias_regularizer_l1;
        this->bias_regularizer_l2 = bias_regularizer_l2;
            }

    void clip_weigts(double ceil=1., double divide=10){
        dweights.clip(ceil, divide);
        dbiases.clip(ceil, divide);
    }

    virtual void forward(Matrix * input) override {}; 
    virtual void forward(Matrix * input, Matrix * y_hat) override {};
    virtual void backward(Matrix * dvalues) override {};
    
    friend class Dense_Layer;
    friend class Optimizer_SGD;
    friend class Optimizer_AdaGrad;
    friend class Optimizer_RMSprop;
    friend class Optimizer_Adam;
    friend class Loss;
   
   
    friend int main();
};


class Dense_Layer : public Layer {
    
    private:

        Matrix ones_l1(bool weights=true, double regular = 1.){
            Matrix * ptr;
            if (weights == true){
                ptr = &this->weights;
            } else {
                ptr = &this->biases;
            }
            Matrix ret;
            for (size_t a = 0; a < ptr->rows; a++){
                vector<double> intermed;
                for (size_t b = 0; b < ptr->columns; b++){
                    if (ptr->vec[a][b] > 0) {
                        intermed.push_back(regular);
                    } else {
                        intermed.push_back((-1.)*regular);
                    }
                }
                ret.append_row(intermed);
            }
            return ret;
        }
           
    public:
        
        explicit Dense_Layer(int n_inputs, int n_neurons, float weight_magnitude=0.5, double weight_regularizer_l1=0, double weight_regularizer_l2=0, double bias_regularizer_l1=0, double bias_regularizer_l2=0) 
                            : Layer(n_inputs, n_neurons, weight_magnitude , weight_regularizer_l1, weight_regularizer_l2, bias_regularizer_l1, bias_regularizer_l2) 
                            {
                                this->dense = true;
                            }


        void forward(Matrix * input) override {      // forward the layer, being multiply the input with the already transposed weights
            this->inputs = *input;
            this->output = this->inputs*&(this->weights);
            this->output = this->output%&(this->biases);
        }

        void backward(Matrix * dvalues) override {
            this->inputs.Transpose();
            this->dweights = this->inputs*dvalues;
            this->inputs.Transpose();
            this->dbiases = dvalues->sum_self(false, 1); // initial code

            if (this->weight_regularizer_l1 > 0){
                this->dweights += this->ones_l1(true, this->weight_regularizer_l1);
            }
            if (this->weight_regularizer_l2 >  0){
                this->dweights += this->weights.return_operatorwhise_calc('*', (2*this->weight_regularizer_l2));
            }
            if (this->bias_regularizer_l1 > 0){
                this->dbiases += this->ones_l1(false, this->bias_regularizer_l1);
            }
            if (this->bias_regularizer_l2 > 0){
                this->dbiases += this->dbiases.return_operatorwhise_calc('*', (this->bias_regularizer_l2));
            }
            
            this->weights.Transpose();
            this->dinputs = (*dvalues)*(&this->weights);
            this->weights.Transpose();
        }

        Matrix layer_weights(){      // return the output after the forwarding
            return this->weights;
        }

        Matrix layer_biases(){      // return the output after the forwarding
            return this->biases;
        }
};

class Dropout_Layer : public Layer {
    
    private:
    
        double dropout_rate=0;
        Matrix inputs;
        Matrix mask;

        Matrix binary_mask(){
            Matrix intermed;
            double total = (double)this->inputs.columns;
            for (size_t a = 0; a < this->inputs.rows; a++){
                double zeros = 0;
                vector<double> vec(this->inputs.columns, 1);
                for (size_t b = 0; b < this->inputs.columns; b++){
                    vec[b] /= this->dropout_rate;
                }
                while (zeros/total < this->dropout_rate){
                    int null_indx = rand_num(0, this->inputs.columns-1);
                    vec[null_indx] = 0;
                    zeros = (double)count(vec.begin(), vec.end(), 0);
                }  
                intermed.append_row(vec);
            } 
            return intermed;   
        }

    public:

        explicit Dropout_Layer (double dropout_rate=0.5, int n_inputs=0, int n_neurons=0, float weight_magnitude=0, double weight_regularizer_l1=0, double weight_regularizer_l2=0, double bias_regularizer_l1=0, double bias_regularizer_l2=0) 
                                : Layer( n_inputs,  n_neurons,  weight_magnitude,  weight_regularizer_l1, weight_regularizer_l2, bias_regularizer_l1, bias_regularizer_l2)
                                {
                                  this->dropout_rate = dropout_rate; 
                                }

        void forward(Matrix * inputs) override {
            this->inputs = *inputs;
            this->mask = this->binary_mask();
            this->output = this->mask;
            this->output *= inputs;
        }

        void backward(Matrix * dvalues) override {
            this->dinputs = this->mask;
            this->dinputs *= dvalues;
        }

        friend int main();
};


class Activation_ReLU : public Super_Layer{  // Rectified Linear Unit Class
    
    private:
        Matrix inputs;

        void maximum(double comparee){   // check for every number in matrix input_layer if its bigger than comparee , if not set to comparee
            for (size_t a = 0; a < this->output.rows; a++){
                for (size_t b = 0; b < this->output.columns; b++){
                    if (this->output.vec[a][b] < comparee){
                        this->output.vec[a][b] = comparee;
                    } 
                }
            } 
        }
 
    public:

        void forward(Matrix * passed_inputs) override {        // Take all inputs and set to zero if negative
            this->inputs = *passed_inputs;
            this->output = this->inputs;
            maximum(0);
        }

        void backward(Matrix * dvalues) override {
            this->dinputs = *dvalues;
            for (size_t a = 0; a < this->dinputs.rows; a++){
                for (size_t b = 0; b < this->dinputs.columns; b++){
                    if (this->inputs.vec[a][b] <= 0){
                        this->dinputs.vec[a][b] = 0;
                    }
                }
            }
        }
};
 

class Activation_Softmax : public Super_Layer {// Softmax activation class

    private:

        Matrix inputs;

        void softmax(){
            double addition = 1e-7;
            for (size_t a = 0; a < this->output.rows; a++){
                addition = 1e-7;
                for (size_t b = 0; b < this->output.columns; b++){
                    this->output.vec[a][b] = exp(this->output.vec[a][b]);
                } 

                for (size_t b = 0; b < this->output.columns; b++){
                    addition += this->output.vec[a][b]; 
                }
                
                for (size_t b = 0; b < this->output.columns; b++){
                    this->output.vec[a][b] /= addition; 
                }
            }
        }

        void normalize(){
            double max = 0;
            for (size_t a = 0; a < this->inputs.rows; a++){
                for (size_t b = 0; b < this->inputs.columns; b++){
                    if (max < this->inputs.vec[a][b]){
                        max = this->inputs.vec[a][b];
                    }
                } 

                for (size_t b = 0; b < this->inputs.columns; b++){
                    this->output.vec[a][b] -= max;
                }
            }
        }

        Matrix ret_jacobian(vector<double> input_vec){
            Matrix eye;
            for (size_t a = 0; a < input_vec.size(); a++){
                vector<double> intermed(input_vec.size());
                intermed[a] = input_vec[a];
                eye.append_row(intermed);
            }
            
            Matrix minus1, minus2;
            minus1.append_row(input_vec);
            minus2.append_row(input_vec);
            minus1.Transpose();
            minus1 = minus1*&minus2;
            minus1 = minus1.return_operatorwhise_calc('*',-1);
            eye = eye+&minus1;
            minus1.empty();
            minus2.empty();
            return eye;
        }

    public:

        void forward(Matrix * passed_inputs) override {
            this->inputs = *passed_inputs;
            this->output = this->inputs;

            this->normalize();
            this->softmax();
        }

        void backward(Matrix * dvalues) override {
            
            for (size_t a = 0; a < dvalues->rows; a++){
                Matrix dval_row;
                dval_row.append_row(dvalues->vec[a]);
                dval_row.Transpose();
                
                Matrix jacob = ret_jacobian(this->output.vec[a])*&dval_row;
                jacob.Transpose();
                this->dinputs.append_row(jacob.vec[0]);
                dval_row.empty();
                jacob.empty();
            }
        }
};

class Activation_Sigmoid : public Super_Layer{
    
    private:

        Matrix inputs;

        double sigmoid(double val){
            return 1/(1+exp(-val));
        }

    public:

        void forward(Matrix * inputs) override {
            this->inputs = *inputs; 
            this->output = this->inputs;
            for (size_t a = 0; a < this->inputs.rows; a++){
                for (size_t b = 0; b < this->inputs.columns; b++){
                    this->output.vec[a][b] = this->sigmoid(this->output.vec[a][b]);
                } 
            }
        }

        void backward(Matrix * dvalues) override {
            this->dinputs = *dvalues;
            for (size_t a = 0; a < this->dinputs.rows; a++){
                for (size_t b = 0; b < this->dinputs.columns; b++){
                    this->dinputs.vec[a][b] *= (1 - this->output.vec[a][b])*this->output.vec[a][b];
                }
            }
        }
};

class Activation_Linear : public Super_Layer{
    private:

    public:

        void forward(Matrix * inputs) override {
            this->output = *inputs;
        }

        void backward(Matrix * dvalues) override {
            this->dinputs = *dvalues;
        }
};


double accuracy(Matrix * output, Matrix * results, bool categorization=true, bool binary=false, double regression_divisor=250){
            double result = 0;
            if (output->equal_proportions(results) == false){
                cout << "\nThe matrix proportions of the softmax output and the one hot encoded result Matrix do not match each other" << endl;
                exit(EXIT_FAILURE);
            }
            if (categorization==false) {
                Matrix accuracy_precision = results->stddev(results, false).return_operatorwhise_calc('/', regression_divisor);
                Matrix intermed = *output;
                for (size_t a = 0; a < results->columns; a++){
                    for (size_t b = 0; b < results->rows; b++){
                        if (abs(output->vec[b][a]-results->vec[b][a]) < accuracy_precision.vec[0][a]){
                            intermed.vec[b][a] = 1;
                        } else {
                            intermed.vec[b][a] = 0;
                        }
                    }
                }
                return intermed.mean(true).mean(false).item();
            }
            if (binary == false){
                double max = 0;
                int max_indx = 0;
                for (size_t a = 0; a < output->rows; a++){
                    max_indx = 0;
                    max = output->vec[a][0];
                    for (size_t b = 0; b < output->columns; b++){
                        if (output->vec[a][b] >= max){
                            max = output->vec[a][b];
                            max_indx = b;
                        }  
                    }
                    result += results->vec[a][max_indx];
                }
                return (result/output->rows);
            } else {
                for (size_t a = 0; a < output->rows; a++){
                    for (size_t b = 0; b < output->columns; b++){
                        if (results->vec[a][b] == 1){
                            result += output->vec[a][b];
                        } else {
                            result += 1 - output->vec[a][b];
                        } 
                    }
                }
                result /= (output->rows*output->columns);
            }
            return result;
        }

class Loss : public Super_Layer {
    
    private:
        Matrix input;
        Matrix sample_losses;

        double mean(){
            return this->sample_losses.mean(false).item();
        }

        double regularization(Layer * layer, bool l1, bool weights){
            Matrix intermed1;
            Matrix intermed2;
            double val = 0;
            if (weights == true){
                if (l1 == true){
                    if (layer->weight_regularizer_l1 <= 0) return 0;
                    intermed1 = layer->weights.return_operatorwhise_calc('|').return_operatorwhise_calc('*', layer->weight_regularizer_l1); 
                } else {
                    if (layer->weight_regularizer_l2 <= 0) return 0;
                    intermed1 = layer->weights.return_operatorwhise_calc('^',2).return_operatorwhise_calc('*', layer->weight_regularizer_l2);;
                }
            } else {
                if (l1 == true) {
                    if(layer->bias_regularizer_l1 <= 0) return 0;
                     intermed1 = layer->biases.return_operatorwhise_calc('|').return_operatorwhise_calc('*', layer->bias_regularizer_l1);;
                } else {
                    if(layer->bias_regularizer_l2 <= 0) return 0;
                    intermed1 = layer->biases.return_operatorwhise_calc('^',2).return_operatorwhise_calc('*', layer->bias_regularizer_l2);;
                } 
            } 
            intermed2 = intermed1.sum_self();
            val = intermed2.item();
            intermed1.empty();
            intermed2.empty();
            return val;
        }

    public:

        double data_loss = 0;
        double regularization_loss = 0;
        double total_loss = 0;

        double reg_loss(vector<Dense_Layer *> * layers = NULL){
            double regularization_loss = 0;
            
            for (Layer * layer : *layers){
                regularization_loss += this->regularization(layer, true, true);
                regularization_loss += this->regularization(layer, true, false);
                regularization_loss += this->regularization(layer, false, true);
                regularization_loss += this->regularization(layer, false, false); 
            }

            return regularization_loss;
        }

        Matrix give_losses(){
            return this->sample_losses;
        }

        virtual void forward(Matrix * input, Matrix * y_hat) override {}
        virtual void backward(Matrix * dvalues, Matrix * y_true) override {}

        double calculate(vector<Dense_Layer *> * layers = NULL) override {
            this->data_loss = this->mean();
            if (layers != NULL){
                this->regularization_loss = this->reg_loss(layers);
            }
            this->total_loss = this->data_loss + this->regularization_loss;
            return this->total_loss;
        }

        friend class Loss_Categorical_Cross_Entropy;
        friend class Loss_BinaryCrossentropy; 
        friend class Loss_Mean_Squared_Error;
        friend class Loss_Mean_Absolute_Error;
        friend int main();
};


class Loss_Categorical_Cross_Entropy : public Loss {

    private:
      
    public:

        void forward(Matrix * input, Matrix * y_hat) override {
            if (input->equal_proportions(y_hat) == false){
                cout << "\nThe actual results do not have the same structure as the predicted results." << endl;
                exit(EXIT_FAILURE);
            }  
            this->sample_losses.empty();
            this->input = *input;
            double border_val = 1e-7;   
            for (size_t a = 0; a < y_hat->rows; a++){
                vector<double> v = {};
                for (size_t b = 0; b < this->input.columns; b++) {
                    if (this->input.vec[a][b] < border_val){
                        this->input.vec[a][b] = border_val;
                    }
                    if (this->input.vec[a][b] > 1.0) {
                        this->input.vec[a][b] = 1.0 - border_val;
                    } 
                } 
                int index = 0;
                for (size_t b = 0; b < y_hat->columns; b++){
                    if (y_hat->vec[a][b] == 1.0){
                        index = b;
                    }
                }
                v.push_back((-1)*log(input->vec[a][index]));
                this->sample_losses.append_row(v);
            }
        }

        void backward(Matrix * dvalues, Matrix * y_true) override {
            double samples = dvalues->vec.size();
            if(y_true->is_one_hot_encoded()==false){
                y_true->return_encoded_one_hot();
            }
            
            this->dinputs = (*y_true)/dvalues;
            this->dinputs = this->dinputs.return_operatorwhise_calc('*',-1);
            this->dinputs = this->dinputs.return_operatorwhise_calc('/',samples);
        }
    
};

class Activation_Softmax_Loss_Crossentropy : public Super_Layer{
    
    private:
        
        Activation_Softmax activation;
        Loss_Categorical_Cross_Entropy loss;
    
    public:

        void forward(Matrix * inputs, Matrix * y_true) override {            
            this->activation_softmax_lc = true;
            this->activation.forward(inputs);
            this->output = this->activation.output;
            this->loss.forward(&this->output, y_true);            
        }

        void backward(Matrix * inputs, Matrix * y_true) override {
            if (inputs->equal_proportions(y_true)==false){
                cout << "\nEntropy_Loss_Softmax: output and y_true have unequal proportions, most likey the input matrix was not one hot encoded" << endl;
                exit(EXIT_FAILURE);
            } 
            this->dinputs = *inputs;
            for (size_t a = 0; a < y_true->rows; a++){
                for (size_t b = 0; b < y_true->columns; b++){
                    if (y_true->vec[a][b]==1){
                        this->dinputs.vec[a][b] -= 1;
                    }
                    this->dinputs.vec[a][b] /= inputs->rows;
                }
            }   
        } 

        double calculate(vector<Dense_Layer *> * layers = NULL){
            return this->loss.calculate(layers);
        }
    
    friend int main();
};

class Loss_BinaryCrossentropy : public Loss {
    
    private:

        Matrix individual_losses;

        Matrix clip(Matrix * ptr){
            Matrix ret = *ptr;
            for (size_t a = 0; a < ret.rows; a++){
                for (size_t b = 0; b < ret.columns; b++){
                    if (ret.vec[a][b] > 1 - 1e-7){
                        ret.vec[a][b] = 1 - 1e-7;
                    } else if (ret.vec[a][b] < 1e-7){
                        ret.vec[a][b] = 1e-7;
                    } 
                } 
            }
            return ret;
        }

    public:

        void forward(Matrix * preds, Matrix * y_true) override{
            this->binary = true;
            this->input = this->clip(preds);
            this->individual_losses = this->input;
            for (size_t a = 0; a < this->individual_losses.rows; a++){
                for (size_t b = 0; b < this->individual_losses.columns; b++){
                    this->individual_losses.vec[a][b] = -(y_true->vec[a][b]*log(this->input.vec[a][b])+(1-y_true->vec[a][b])*log(1-this->input.vec[a][b]));
                }
            }
            this->sample_losses = this->individual_losses.mean(true);
        }

        void backward(Matrix * dvalues, Matrix * y_true) override {
            Matrix clipped_dvalues = this->clip(dvalues);
            this->dinputs = clipped_dvalues;
            for (size_t a = 0; a < clipped_dvalues.rows; a++){
                for (size_t b = 0; b < clipped_dvalues.columns; b++){
                    this->dinputs.vec[a][b] = -(y_true->vec[a][b]/clipped_dvalues.vec[a][b] - (1-y_true->vec[a][b])/(1-clipped_dvalues.vec[a][b])) / dvalues->columns; // derivative
                }
            }
            this->dinputs = this->dinputs.return_operatorwhise_calc('/',dvalues->rows ); // normalize gradient
        }
}; 

class Loss_Mean_Squared_Error : public Loss {
    
    private:

    public:

        void forward(Matrix * pred, Matrix * y_true) override {
            this->categorization = false;
            Matrix intermed = pred->return_operatorwhise_calc('*', -1.0);
            this->sample_losses = ((*y_true)+&intermed).return_operatorwhise_calc('^', 2).mean(true);
        }

        void backward(Matrix * dvalues, Matrix * y_true) override {
            Matrix intermed = dvalues->return_operatorwhise_calc('*', -1.0);
            this->dinputs = ((*y_true)+&intermed).return_operatorwhise_calc('*', -2.0/dvalues->columns);
            this->dinputs = this->dinputs.return_operatorwhise_calc('/', dvalues->rows);
        }

        void put_regression_divisor(double regr_div){
            this->regression_divisor = regr_div;
        }
};

class Loss_Mean_Absolute_Error : public Loss{
    
    private:

        Matrix sign(Matrix mat){
            for (size_t a = 0; a < mat.rows; a++){
                for (size_t b = 0; b < mat.columns; b++){
                    if (mat.vec[a][b] >= 0){
                        mat.vec[a][b] = 1.0;
                    } else {
                        mat.vec[a][b] = -1.0;
                    }
                }
            }
            return mat;
        }

    public:

        void forward(Matrix * pred, Matrix * y_true) override {
            this->categorization = false;
            this->regression = true;
            Matrix intermed = pred->return_operatorwhise_calc('*', -1.0);
            this->sample_losses = ((*y_true)+&intermed).return_operatorwhise_calc('|').mean(true);
        }

        void backward(Matrix * dvalues, Matrix * y_true) override {
            Matrix intermed = dvalues->return_operatorwhise_calc('*', -1.0);
            this->dinputs = this->sign((*y_true)+&intermed).return_operatorwhise_calc('/', dvalues->ret_cols());
            this->dinputs = this->dinputs.return_operatorwhise_calc('/', dvalues->ret_rows());
        }

};

class Optimizer {
    
    private:

        double learning_rate; 
        double current_learning_rate;
        double decay;
        double iterations=0;

    public:
    
        void pre_update_params(){
            if(this->decay){
                this->current_learning_rate = this->learning_rate*(1./(1. + this->decay * this->iterations));
            }
        }

        virtual void update_params(Layer * layer) {}

        void post_update_params(){
            this->iterations += 1;
        }

    friend class Optimizer_SGD;
    friend class Optimizer_AdaGrad;
    friend class Optimizer_RMSprop;
    friend class Optimizer_Adam;
};

class Optimizer_SGD : public Optimizer {
    
    private:

        double momentum=0;
    
    public:
        
        Optimizer_SGD(double learning_rate=1.0 , double decay=0.0, double momentum=0.0){
            this->learning_rate = learning_rate;
            this->current_learning_rate = learning_rate;
            this->decay = decay;
            this->momentum = momentum;
        }

        void update_params(Layer * layer) override {
            Matrix weight_updates;
            Matrix bias_updates; 
            Matrix intermed;
            Matrix * ptr_w = NULL;
            Matrix * ptr_b = NULL;

            if (this->momentum){
                
                if (layer->weight_momentums.ret_rows() == 0 && layer->weight_momentums.ret_cols() == 0){
                    layer->weight_momentums.null_init(layer->weights.ret_rows(),layer->weights.ret_cols());
                    layer->bias_momentums.null_init(1,layer->biases.ret_cols());
                }

                intermed = layer->dweights.return_operatorwhise_calc('*', this->current_learning_rate*(-1.));
                ptr_w = &layer->dweights;
                weight_updates = layer->weight_momentums.return_operatorwhise_calc('*', this->momentum)+&intermed;
                layer->weight_momentums = weight_updates;
                intermed.empty();

                intermed = layer->dbiases.return_operatorwhise_calc('*', this->current_learning_rate*(-1.));
                ptr_b = &layer->dbiases;
                bias_updates = layer->bias_momentums.return_operatorwhise_calc('*', this->momentum)+&intermed;
                layer->bias_momentums = bias_updates;
                intermed.empty();

            } else {

                weight_updates = layer->dweights.return_operatorwhise_calc('*', this->current_learning_rate*(-1.));
                ptr_w = &layer->dweights;
                
                bias_updates = layer->dbiases.return_operatorwhise_calc('*', this->current_learning_rate*(-1.));
                ptr_b = &layer->dbiases;
            }

            layer->weights = layer->weights+&weight_updates;
            layer->biases = layer->biases+&bias_updates;
            ptr_w->empty();
            ptr_b->empty();  
        }   
};


class Optimizer_AdaGrad : public Optimizer {
    
    private:

        double epsilon=0;

        void descent(Layer * layer, bool weights){
            Matrix  intermed1;
            Matrix  intermed2;
            Matrix  intermed3;
            Matrix * ptr = NULL;
            if (weights == true){
                intermed1 = layer->dweights.return_operatorwhise_calc('*', this->current_learning_rate*(-1.));
                intermed2 = layer->weight_cache.return_operatorwhise_calc('r',2.);
            } else {
                intermed1 = layer->dbiases.return_operatorwhise_calc('*', this->current_learning_rate*(-1.));
                intermed2 = layer->bias_cache.return_operatorwhise_calc('r',2.);
            }

            intermed3 = intermed2.return_operatorwhise_calc('+', this->epsilon);
            intermed2.empty();
            intermed2 = intermed1/&intermed3;
            intermed1.empty();
            intermed3.empty();
            
            if (weights == true){
                layer->weights = layer->weights+&intermed2;
                intermed2.empty();
            }else{
                layer->biases = layer->biases+&intermed2;
                intermed2.empty();
            }
        }
    
    public:
        
        Optimizer_AdaGrad(double learning_rate=1.0 , double decay=0.0, double epsilon=1e-7){
            this->learning_rate = learning_rate;
            this->current_learning_rate = learning_rate;
            this->decay = decay;
            this->epsilon = epsilon;
        }

        void update_params(Layer * layer) override {
            vector<vector<double>> * ptr_w = NULL;
            vector<vector<double>> * ptr_b = NULL;
            Matrix * ptr = NULL;
                
            if (layer->weight_cache.ret_rows() == 0 && layer->weight_cache.ret_cols() == 0){
                layer->weight_cache.null_init(layer->weights.ret_rows(),layer->weights.ret_cols());
                layer->bias_cache.null_init(1,layer->biases.ret_cols());
            }

            layer->weight_cache += layer->dweights.return_operatorwhise_calc('^',2.);
            layer->bias_cache += layer->dbiases.return_operatorwhise_calc('^',2.);

            this->descent(layer, true);
            this->descent(layer, false);
        }   

};


class Optimizer_RMSprop : public Optimizer{
    
    private:

        double epsilon=0;
        double rho=0;

        void make_cache(Layer * layer, bool weight=true){
            Matrix intermed1;
            Matrix intermed2;
            Matrix intermed3;

            if (weight == true){
                intermed1 = layer->weight_cache.return_operatorwhise_calc('*',this->rho);
                intermed2 = layer->dweights.return_operatorwhise_calc('^',2);
            } else {
                intermed1 = layer->bias_cache.return_operatorwhise_calc('*',this->rho);
                intermed2 = layer->bias_cache.return_operatorwhise_calc('^',2);
            }

            intermed3 = intermed2.return_operatorwhise_calc('*',(1-this->rho));
            intermed2.empty();
            if (weight == true){
                layer->weight_cache = intermed1+&intermed3;
            } else {
                layer->bias_cache = intermed1+&intermed3;
            }
 
            intermed1.empty();
            intermed3.empty();
        }

        void descent(Layer * layer, bool weight = true){
            Matrix intermed1;
            Matrix intermed2;
            Matrix intermed3;

            if (weight == true){
                intermed1 = layer->dweights.return_operatorwhise_calc('*', this->current_learning_rate*(-1.));
                intermed2 = layer->weight_cache.return_operatorwhise_calc('r',2.);
            } else {
                intermed1 = layer->dbiases.return_operatorwhise_calc('*', this->current_learning_rate*(-1.));
                intermed2 = layer->bias_cache.return_operatorwhise_calc('r', 2.);
            }

            intermed3 = intermed2.return_operatorwhise_calc('+', this->epsilon);
            intermed2.empty();
            intermed2 =intermed1/&intermed3;
            
            if (weight == true){
                layer->weights = layer->weights+&intermed2;
            } else {
                layer->biases = layer->biases+&intermed2;
            }
            
            intermed1.empty();
            intermed2.empty();
            intermed3.empty();
        }
    
    public:
        
        Optimizer_RMSprop(double learning_rate=0.001 , double decay=0.0, double epsilon=1e-7, double rho=0.9){
            this->learning_rate = learning_rate;
            this->current_learning_rate = learning_rate;
            this->decay = decay;
            this->epsilon = epsilon;
            this->rho = rho;
        }

        void update_params(Layer * layer) override {
           
           if (layer->weight_cache.ret_rows() == 0 && layer->weight_cache.ret_cols() == 0){
                layer->weight_cache.null_init(layer->weights.ret_rows(),layer->weights.ret_cols());
                layer->bias_cache.null_init(1,layer->biases.ret_cols());
            }

            make_cache(layer, true);
            make_cache(layer, false);

            descent(layer, true);
            descent(layer, false);
        }   

};


class Optimizer_Adam : public Optimizer {

    private:

        double epsilon=0;
        double beta_1=0;
        double beta_2=0;

    public:

        Optimizer_Adam(double learning_rate=0.001, double decay=0.0, double epsilon=1e-7, double beta_1 = 0.9, double beta_2 = 0.999){
            this->learning_rate = learning_rate;
            this->current_learning_rate = learning_rate;
            this->decay = decay;
            this->epsilon = epsilon;
            this->beta_1 = beta_1;
            this->beta_2 = beta_2;
        }

        void update_params(Layer * layer) override {
            Matrix intermed1;
            Matrix intermed2;
            Matrix weight_momentums_corrected;
            Matrix bias_momentum_corrected;
            Matrix weight_cache_corrected;
            Matrix bias_cache_corrected;

            if (layer->weight_cache.rows == 0 && layer->weight_cache.columns == 0 ){
                layer->weight_momentums.null_init(layer->weights.ret_rows(), layer->weights.ret_cols());
                layer->weight_cache.null_init(layer->weights.ret_rows(), layer->weights.ret_cols());
                layer->bias_momentums.null_init(layer->biases.ret_rows(), layer->biases.ret_cols());   // keep rows being 1 usually in mind
                layer->bias_cache.null_init(layer->biases.ret_rows(), layer->biases.ret_cols());
            }

            intermed1 = layer->weight_momentums.return_operatorwhise_calc('*', this->beta_1);
            layer->weight_momentums = layer->dweights.return_operatorwhise_calc('*', (1-this->beta_1))+&intermed1;
            intermed1.empty();

            intermed1 = layer->bias_momentums.return_operatorwhise_calc('*', this->beta_1);
            layer->bias_momentums = layer->dbiases.return_operatorwhise_calc('*', (1-this->beta_1))+&intermed1;
            intermed1.empty();

            weight_momentums_corrected = layer->weight_momentums.return_operatorwhise_calc('/', (1 - pow(this->beta_1, this->iterations+1)));
            bias_momentum_corrected = layer->bias_momentums.return_operatorwhise_calc('/', (1 - pow(this->beta_1, this->iterations+1)));

            intermed1 = layer->dweights.return_operatorwhise_calc('^', 2);
            intermed2 = intermed1.return_operatorwhise_calc('*', (1 - this->beta_2));
            intermed1.empty();
            layer->weight_cache = layer->weight_cache.return_operatorwhise_calc('*', this->beta_2)+&intermed2;
            intermed2.empty();

            intermed1 = layer->dbiases.return_operatorwhise_calc('^', 2);
            intermed2 = intermed1.return_operatorwhise_calc('*', (1 - this->beta_2));
            intermed1.empty();
            layer->bias_cache = layer->bias_cache.return_operatorwhise_calc('*', this->beta_2)+&intermed2;
            intermed2.empty(); 

            weight_cache_corrected = layer->weight_cache.return_operatorwhise_calc('/', (1 - pow(this->beta_2, this->iterations+1)));
            bias_cache_corrected = layer->bias_cache.return_operatorwhise_calc('/', (1 - pow(this->beta_2, this->iterations+1)));

            intermed1 = weight_cache_corrected.return_operatorwhise_calc('r', 2);
            intermed2 = intermed1.return_operatorwhise_calc('+', this->epsilon);
            intermed1.empty();
            intermed1 = weight_momentums_corrected.return_operatorwhise_calc('*', this->current_learning_rate*(-1));
            layer->weights += intermed1/&intermed2;
            intermed1.empty();
            intermed2.empty();

            intermed1 = bias_cache_corrected.return_operatorwhise_calc('r', 2);
            intermed2 = intermed1.return_operatorwhise_calc('+', this->epsilon);
            intermed1.empty();
            intermed1 = bias_momentum_corrected.return_operatorwhise_calc('*', this->current_learning_rate*(-1));
            layer->biases += intermed1/&intermed2;
            intermed1.empty();
            intermed2.empty();
        }

};

class Model {

    private:

        void forwarding(Matrix * inp_data , Matrix * y_hat){
            this->list[0]->forward(inp_data);
            for (size_t a = 1; a < this->list.size()-1; a++){
                this->list[a]->forward(&list[a-1]->output);
            }
            this->list[this->list.size()-1]->forward(&this->list[this->list.size()-2]->output, y_hat);
        }

        void backwarding(Matrix * y_hat){           
            int which_lay = 2;
            if (this->list[this->list.size()-1]->activation_softmax_lc == true){which_lay = 1;}
            this->list[this->list.size()-1]->backward(&this->list[this->list.size()-which_lay]->output, y_hat);
            for (size_t a = this->list.size()-2; a != -1; a--){
                this->list[a]->backward(&this->list[a+1]->dinputs);
            }        
        }

    public:
    
        vector<Super_Layer*> list; 
        vector<Dense_Layer*> dense_list;
        Data * data;
    
        Model(Data * input, bool onehot=false, bool shuffle=false, bool standardize=false, bool result_norm=false, double train_prop=0.8, double val_prop=0.1, double test_prop=0.1){
            if (standardize == true){
                input->max_standardization(onehot, result_norm);
            }
            if (shuffle == true){
                input->data_random_shuffle();
            }
            if (onehot==true){
                input->results_one_hot_ecoded_matrix();
            }
            input->dataset_distribution(train_prop, val_prop, test_prop);
            this->data = input;
        }

        void push_sequential(Super_Layer * super_layer){
            list.push_back(super_layer); 
            if (super_layer->dense == true){
                this->dense_list.push_back((Dense_Layer*)super_layer);
            }
        }

        void pop_sequential(int index=0){
            this->list.erase(this->list.begin()+index, this->list.begin()+index+1);
        }

        void train(Optimizer * optim ,int batchsize=0, int epochs=1000, int epoch_showing=100){
            for (size_t epoch = 0; epoch < epochs; epoch++){
                double loss = 0, acc = 0, divisor = 0;
                do {
                    Matrix data_batch = this->data->batch_return(true, &this->data->training_data, &this->data->training_results, batchsize);
                    Matrix result_batch = this->data->batch_return(false);
                    this->forwarding(&data_batch, &result_batch);
                    if (epoch % epoch_showing == 0){
                        loss += this->list[this->list.size()-1]->calculate(&dense_list);
                        acc += accuracy(&this->list[this->list.size()-2]->output, &result_batch, this->list[this->list.size()-1]->categorization, this->list[this->list.size()-1]->binary, this->list[this->list.size()-1]->regression_divisor);
                        divisor += 1.0;
                    }
                    this->backwarding(&result_batch);
                    optim->pre_update_params();
                    for (size_t a = 0; a < this->dense_list.size(); a++){
                        optim->update_params(this->dense_list[a]);
                    }
                    optim->post_update_params();
                } while(this->data->batch_iterate == true);
                if (epoch % epoch_showing == 0){
                           cout << "\nEpoch: " << epoch << " / " << epochs << " , Training loss: " << loss/divisor << " , Training accuracy : " << acc / divisor << endl;
                }
            }
        }

        void validate(){
            this->forwarding(&this->data->validation_data, &this->data->validation_results);
            cout << "\nValidation loss: " << this->list[this->list.size()-1]->calculate(&dense_list) << " , Validation accuracy : " << accuracy(&this->list[this->list.size()-2]->output, &this->data->validation_results, this->list[this->list.size()-1]->categorization, this->list[this->list.size()-1]->binary, this->list[this->list.size()-1]->regression_divisor) << endl;
        }

        void test(){
            this->forwarding(&this->data->testing_data, &this->data->testing_results);
            cout << "\nTesting loss: " << this->list[this->list.size()-1]->calculate(&dense_list) << " , Testing accuracy : " << accuracy(&this->list[this->list.size()-2]->output, &this->data->testing_results, this->list[this->list.size()-1]->categorization, this->list[this->list.size()-1]->binary, this->list[this->list.size()-1]->regression_divisor) << endl;
        }

        Matrix predict(Matrix * inp_data , Matrix * y_hat){
            this->forwarding(inp_data, y_hat);
            if (this->list[this->list.size()-1]->activation_softmax_lc==true)
            {
                return this->list[this->list.size()-1]->output;
                
            } 
            return this->list[this->list.size()-2]->output;
        }
};



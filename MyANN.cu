#include <iostream>
#include <fstream>
using namespace std;
float mse=0;
float l1=0;
class matrix{
    public:
    float* p_host;
    float* p_device;
    int row, col;
    matrix* next;//next matrix...can be initialised to null but no need
    matrix* last;//last matrix

    void HostMem(int x,int y){
        p_host = (float *)calloc(x*y,sizeof(float));//calloc to initialize all elements to 0
    }
    void DeviceMem(int x,int y){
        cudaMalloc(&p_device, x*y*sizeof(float));
        row=x;
        col=y;
        this->HostMem(x,y);
    }
};

class ann{
    public:
    matrix input;//input neuron
    matrix hidden;//hidden neurons
    matrix output;//output neurons
    matrix weights_i;//intput weights
    matrix weights_h;//hidden weights
    matrix bias_i;//input bias
    matrix bias_h;//hidden bias
    matrix delta_o;//delta of output layer
    matrix delta_h;//delta of hidden layer
    matrix delta_i;//delta of input layer
    float result;//make matrix if needed

};
//made class ann to keep everything together
/*
layer:   {neurons->


                             }
weights: {neurons->
 weights
 !       
}
row of layer*col of weights-> value of next neuron
*/
//blocks->rows, threads->elements,columns, grids
//for deltas, divide value of neuron by all last weights(row divided by column)
__global__ void matrix_mul_f(matrix *layer, matrix *weights, matrix *bias){
    layer->next->p_device[threadIdx.x]+=(layer->p_device[blockIdx.x]*weights->p_device[blockDim.x*blockIdx.x+threadIdx.x]+bias->p_device[blockDim.x*blockIdx.x+threadIdx.x]);
}

__global__ void setval(float a, matrix* x){
    x->p_device[(blockDim.x*blockIdx.x)+threadIdx.x]=a;
    //works only for device memory...for host memory just copy...
}

__global__ void del(matrix *del, matrix *weights){//the working layer and weights of the last layer
    del->last->p_device[(blockDim.x*blockIdx.x)+threadIdx.x]+=(del->p_device[((blockDim.x*blockIdx.x)+threadIdx.x)]/weights->p_device[((blockDim.x*blockIdx.x)+threadIdx.x)]);
}

__global__ void weight_update(matrix *weight, matrix *delta,matrix *neuron){
    weight->p_device[(blockDim.x*blockIdx.x+threadIdx.x)]-=(neuron->p_device[blockIdx.x]*0.1*delta->p_device[(blockDim.x*blockIdx.x)+threadIdx.x]);
}

__global__ void bias_update(matrix *delta, matrix *bias){
	      //args:  bias,del
    bias->p_device[(blockDim.x*blockIdx.x)+threadIdx.x]-=(0.1*delta->p_device[(blockDim.x*blockIdx.x)+threadIdx.x]);
}

void getacc(float a, float b){
    cout<<((a-b)/b)*100<<"\n";
}

void feed_forward(ann *a,int choice,int pos){
    std::ifstream test;
    if(choice==0)//for training
    test.open("abelone.train");
    else 
    test.open("abelone.test");
    int iter,in,out;
    test>>iter;
    test>>in;
    test>>out;
    test.seekg(in*pos, ios::beg); 
    int row,col;
   // cout<<pos<<"    "<<" "<<iter<<" "<<in<<" "<<out<<"\n";
    for(int i=0;i<in;i++){
        float temp;
        test>>temp;
       // cout<<" "<<temp;
        a->input.p_host[i]=temp;
    }
     //   cout<<"\n";
    test.seekg(in*pos+1, ios::beg);
    float temp=0;
    test>>temp;
    a->result=temp;
   // cout<<"\n"<<temp<"\n";
    cudaMemcpy(a->input.p_device, a->input.p_host,sizeof(float)*(a->input.row)*(a->input.col),cudaMemcpyHostToDevice);
    //reading inputs
    //reading final answer
    row=a->input.row;//in
    col=a->weights_i.col;//hid
    //cout<<row<<"x"<<col<<"\n";
    matrix_mul_f<<<col,row>>>(&(a->input), &(a->weights_i), &(a->bias_i));//row number of blocks and col numbe rof threads
    row=a->hidden.row;//hid
    col=a->weights_h.col;//out
    // cout<<row<<"x"<<col<<"\n";
    matrix_mul_f<<<col,row>>>(&(a->hidden), &(a->weights_h), &(a->bias_h));
    cudaMemcpy(a->output.p_host, a->output.p_device,sizeof(float)*(a->output.row)*(a->output.col),cudaMemcpyDeviceToHost);//device to host
    //output can now be accessed from anywhere
    //cout<<*(a->output.p_host)<<"\n\n";
    getacc(*(a->output.p_host),a->result);
         } 

void back_propagation(ann *a){
    //feed_forward(a,0);
    std::ifstream train;
    train.open("abelone.train"); 
    int iter,in,out;
    train>>iter;//number of samples
    train>>in;//number of inputs
    train>>out;//number of outputs
    int row,col;
    int batch;
    batch=75;//number of batches(best to have a factor of number of input entries)
    //INITIALISE BATCH
    for(int j=0;j<batch-1;j++){ 
        for(int i=0;i<iter/batch;i++){
            feed_forward(a,0,(j*batch)+i);
            mse+=((*(a->output.p_host)-(*a).result)*(*(a->output.p_host)-(*a).result));
            l1+=(*(a->output.p_host)-(*a).result);}
        mse/=batch;
        l1/=batch;
        *(a->delta_o.p_host)=l1;
        cudaMemcpy(a->delta_o.p_device, a->delta_o.p_host,sizeof(float)*(a->delta_o.row)*(a->delta_o.col),cudaMemcpyHostToDevice);
        row=a->output.row;
        col=a->weights_h.col;
        del<<<col,row>>>(&(a->delta_h), &(a->weights_h));//row number of threads and col number of blocks
        row=a->hidden.row;
        col=a->weights_i.col;
        //wether we uise weights or neuron columns doesnt matter as we have only two values possible(row col) for all variables for a particular layer  
        del<<<col,row>>>(&(a->delta_i),&(a->weights_i));//row number of blocks and col numbe rof threads
        weight_update<<<a->weights_h.col,a->output.row>>>(&(a->delta_h), &(a->weights_h),&(a->hidden));//row number of blocks and col numbe rof threads
        bias_update<<<a->weights_h.col,a->output.row>>>(&(a->delta_h), &(a->bias_h));
        weight_update<<<a->weights_i.col,a->hidden.row>>>(&(a->weights_i),&(a->delta_i),&(a->input));//row number of blocks and col numbe rof threads
        bias_update<<<a->weights_h.col,a->output.row>>>(&(a->delta_i), &(a->bias_i));
        l1=0;
        setval<<<32,in>>>(0,&(a->delta_i));
        setval<<<out,32>>>(0,&(a->delta_h));
        setval<<<1,out>>>(0,&(a->delta_o));
        setval<<<1,32>>>(0,&(a->hidden));
        setval<<<1,out>>>(0,&(a->output));
}
}

//1 block->number next layer threads....no of present layer no of blocks
int main(){
    ann my_ann;//made new ann
    std::ifstream data;//this for input
    int in, iter , out,hid;
    data.open("abelone.train");
    data>>iter;
    data>>in;
    data>>out;
    hid=32;//in general good number for hidden layer. May help with making warps
    my_ann.input.DeviceMem(in,1);
    my_ann.hidden.DeviceMem(hid,1);
    my_ann.output.DeviceMem(out,1);
    my_ann.weights_i.DeviceMem(in,hid);
    my_ann.weights_h.DeviceMem(hid,out);
    my_ann.bias_i.DeviceMem(in,hid);
    my_ann.bias_h.DeviceMem(hid,out);
    my_ann.delta_i.DeviceMem(in,hid);
    my_ann.delta_h.DeviceMem(hid,out);
    my_ann.delta_o.DeviceMem(out,1);
    //all matrices initialized in device memory

    setval<<<1,in>>>(0,&my_ann.input);
    setval<<<1,hid>>>(0,&my_ann.hidden);
    setval<<<1,out>>>(0,&(my_ann.output));
    setval<<<hid,in>>>(1,&(my_ann.weights_i));
    setval<<<hid,in>>>(0,&(my_ann.delta_i));
    setval<<<hid,in>>>(0,&(my_ann.bias_i));
    setval<<<out,hid>>>(1,&(my_ann.weights_h));
    setval<<<out,hid>>>(0,&(my_ann.bias_h));
    setval<<<out,hid>>>(0,&(my_ann.delta_h));
    //all matrices given appropriate values


    my_ann.input.next=&(my_ann.hidden);
    my_ann.hidden.next=&(my_ann.output);
    my_ann.output.last=&(my_ann.hidden);
    my_ann.hidden.last=&(my_ann.input);
    my_ann.delta_o.last=&(my_ann.delta_h);
    my_ann.delta_h.last=&(my_ann.delta_i);
    my_ann.delta_i.next=&(my_ann.delta_h);
    my_ann.delta_h.next=&(my_ann.delta_o);
    my_ann.bias_h.last=&(my_ann.bias_i);
    my_ann.bias_i.next=&(my_ann.bias_h);
    //all layers and matrices connected

    //ann ready for data manipulation
    int epoches=10;//this can be changed depending on accuracy
    while(epoches!=0){
    back_propagation(&my_ann);
    epoches--;}

    std::ifstream read;
    read.open("abelone.train");
    read>>iter;
    read>>in;
    read>>out;
    for(int i=0;i<iter;i++)
    feed_forward(&my_ann,1,i);
    return 0;
}


//<<<blocks per grid,threads per block>>>
//cannot treat weight ad 2d array

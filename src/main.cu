#include "../include/tensor.cuh"
#include <cstddef>
#include <iostream>
#include <utility>

using namespace std;
int main(){
    Tensor<float> t(6, Device::CPU);
    t.fill(3.14f);

    cout<<"CPU tensor values: ";
    for(size_t i=0;i<t.size;i++){
        cout<<t.data[i]<<" ";
    }
    cout<<"\n";
    
    //move to GPU
    t.to_gpu();
    cout<<"Moved to GPU now\n";
    
    //moving back to CPU and verify
    t.to_cpu();
    cout<<"Back to CPU: ";
    for(size_t i=0;i<t.size;i++){
        cout<<t.data[i]<<" ";
    }
    cout<<"\n";
    //testing move sementics
    Tensor<float> t2 = move(t);
    cout<< "After move: t.data is "
        << (t.data ==nullptr ? "null (corret)" : "not null (bug!)")
        << "\n";
    return 0;
}
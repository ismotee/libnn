//
//  libnn.h
//  libnn
//
//  Created by Ismo Torvinen on 23.5.2016.
//  Copyright (c) 2016 Ismo Torvinen. All rights reserved.
//

#ifndef __libnn__libnn__
#define __libnn__libnn__

#include <stdio.h>
#include <vector>

class Neuron
{
public:
    
    Neuron();
    
    virtual void forward(void) = 0;
    virtual void back(void) = 0;
    virtual void setInput(float value) = 0;
    virtual void addLink(Neuron* neuron) = 0;
    
    float error;
    float output;

protected:
    std::vector<float> weights;  // bias + weights
    float learnRate;
};

class HiddenNeuron : public Neuron
{
public:
    HiddenNeuron(float learn_rate = 0.01f);
    
    std::vector<Neuron*> upper;
    
    void forward(void);
    void back(void);
    void setInput(float value);
    void addLink(Neuron* neuron);
};

class InputNeuron : public Neuron
{
public:
    InputNeuron(float learn_rate = 0.01f);
    
    float input;
    
    void forward(void);
    void back(void);
    void setInput(float value);
    void addLink(Neuron* neuron);
    
};


class NLayer
{
public:
    NLayer();
    
    std::vector<Neuron*> neurons;
    
    void init(int neurons_n, bool is_input_layer = false);
    void forward();
    void back();
    void clearErrors();

    // for output layer;
    std::vector<float> getOutput();
    void setError(std::vector<float> errors);
    
    // for inputLayer
    void setInputs(std::vector<float> inputs);
    
};


class NNet
{
public:
    NNet();
    
    void init(int inputs_n, int hidden_layers_n, int hidden_neurons_n, int outputs_n);
    
    std::vector<float> forward(std::vector<float>& inputs);
    void back(std::vector<float> desired_output);
    void linkInput(int input_id, int hidden_id);
    void linkHidden(int source_layer, int source_id, int dest_layer, int dest_id);
    
protected:
    NLayer inputLayer;
    std::vector<NLayer> hiddenLayers;
    NLayer outputLayer;
};


#endif /* defined(__libnn__libnn__) */

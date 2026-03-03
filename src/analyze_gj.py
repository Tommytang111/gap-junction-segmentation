"""
Analyze neuron-specific gap junction expression and neuron-neuron gap junction 
connectivity in C. elegans nerve ring.
Tommy Tang
Last Updated: Feb 27, 2025
"""

#Libraries
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from transforms.gj_membrane import *
from utils import check_output_directory

class GapJunctionAnalyzer:
    def __init__(self, name, neurons, gap_junctions):
        self.name = name
        self.neurons = neurons
        self.gap_junctions = gap_junctions
        
    def extract_membranes(self):
        self.membrane = extract_membranes_with_gradient(neurons=self.neurons)
        
    def expand_neurons(self):
        self.expanded_neurons = expand_neurons_to_membrane(self.neurons, self.membrane)
        
    def analyze_neurons(self):
        self.neuron_gjs = analyze_gj_per_neuron(self.membrane, self.expanded_neurons, self.gap_junctions)
        
    def analyze_neuron_pairs(self):
        self.contactome, self.electrical_connectivity self.normalized_electrical_connectivity = get_electrical_connectivity(self.membrane, self.expanded_neurons, self.gap_junctions)
        
def main():
    #Count starting time
    start_time = time.time()
    
    pipeline = GapJunctionAnalyzer(
        #Name of job (I recommend model + data + "gj analysis")
        name="unet_p03lmvzp_sem_adult_s000-699_gj_analysis",
        #Path to volume of proofread neuron labels
        neurons="/home/tommy111/scratch/Neurons/SEM_adult_neurons_only_with_labels_block_downsampled4x.npy",
        #Path to volume of gap junction predictions
        gap_junctions="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_p03lmvzp/sem_adult_s000-699/volume_constrained_in_NR_block_downsampled4x.npy"
    )
    
    print("Analysis Pipeline initialized with name:", pipeline.name)
    print(f"Shape of neuron volume: {pipeline.neurons.shape}")
    print()
    
    #Step 1: Neuron mask -> Membrane
    pipeline.extract_membranes()
    print("Membrane extracted and saved in:", )
    print()
    
    #Step 2: Membrane + Neuron mask -> Expanded Neuron mask
    pipeline.expand_neurons()
    print("Expanded neuron mask created and saved in:", )
    print()
    
    #Step 3: Expanded Neuron Mask + Membrane + Gap Junctions -> Neuron-specific gap junction expression
    pipeline.analyze_neurons()
    print("Neurons analysis completed and saved in:", )
    print()
    
    #Step 4: Expanded Neuron Mask + Membrane + Gap Junctions -> Neuron-neuron gap junction connectivity
    pipeline.analyze_neuron_pairs()
    print("Neuron pair analysis completed and saved in:", )
    print()
    
    #Count ending time
    end_time = time.time()
    print(f"Gap Junction Analysis completed in : {(end_time - start_time) / 3600:.2f} hours")
    
if __name__ == "__main__":
    main()
        
    
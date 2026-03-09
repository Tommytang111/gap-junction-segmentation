"""
Analyze neuron-specific gap junction expression and neuron-neuron gap junction 
connectivity in C. elegans nerve ring.
Tommy Tang
Last Updated: March 5, 2025
"""

#Libraries
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from transforms.gj_membrane import *
from utils import check_output_directory

class GapJunctionAnalyzer:
    def __init__(self, name, neurons, gap_junctions, membrane_dir, neurons_expanded_dir, results_dir, save):
        self.name = name
        self.model_name = self.name.split("_")[1]
        self.dataset_name = self.name.split("_")[2] + "_" + self.name.split("_")[3]
        self.neurons = np.load(neurons)
        self.expanded_neurons_dir = neurons_expanded_dir
        self.membrane_dir = membrane_dir
        self.results_dir = results_dir
        self.gap_junctions = np.load(gap_junctions)
        self.save = save
        
    def extract_membranes(self):
        save_file = Path(self.membrane_dir) / (self.dataset_name + "_neuron_membrane.npy")
        self.membrane = extract_membranes_with_gradient(neurons=self.neurons, save=self.save, save_path=save_file)
        
    def expand_neurons(self):
        save_file = Path(self.expanded_neurons_dir) / (self.dataset_name + "_neurons_only_with_labels_non_uniform_expanded.npy")
        self.expanded_neurons = expand_neurons_to_membrane(neuron_labels=self.neurons, membrane_mask=self.membrane, save=self.save, save_path=save_file)
        
    #Below methods have outputs always saved
    def analyze_neurons(self):
        save_file = Path(self.results_dir) / (self.dataset_name + "_neuronal_gj_analysis_" + self.model_name +".pkl")
        self.neuron_gjs = analyze_gj_per_neuron(self.membrane, self.expanded_neurons, self.gap_junctions, save_path=save_file)
        
    def analyze_neuron_pairs(self):
        save_file1 = Path(self.results_dir) / (self.dataset_name + "_contactome_" + self.model_name + ".pkl")
        save_file2 = Path(self.results_dir) / (self.dataset_name + "_gj_connectivity_" + self.model_name + ".pkl")
        save_file3 = Path(self.results_dir) / (self.dataset_name + "_normalized_gj_connectivity_" + self.model_name + ".pkl")
        self.contactome, self.electrical_connectivity, self.normalized_electrical_connectivity = get_electrical_connectivity(self.membrane, 
                                                                                                                             self.expanded_neurons, 
                                                                                                                             self.gap_junctions,
                                                                                                                             contactome=str(save_file1),
                                                                                                                             gj_connectivity=str(save_file2),
                                                                                                                             normalized_gj_connectivity=str(save_file3),
                                                                                                                            )
        
def main():
    #Count starting time
    start_time = time.time()
    
    pipeline = GapJunctionAnalyzer(
        #Name of job (Must use model + data + "gj analysis"). It is essential to include underscores in this format for the pipeline to work.
        #EXAMPLE "unet_p03lmvzp_sem_adult_s000-699_gj_analysis"
        name="unet_p03lmvzp_sem_adult_s000-699_gj_analysis",
        #Path to volume of proofread neuron labels
        neurons="/home/tommy111/scratch/Neurons/SEM_adult/SEM_adult_neurons_only_with_labels_block_downsampled4x.npy",
        #Path to volume of gap junction predictions
        gap_junctions="/home/tommy111/projects/def-mzhen/tommy111/outputs/volumetric_results/unet_u4lqcs5g/sem_adult_s000-699/volume_constrained_in_NR_block_downsampled4x.npy",
        #Membrane save path
        membrane_dir="/home/tommy111/projects/def-mzhen/tommy111/outputs/membranes/sem_adult",
        #Expanded neurons save path
        neurons_expanded_dir="/home/tommy111/projects/def-mzhen/tommy111/outputs/neurons/sem_adult",
        #Analysis results save path
        results_dir="/home/tommy111/projects/def-mzhen/tommy111/outputs/analysis_results/sem_adult",
        #Save membrane and expanded neuron objects
        save=True
    )
    
    print("Analysis Pipeline initialized with name:", pipeline.name)
    print(f"Shape of neuron volume: {pipeline.neurons.shape}")
    print()
    
    #Step 1: Neuron mask -> Membrane
    pipeline.extract_membranes()
    print("Membrane extracted.\n")
    
    #Step 2: Membrane + Neuron mask -> Expanded Neuron mask
    pipeline.expand_neurons()
    print("Expanded neuron mask created.\n")
    
    #Step 3: Expanded Neuron Mask + Membrane + Gap Junctions -> Neuron-specific gap junction expression
    pipeline.analyze_neurons()
    print("Neurons analysis completed.\n")
    
    #Step 4: Expanded Neuron Mask + Membrane + Gap Junctions -> Neuron-neuron gap junction connectivity
    pipeline.analyze_neuron_pairs()
    print("Neuron pair analysis completed.\n")
    
    #Count ending time
    end_time = time.time()
    print(f"Gap Junction Analysis completed in : {(end_time - start_time) / 3600:.2f} hours")
    
if __name__ == "__main__":
    main()
        
    
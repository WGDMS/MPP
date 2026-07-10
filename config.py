

CONFIG = {
    "dataset": "ESOL", # BACE, HIV, Tox21, SIDER, ClinTox, ESOL,FreeSolv, Lipophilicity
    "task_type": "regression",   # binary, multitask, regression
    "split_type": "scaffold",  # random or scaffold

    "n_output": 1,              # BBBP/BACE/HIV/ESOL/FreeSolv/Lipophilicity = 1
                                #Tox21= 12 SIDER=27 ClinTox=2
    #"tasks": [                                              
    #    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    #    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    #    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    #],
    
    #"tasks" : ['Hepatobiliary disorders',                   #SIDER
    #   'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
    #   'Investigations', 'Musculoskeletal and connective tissue disorders',
    #   'Gastrointestinal disorders', 'Social circumstances',
    #   'Immune system disorders', 'Reproductive system and breast disorders',
    #   'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
    #  'General disorders and administration site conditions',
    #   'Endocrine disorders', 'Surgical and medical procedures',
    #   'Vascular disorders', 'Blood and lymphatic system disorders',
    #   'Skin and subcutaneous tissue disorders',
    #   'Congenital, familial and genetic disorders',
    #   'Infections and infestations',
    #   'Respiratory, thoracic and mediastinal disorders',
    #   'Psychiatric disorders', 'Renal and urinary disorders',
    #   'Pregnancy, puerperium and perinatal conditions',
    #   'Ear and labyrinth disorders', 'Cardiac disorders',
    #  'Nervous system disorders',
    #   'Injury, poisoning and procedural complications'],
    
    
    #"tasks" : ['CT_TOX', 'FDA_APPROVED'],   #ClinTox
    

  #  "sampling_modes": ["uniform", "chem"],   
    "sampling_modes": ["chem"],  
    "w_conj": 0.5,                            
    "w_ring": 0.3,                           

    "batch_sizes": [40],
    "lrs": [1e-3],
    "epochs_list": [100],
    "patiences": [30],
    
    "seeds": [0, 1, 2, 3, 4],
   
    "walk_lengths": [30],
    "num_layers_list": [3],

    "sample_rate": 1.0,
    "window_size": 8,

    "walk_encoder": "mamba", 
    "models_dir": "models",
    "results_dir": "results",
}
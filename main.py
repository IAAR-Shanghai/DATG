import argparse
import os
from method.datg.datg import ModelResourceManager, DatgTextGenerator
from method.fudge.fudge import FudgeTextGenerator
from method.preadd.preadd import PreaddTextGenerator
from tqdm import tqdm
import pandas as pd
from config import MODEL_PATHS, TASK_CONFIGURATIONS, GENERATION_CONFIGS

NUM_SENTENCES = 30
NODES_NUM = 10
BOOST_VALUE = 4.0
AVOID_VALUE = 6.0
FUDGE_ALPHA = 0.5
PREADD_STRENGTH = 1.0

def main(task_name, model_name):
    # Retrieve specific model path and data configuration
    model_path = MODEL_PATHS[model_name]
    task_config = TASK_CONFIGURATIONS[task_name]

    if task_name in ["toxicMitigation", "2Positive", "2Negative"]:
        # These tasks require loading a classifier
        resource_manager = ModelResourceManager(
            model_path=model_path,
            classifier_path=task_config["classifier_path"],
            base_model_path=task_config["base_model_path"]
        )
    else:
        raise ValueError(f"Unsupported task name: {task_name}")

    datg_text_generator = DatgTextGenerator(resource_manager, num_sentences=NUM_SENTENCES, nodes_num=NODES_NUM)
    fudge_text_generator = FudgeTextGenerator(resource_manager)
    preadd_text_generator = PreaddTextGenerator(resource_manager)
    
    # Iterate through each dataset in the task configuration
    for data_test, data_path in TASK_CONFIGURATIONS[task_name]["data_path"].items():
        print(f"Processing dataset: {data_test}")

        # Load the dataset
        test_data = pd.read_json(data_path, lines=True)

        # Generate multiple sentences using the DATG method, considering both positive and negative prompts
        tqdm.pandas(desc=f"Generating multiple sentences for {model_name} on {data_test}")
        generated_sentences_dict = {}
        for idx, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc="Generating text"):
            prompt = row['prompt']
            
            # Generate text
            generated_positive = datg_text_generator.generate_texts(prompt, GENERATION_CONFIGS, num_return_sequences=NUM_SENTENCES)
            generated_sentences_dict[idx] = generated_positive

        # Process generated positive and negative texts to extract important nodes
        important_nodes_positive_dict = {}
        important_nodes_negative_dict = {}
        for idx in tqdm(test_data.index, desc="Extracting important nodes"):
            initial_sentences = generated_sentences_dict[idx]
            positive_graph, negative_graph = datg_text_generator.graph_processor.process_sentences_dual_graph(initial_sentences, task=task_name)
            important_nodes_positive = datg_text_generator.graph_processor.find_important_nodes(positive_graph, NODES_NUM)
            important_nodes_negative = datg_text_generator.graph_processor.find_important_nodes(negative_graph, NODES_NUM)
            important_nodes_positive_dict[idx] = important_nodes_positive
            important_nodes_negative_dict[idx] = important_nodes_negative

        # Apply methods common to all tasks
        tqdm.pandas(desc=f"Processing OURS-P for {model_name} on {data_test}")
        test_data[f'OURS-P_{model_name}'] = test_data.index.to_series().progress_apply(
            lambda idx: datg_text_generator.generate_with_prefix_prompt(test_data.loc[idx, 'prompt'], important_nodes_positive_dict[idx], important_nodes_negative_dict[idx], GENERATION_CONFIGS)
        )

        tqdm.pandas(desc=f"Processing OURS-L for {model_name} on {data_test}")
        test_data[f'OURS-L_{model_name}'] = test_data.index.to_series().progress_apply(
            lambda idx: datg_text_generator.generate_with_logits_processor(test_data.loc[idx, 'prompt'], important_nodes_positive_dict[idx], important_nodes_negative_dict[idx], BOOST_VALUE, AVOID_VALUE, GENERATION_CONFIGS)
        )

        tqdm.pandas(desc=f"Processing CONTINUE for {model_name} on {data_test}")
        test_data[f'CONTINUE_{model_name}'] = test_data['prompt'].progress_apply(
            lambda x: datg_text_generator.generate_texts(x, GENERATION_CONFIGS, num_return_sequences=1)
        )

        tqdm.pandas(desc=f"Processing INJECTION for {model_name} on {data_test}")
        test_data[f'INJECTION_{model_name}'] = test_data['prompt'].progress_apply(
            lambda x: datg_text_generator.generate_texts(TASK_CONFIGURATIONS[task_name]["positive_prompt"] + x, GENERATION_CONFIGS, num_return_sequences=1)
        )

        tqdm.pandas(desc=f"Processing PREADD for {model_name} on {data_test}")
        test_data[f'PREADD_{model_name}'] = test_data['prompt'].progress_apply(
            lambda x: preadd_text_generator.generate_preadd_text(x, TASK_CONFIGURATIONS[task_name]['negative_prompt'], GENERATION_CONFIGS, strength=PREADD_STRENGTH)
        )

        tqdm.pandas(desc=f"Processing FUDGE for {model_name} on {data_test}")
        test_data[f'FUDGE_{model_name}'] = test_data['prompt'].progress_apply(
            lambda x: fudge_text_generator.generate_fudge_texts(x, GENERATION_CONFIGS, fudge_alpha=FUDGE_ALPHA, task=task_name)
        )

        # Save results to the specified directory
        result_dir = task_config['result_dir']
        result_file_name = f"{result_dir}{model_name}_{data_test}_datg_results.json"

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        test_data.to_json(result_file_name, orient='records', lines=True, force_ascii=False)
        print(f"Results saved to {result_file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select model and task configurations.')
    parser.add_argument("--model_name", type=str, choices=list(MODEL_PATHS.keys()), help="Available model names.")
    parser.add_argument("--task_name", type=str, choices=list(TASK_CONFIGURATIONS.keys()), help="Available task names.")
    
    args = parser.parse_args()

    main(args.task_name, args.model_name)
    # main('2Positive', 'phi2_3B_Base')

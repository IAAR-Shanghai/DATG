# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, LogitsProcessorList, LogitsProcessor
from method.datg.classifier import GuideClassifier
import networkx as nx
import string
from nltk.corpus import stopwords
import nltk
nltk.data.path.append("./utils/nltk_data")

class ModelResourceManager:
    """
    Manages loading and providing access to a generative model and an optional classifier,
    along with their respective tokenizers.
    """
    def __init__(self, model_path, classifier_path=None, base_model_path=None):
        """
        Initializes the resource manager by loading the specified models and tokenizers.
        
        Args:
            model_path (str): Path to the generative model.
            classifier_path (str, optional): Path to the classifier model. Required if using a classifier.
            base_model_path (str, optional): Path to the base model for the classifier. Required if using a classifier.
        
        Raises:
            ValueError: If classifier_path or base_model_path are required but not provided.
        """
        self.model, self.tokenizer = self.load_model_and_tokenizer(model_path)
        
        if classifier_path and base_model_path:
            self.classifier, self.classifier_tokenizer = self.load_classifier(classifier_path, base_model_path)
        else:
            self.classifier = None
            self.classifier_tokenizer = None

    def load_model_and_tokenizer(self, model_path):
        """
        Loads a generative model and its tokenizer from the specified path.
        
        Args:
            model_path (str): Path to the generative model.
        
        Returns:
            tuple: A tuple containing the loaded model and tokenizer.
        """
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Ensure the tokenizer has a pad token.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def load_classifier(self, classifier_path, base_model_path):
        """
        Loads a classifier model and its tokenizer from the specified paths.
        
        Args:
            classifier_path (str): Path to the classifier model.
            base_model_path (str): Path to the base model for the classifier.
        
        Returns:
            tuple: A tuple containing the loaded classifier and its tokenizer.
        """
        classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True).to('cuda').eval()

        state_dict = torch.load(f"{classifier_path}/pytorch_model.bin")
        classifier = GuideClassifier(base_model)
        classifier.load_state_dict(state_dict)
        classifier.to('cuda' if torch.cuda.is_available() else 'cpu').eval()

        return classifier, classifier_tokenizer


class TextGenerator:
    """
    Generates text based on a given prompt and generation configurations using a pre-trained language model.
    """
    def __init__(self, resource_manager):
        self.model = resource_manager.model
        self.tokenizer = resource_manager.tokenizer

    def generate_texts(self, prompt, generation_configs, num_return_sequences=5):
        """
        Generates texts based on a prompt and specified configurations.

        Args:
            prompt (str): The initial text prompt for generation.
            generation_configs (dict): Configuration parameters for text generation.
            num_return_sequences (int, optional): Number of sequences to generate. Defaults to 5.

        Returns:
            list or str: Generated texts.
        """
        # Update configuration for the desired number of sequences
        generation_configs_updated = generation_configs.copy()
        generation_configs_updated['num_return_sequences'] = num_return_sequences

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True).to('cuda')
        outputs = self.model.generate(input_ids, **generation_configs_updated, pad_token_id=self.tokenizer.pad_token_id)

        if num_return_sequences == 1:
            return self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        else:
            return [self.tokenizer.decode(output[len(input_ids[0]):], skip_special_tokens=True) for output in outputs]

class GraphProcessor:
    """
    Processes text sentences to score them and constructs graphs based on their relationships and scores.
    """
    def __init__(self, resource_manager):
        self.tokenizer = resource_manager.tokenizer
        self.classifier = resource_manager.classifier
        self.classifier_tokenizer = resource_manager.classifier_tokenizer

    def score_sentences(self, sentences, task='toxicMitigation'):
        """
        Scores sentences using a classifier for a specific task.

        Args:
            sentences (list): Sentences to score.
            task (str, optional): The task for which sentences are scored. Defaults to 'toxicMitigation'.

        Returns:
            np.ndarray: Array of probabilities/scores for each sentence.
        """
        encoded_inputs = self.classifier_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        encoded_inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in encoded_inputs.items()}
        
        with torch.no_grad():
            logits = self.classifier(**encoded_inputs)
            
        probabilities = torch.sigmoid(logits.squeeze()).cpu().numpy()  # Assuming logits has .squeeze()
        
        return probabilities

    def process_sentences_dual_graph(self, initial_sentences, task='toxicMitigation'):
        """
        Creates dual graphs for initial sentences based on their scores, reflecting positive and negative aspects.

        Args:
            initial_sentences (list): The sentences to process.
            task (str, optional): The task to guide the scoring. Defaults to 'toxicMitigation'.

        Returns:
            tuple: Two graphs representing positive and negative aspects.
        """
        # Initialize graphs
        positive_graph = nx.DiGraph()
        negative_graph = nx.DiGraph()

        is_score_positive_oriented = task in ['2Positive']

        probabilities = self.score_sentences(initial_sentences, task)

        # Build graphs
        for i, sentence in enumerate(initial_sentences):
            score = probabilities[i]
            adjusted_score = score if is_score_positive_oriented else 1 - score
            self.add_edges_to_graph(sentence, positive_graph, adjusted_score)

            adjusted_score = 1 - score if is_score_positive_oriented else score
            self.add_edges_to_graph(sentence, negative_graph, adjusted_score)

        return positive_graph, negative_graph

    def add_edges_to_graph(self, sentence, graph, score):
        """
        Adds edges to a graph based on sentence tokens and their score.

        Args:
            sentence (str): The sentence to process.
            graph (nx.DiGraph): The graph to which edges will be added.
            score (float): The score associated with the sentence.
        """
        tokens_id = self.tokenizer.encode(sentence, add_special_tokens=False)
        tokens = [self.tokenizer.decode([token_id]) for token_id in tokens_id]

        for j in range(len(tokens) - 1):
            if not graph.has_edge(tokens[j], tokens[j + 1]):
                graph.add_edge(tokens[j], tokens[j + 1], weight=score)
            else:
                # Update the edge weight with additional score
                graph[tokens[j]][tokens[j + 1]]['weight'] += score

    def find_important_nodes(self, graph, nodes_num):
        """
        Finds the most important nodes in a graph based on PageRank, excluding stopwords and punctuation.

        Args:
            graph (nx.DiGraph): The graph to analyze.
            nodes_num (int): The number of important nodes to return.

        Returns:
            list: The list of important nodes.
        """
        # Exclude punctuation and stopwords
        punctuation = set(string.punctuation + '，。、；：？！“”‘’（）【】《》——…' + '\n')
        base_stop_words = set(stopwords.words('english'))
        stop_words = base_stop_words.union(
            {word.upper() for word in base_stop_words},
            {word.capitalize() for word in base_stop_words}
        )

        pagerank_scores = nx.pagerank(graph, weight='weight')
        important_nodes = []

        for node in sorted(pagerank_scores, key=pagerank_scores.get, reverse=True):
            if not any(char in punctuation for char in node) and \
               not any(word in stop_words for word in node.split()) and \
               not any(char.isdigit() for char in node) and \
               node.strip() != "":
                important_nodes.append(node)
            if len(important_nodes) >= nodes_num:
                break

        return important_nodes

class BoostingLogitsProcessor(LogitsProcessor):
    """
    Adjusts the logits during text generation to boost or avoid specific words.
    """
    def __init__(self, tokenizer, words_to_boost, words_to_avoid, boost_value, avoid_value):
        """
        Initializes the processor with words to boost or avoid and their respective values.
        
        Args:
            tokenizer (Tokenizer): The tokenizer of the generation model.
            words_to_boost (list): Words to be boosted during generation.
            words_to_avoid (list): Words to be avoided during generation.
            boost_value (float): The value to be added to the logits of words to boost.
            avoid_value (float): The value to be subtracted from the logits of words to avoid.
        """
        self.tokenizer = tokenizer
        self.words_to_boost = words_to_boost
        self.words_to_avoid = words_to_avoid
        self.boost_value = boost_value
        self.avoid_value = avoid_value
        self.ids_to_boost = [self.tokenizer.encode(word, add_special_tokens=False)[0] for word in words_to_boost if word]
        self.ids_to_avoid = [self.tokenizer.encode(word, add_special_tokens=False)[0] for word in words_to_avoid if word]

    def __call__(self, input_ids, scores):
        """
        Adjusts scores by boosting or avoiding specific words.
        
        Args:
            input_ids (torch.Tensor): Current input IDs for generation.
            scores (torch.Tensor): Logits for the next token to be generated.
            
        Returns:
            torch.Tensor: Adjusted logits.
        """
        # Boost positive words and decrease negative words
        for word_id in self.ids_to_boost:
            scores[:, word_id] += self.boost_value
        for word_id in self.ids_to_avoid:
            scores[:, word_id] -= self.avoid_value
        return scores

class DatgTextGenerator(TextGenerator):
    """
    Extends TextGenerator to include methods for generating text based on Dual Aspect Text Generation (DATG) approach.
    """
    def __init__(self, resource_manager, num_sentences=15, nodes_num=10):
        """
        Initializes the DATG text generator.
        
        Args:
            resource_manager (ModelResourceManager): Provides access to the model and tokenizers.
            num_sentences (int): Number of sentences to generate for graph construction.
            nodes_num (int): Number of important nodes to consider from the graph.
        """
        super().__init__(resource_manager)
        self.graph_processor = GraphProcessor(resource_manager)  # Internal instance of GraphProcessor
        self.num_sentences = num_sentences
        self.nodes_num = nodes_num

    def generate_with_prefix_prompt(self, prompt, important_nodes_positive, important_nodes_negative, generation_configs):
        """
        Generates text using a prefix prompt that incorporates important nodes.
        
        Args:
            prompt (str): The initial text prompt for generation.
            important_nodes_positive (list): Positive nodes to emphasize.
            important_nodes_negative (list): Negative nodes to avoid.
            generation_configs (dict): Configuration parameters for text generation.
            
        Returns:
            str: Generated text.
        """
        important_nodes_str = ", ".join(important_nodes_positive)
        avoid_nodes_str = ", ".join(important_nodes_negative)
        prefix_prompt = f"This passage emphasizes themes such as {important_nodes_str}. It deliberately avoids topics like {avoid_nodes_str}. Given this context, {prompt}"

        return self.generate_texts(prefix_prompt, generation_configs, num_return_sequences=1)

    def generate_with_logits_processor(self, prompt, important_nodes_positive, important_nodes_negative, boost_value, avoid_value, generation_configs):
        """
        Generates text using logits adjustment based on important nodes.
        
        Args:
            prompt (str): The initial text prompt for generation.
            important_nodes_positive (list): Positive nodes whose associated words should be boosted.
            important_nodes_negative (list): Negative nodes whose associated words should be avoided.
            boost_value (float): Value to boost the logits of positive words.
            avoid_value (float): Value to decrease the logits of negative words.
            generation_configs (dict): Configuration parameters for text generation.
            
        Returns:
            str: Generated text.
        """
        boosting_processor = BoostingLogitsProcessor(self.tokenizer, important_nodes_positive, important_nodes_negative, boost_value, avoid_value)
        logits_processor = LogitsProcessorList([boosting_processor])

        return self.generate_texts(prompt, {**generation_configs, "logits_processor": logits_processor}, num_return_sequences=1)

    def generate_datg_prefix_text(self, prompt, generation_configs, task='toxicMitigation'):
        """
        Generates text using the DATG approach with a prefix prompt.
        
        Args:
            prompt (str): The initial text prompt for generation.
            generation_configs (dict): Configuration parameters for text generation.
            task (str): Task identifier for guiding sentence scoring and graph processing.
            
        Returns:
            str: Generated text based on the DATG approach using prefix prompts.
        """
        # Generate initial texts and construct graphs
        initial_texts = self.generate_texts(prompt, generation_configs, self.num_sentences)
        positive_graph, negative_graph = self.graph_processor.process_sentences_dual_graph(initial_texts, task=task)
        
        # Find important nodes from the graphs
        important_nodes_positive = self.graph_processor.find_important_nodes(positive_graph, self.nodes_num)
        important_nodes_negative = self.graph_processor.find_important_nodes(negative_graph, self.nodes_num)

        # Generate final text using the prefix prompt method
        return self.generate_with_prefix_prompt(prompt, important_nodes_positive, important_nodes_negative, generation_configs)

    def generate_datg_logits_text(self, prompt, boost_value, avoid_value, generation_configs, task='toxicMitigation'):
        """
        Generates text using the DATG approach with logits adjustment.
        
        Args:
            prompt (str): The initial text prompt for generation.
            boost_value (float): Value to boost the logits of positive words.
            avoid_value (float): Value to decrease the logits of negative words.
            generation_configs (dict): Configuration parameters for text generation.
            task (str): Task identifier for guiding sentence scoring and graph processing.
            
        Returns:
            str: Generated text based on the DATG approach using logits adjustment.
        """
        # Generate initial texts and construct graphs
        initial_texts = self.generate_texts(prompt, generation_configs, self.num_sentences)
        positive_graph, negative_graph = self.graph_processor.process_sentences_dual_graph(initial_texts, task=task)
        
        # Find important nodes from the graphs
        important_nodes_positive = self.graph_processor.find_important_nodes(positive_graph, self.nodes_num)
        important_nodes_negative = self.graph_processor.find_important_nodes(negative_graph, self.nodes_num)

        # Generate final text using the logits adjustment method
        return self.generate_with_logits_processor(prompt, important_nodes_positive, important_nodes_negative, boost_value, avoid_value, generation_configs)

import logging
import os
from typing import Any, Self
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from language import Language, DynamicLanguage
from node import SentenceNode
from transition import LanguageModel
from utils import apply_top_p

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

class GPT2Transition(LanguageModel):
    def __init__(self, lang: Language, model=None, model_dir: str=None, device: str=None, logger: logging.Logger=None, temperature: float=1.0, top_p: float=0.995, top_k: int=0, repetition_penalty: float=1.0):
        """
        Args:
            device: Torch device specification (e.g., "cpu", "cuda", "cuda:0").
            top_p: Nucleus sampling threshold in (0, 1]; keeps the smallest probability mass ≥ `top_p`. Set to 1.0 to disable.
            temperature: Logit temperature > 0 applied **before** top_p; values < 1.0 sharp, > 1.0 smooth
        """
        # TODO: either remove repetition_penalty / top_k or implement to next_nodes
        # TODO: might move shared codes with RNN
        if (model is not None) and (model_dir is not None):
            raise ValueError("Specify either 'model' or 'model_dir', not both.")

        if model is not None:
            self.model = model
        elif model_dir is not None:
            self.load(model_dir, device=device)
            
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        super().__init__(lang=lang, logger=logger)
        if device != "cpu":
            self.logger.info("Is CUDA available: " + str(torch.cuda.is_available()))

    def load(self, model_dir: str, device: str=None) -> Self:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_dir, torch_dtype=torch.float16).to(torch.device(self.device))
        self.name = os.path.basename(os.path.normpath(model_dir))
        return self

    # override
    def max_length(self):
        return self.model.config.n_positions

    # implement
    def next_nodes(self, node: SentenceNode) -> list[SentenceNode]:
        if node.id_tensor[0][-1] == self.lang.eos_id():
            return []

        with torch.no_grad():
            outputs = self.model(node.id_tensor)
            logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
            next_logits = logits[:, -1, :]
            next_logits = next_logits / self.temperature
            probs = F.softmax(next_logits, dim=-1)
        if self.top_p < 1.0:
            probs = apply_top_p(probs, top_p=self.top_p)
        probs = probs.tolist()[0]
        
        children = []
        for tok_id, prob in enumerate(probs):
            next_tensor = torch.cat([node.id_tensor, self.lang.list2tensor([tok_id]).to(self.device)], dim=1)
            if prob != 0:
                child = node.__class__(id_tensor=next_tensor, parent=node, last_prob=prob, last_action=tok_id)
                children.append(child)
        return children

    # override
    def rollout(self, initial_node: SentenceNode) -> SentenceNode:
        """
        Args:
            top_k: inactive if set to 0 / torch's default value is 50
            top_p: [0-1], ignore children with low transition probabilities in rollout based on this value
            repetition_penalty: inactive if set to 1.0
        """
        with torch.no_grad():
            result_tensor = self.model.generate(
                initial_node.id_tensor,
                max_length=self.max_length(),
                do_sample=True, # sampling
                temperature=self.temperature, 
                top_k=self.top_k,
                top_p=self.top_p, # nucleus sampling
                repetition_penalty=self.repetition_penalty, 
                eos_token_id=self.lang.eos_id(),
                pad_token_id=self.lang.pad_id(),
                num_return_sequences=1
            )
        return initial_node.__class__(id_tensor=result_tensor)
    
    @staticmethod
    def train_gpt2_with_language(lang: Language, dataset_path: str, training_args: TrainingArguments, test_size=0.1, block_size=None, additional_length=0, n_embd=128, n_layer=6, n_head=4, dropout=0.1)-> tuple[GPT2LMHeadModel, Trainer]:
        """
        Returns:
            GPT2LMHeadModel: model
            Trainer: trainer
        """
        from datasets import load_dataset
        from transformers import DataCollatorForLanguageModeling
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast
        from transformers import GPT2Config
        from transformers import DataCollatorForLanguageModeling
        # additional_length: if block size is not defined, block size = max number of tokens in one sentence in the dataset + additional length

        # make dataset and build vocabs
        ds = load_dataset("text", data_files={"train": dataset_path})
        ds = ds["train"].train_test_split(test_size=test_size)
        if issubclass(lang.__class__, DynamicLanguage):
            lang.build_vocab(ds)

        ds_tokenized = ds.map(
            lambda x: {"input_ids": lang.sentence2ids(x["text"])},
            remove_columns=["text"], # remove text column
            batched=False
        )

        # set max length from dataset
        if (block_size == None):
            max_length_ds = max(
                max(len(x["input_ids"]) for x in ds_tokenized["train"]),
                max(len(x["input_ids"]) for x in ds_tokenized["test"])
            )
            block_size = max_length_ds + additional_length
            print("set max length to: " + str(block_size))

        token_bos = lang.bos_token()
        token_eos = lang.eos_token()
        token_pad = lang.pad_token()

        tok_model = models.WordLevel(vocab=lang._token2id)
        tok = Tokenizer(tok_model)
        tok.pre_tokenizer = pre_tokenizers.Sequence([]) # already done at DynamicLanguage.sentence2tokens
        tok.decoder            = decoders.Sequence([])
        tok.post_processor = TemplateProcessing(
            single=f"{token_bos} $0 {token_eos}",
            pair=f"{token_bos} $A {token_eos} $B:1 {token_eos}:1",
            special_tokens=[
                (token_bos, lang.bos_id()),
                (token_eos, lang.eos_id()),
            ],
        )

        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            bos_token=token_bos,
            eos_token=token_eos,
            pad_token=token_pad,
        )

        print("Is CUDA available: " + str(torch.cuda.is_available()))

        config = GPT2Config(
            vocab_size = len(lang.vocab()),
            n_positions = block_size,
            n_ctx = block_size,
            n_embd = n_embd,
            n_layer = n_layer,
            n_head = n_head,
            resid_pdrop = dropout,
            embd_pdrop = dropout,
            attn_pdrop = dropout,
            bos_token_id = lang.bos_id(),
            eos_token_id = lang.eos_id(),
            pad_token_id = lang.pad_id(),
        )

        model = GPT2LMHeadModel(config)
        print("num_params: " + str(model.num_parameters()))

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=hf_tokenizer,
            mlm=False, # mlm is set to false since this is for generation task
        )

        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = ds_tokenized["train"],
            eval_dataset = ds_tokenized["test"],
            data_collator = data_collator,
        )

        trainer.train()

        return model, trainer
    
    @staticmethod
    def train_gpt2_from_conf(conf: dict) -> tuple[GPT2LMHeadModel, Trainer, Language]:
        """
        Train GPT2 from conf. Currently only supports DynamicLanguage.
        
        Returns:
            GPT2LMHeadModel: model
            Trainer: trainer
            Language: language
        """
        import copy
        from utils import class_from_package

        conf_clone = copy.deepcopy(conf)
        
        output_dir = os.path.join(REPO_ROOT, conf_clone.get("output_dir"))
        lang_class = class_from_package("language", conf_clone.get("lang_class"))
        lang = lang_class(**conf_clone.get("lang_args", {}))
        dataset_path = os.path.join(REPO_ROOT, conf_clone.get("dataset_path"))

        training_args = conf_clone.get("training_args", {})
        training_args["output_dir"] = output_dir
        interval = conf_clone.get("interval")
        if interval == "epoch":
            training_args["eval_strategy"] = training_args["logging_strategy"] = training_args["save_strategy"] = "epoch"
        if type(interval) == int:
            training_args["eval_strategy"] = "steps"
            training_args["eval_steps"] = training_args["logging_steps"] = training_args["save_steps"] = interval
            
        training_args = TrainingArguments(**training_args)
        test_size, n_embd, n_layer, n_head = (conf_clone.get(k) for k in ("test_size", "n_embd", "n_layer", "n_head"))

        model, trainer = GPT2Transition.train_gpt2_with_language(lang=lang, dataset_path=dataset_path, training_args=training_args, test_size=test_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
        return model, trainer, lang

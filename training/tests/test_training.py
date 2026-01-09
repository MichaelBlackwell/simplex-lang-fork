"""Integration tests for training pipeline."""

import pytest
import sys
import os
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip tests if torch not available
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


class TestModelLoading:
    """Tests for model loading."""

    @pytest.mark.slow
    def test_can_load_base_model(self):
        """Test that base model can be loaded."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for testing

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        assert model is not None
        assert tokenizer is not None

    @pytest.mark.slow
    def test_model_can_generate(self):
        """Test that model can generate text."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        inputs = tokenizer("Hello, how are you?", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert len(response) > 0


class TestLoRATraining:
    """Tests for LoRA training."""

    @pytest.mark.slow
    @pytest.mark.requires_torch
    def test_lora_config_creation(self):
        """Test LoRA config can be created."""
        from peft import LoraConfig, TaskType

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )

        assert config.r == 8
        assert config.lora_alpha == 16

    @pytest.mark.slow
    @pytest.mark.requires_torch
    def test_lora_model_creation(self):
        """Test LoRA model can be created from base model."""
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )

        model = get_peft_model(base_model, config)
        assert model is not None

        # Check trainable parameters are reduced
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        assert trainable_params < total_params
        assert trainable_params > 0


class TestTrainingDataFormat:
    """Tests for training data format."""

    def test_sentiment_data_format(self):
        """Test sentiment training data has correct format."""
        sample = {
            "text": "Analyze sentiment: This is great!\n\nAssistant: POSITIVE",
            "prompt": "Analyze sentiment: This is great!",
            "response": "POSITIVE",
            "metadata": {"source": "test", "label": "positive"},
        }

        assert "prompt" in sample
        assert "response" in sample
        assert "metadata" in sample
        assert "label" in sample["metadata"]

    def test_sql_data_format(self):
        """Test SQL training data has correct format."""
        sample = {
            "prompt": "Convert to SQL: Show all users",
            "response": "SELECT * FROM users",
            "metadata": {"source": "test", "schema": "users(id, name)"},
        }

        assert "prompt" in sample
        assert "response" in sample

    def test_can_save_and_load_jsonl(self):
        """Test JSONL save/load roundtrip."""
        samples = [
            {"prompt": "test1", "response": "resp1"},
            {"prompt": "test2", "response": "resp2"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
            temp_path = f.name

        try:
            loaded = []
            with open(temp_path, "r") as f:
                for line in f:
                    loaded.append(json.loads(line))

            assert len(loaded) == 2
            assert loaded[0]["prompt"] == "test1"
        finally:
            os.unlink(temp_path)


class TestTrainingSmokeTest:
    """Smoke tests for training (minimal runs)."""

    @pytest.mark.slow
    @pytest.mark.requires_torch
    def test_sft_trainer_can_initialize(self):
        """Test SFT trainer can be initialized."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
        from peft import LoraConfig, TaskType
        from datasets import Dataset

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )

        # Tiny dataset
        dataset = Dataset.from_dict(
            {
                "text": [
                    "User: Hello\n\nAssistant: Hi there!",
                    "User: Bye\n\nAssistant: Goodbye!",
                ]
            }
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                num_train_epochs=1,
                max_steps=1,  # Just 1 step for smoke test
                per_device_train_batch_size=1,
                logging_steps=1,
                save_steps=100,
                report_to="none",
            )

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=tokenizer,
                peft_config=lora_config,
            )

            assert trainer is not None

    @pytest.mark.slow
    @pytest.mark.requires_torch
    def test_can_run_single_training_step(self):
        """Test that a single training step can run."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
        from peft import LoraConfig, TaskType
        from datasets import Dataset

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )

        dataset = Dataset.from_dict(
            {"text": ["User: Test\n\nAssistant: Response"] * 4}
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                num_train_epochs=1,
                max_steps=1,
                per_device_train_batch_size=2,
                logging_steps=1,
                save_steps=100,
                report_to="none",
            )

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=tokenizer,
                peft_config=lora_config,
            )

            # Run training
            result = trainer.train()

            assert result is not None
            assert "train_loss" in result.metrics or hasattr(result, "training_loss")

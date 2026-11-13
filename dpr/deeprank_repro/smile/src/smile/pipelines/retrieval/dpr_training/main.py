"""Thin SMILE wrapper around the shared SMILE-Lightning trainer for DPR."""

from typing import Dict, List

from smile.pipelines.smile_lightning.train import train as smile_lightning_train
from smile.utils import Input, Pipeline, PipelineLoader, dsl


config = PipelineLoader.get_pipeline_config()


def _compose_overrides(
    *,
    trainer_config: str,
    strategy_config: str,
    model_name_or_path: str,
    epochs: int,
    per_device_batch_size: int,
    val_batch_size: int,
    learning_rate: float,
    adam_eps: float,
    weight_decay: float,
    warmup_steps: int,
    seq_length: int,
    workers: int,
    seed: int,
    enable_average_rank: bool,
    average_rank_start_epoch: int,
    average_rank_chunk_size: int,
    validation_context_batch_size: int,
    validation_num_hard_negatives: int,
    validation_num_other_negatives: int,
    validation_max_questions: int,
    extra_override: str,
) -> str:
    """Build overrides only from graph-build-time values.

    Every value used in this function is listed in ``non_pipeline_parameters``.
    The two data paths intentionally remain runtime data ports and are referenced
    through the component's stable ``$input_path_*`` mount variables.
    """

    if not model_name_or_path.strip():
        raise ValueError("model_name_or_path must not be empty")
    if any(character.isspace() for character in model_name_or_path):
        raise ValueError("model_name_or_path cannot contain whitespace in the SMILE command override")
    if epochs <= 0 or per_device_batch_size <= 0 or val_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if seq_length <= 0 or workers < 0 or warmup_steps < 0:
        raise ValueError("seq_length must be positive; workers and warmup_steps cannot be negative")

    average_rank_enabled = str(bool(enable_average_rank)).lower()
    parts = [
        f"trainer={trainer_config}",
        f"trainer/strategy={strategy_config}",
        "trainer.precision=32-true",
        f"trainer.max_epochs={epochs}",
        "trainer.gradient_clip_val=2.0",
        "trainer.gradient_clip_algorithm=norm",
        f"++seed={seed}",
        "datamodule.train_path=$input_path_0",
        "datamodule.val_path=$input_path_1",
        f"datamodule.tokenizer_name_or_path={model_name_or_path}",
        "datamodule.use_fast_tokenizer=false",
        f"datamodule.batch_size={per_device_batch_size}",
        f"datamodule.val_batch_size={val_batch_size}",
        f"datamodule.num_workers={workers}",
        f"datamodule.max_question_length={seq_length}",
        f"datamodule.max_context_length={seq_length}",
        f"datamodule.seed={seed}",
        "datamodule.drop_last=true",
        f"datamodule.validation_num_hard_negatives={validation_num_hard_negatives}",
        f"datamodule.validation_num_other_negatives={validation_num_other_negatives}",
        f"datamodule.validation_max_questions={validation_max_questions}",
        f"module.model.question_model_name_or_path={model_name_or_path}",
        "module.model.context_model_name_or_path=null",
        f"module.learning_rate={learning_rate}",
        f"module.adam_eps={adam_eps}",
        f"module.weight_decay={weight_decay}",
        f"module.warmup_steps={warmup_steps}",
        f"module.enable_average_rank={average_rank_enabled}",
        f"module.average_rank_start_epoch={average_rank_start_epoch}",
        f"module.average_rank_chunk_size={average_rank_chunk_size}",
        f"module.validation_context_batch_size={validation_context_batch_size}",
    ]
    if extra_override and extra_override.strip():
        parts.append(extra_override.strip())
    return " ".join(parts)


@dsl.pipeline(
    name="dpr_training",
    description="Train the EMNLP 2020 Dense Passage Retriever with SMILE-Lightning",
    default_datastore=config.storage.default_datastore,
    non_pipeline_parameters=[
        "trainer_config",
        "strategy_config",
        "model_name_or_path",
        "epochs",
        "per_device_batch_size",
        "val_batch_size",
        "learning_rate",
        "adam_eps",
        "weight_decay",
        "warmup_steps",
        "seq_length",
        "workers",
        "seed",
        "enable_average_rank",
        "average_rank_start_epoch",
        "average_rank_chunk_size",
        "validation_context_batch_size",
        "validation_num_hard_negatives",
        "validation_num_other_negatives",
        "validation_max_questions",
        "config_additional_override",
        "node_count",
        "process_count_per_node",
        "job_type",
        "compute_target",
        "is_preemptible",
        "environment_variables",
        "instance_type",
        "is_singularity_job",
        "singularity_vc",
        "vc_arm_id",
        "sla_tier",
        "priority",
        "locations",
        "environment_name",
        "output_datastore",
        "output_path",
    ],
)
def dpr_training(
    train_data: Input,
    validation_data: Input,
    trainer_config: str = "gpu",
    strategy_config: str = "ddp",
    model_name_or_path: str = "bert-base-uncased",
    epochs: int = 40,
    per_device_batch_size: int = 16,
    val_batch_size: int = 16,
    learning_rate: float = 2.0e-5,
    adam_eps: float = 1.0e-8,
    weight_decay: float = 0.0,
    warmup_steps: int = 1237,
    seq_length: int = 256,
    workers: int = 4,
    seed: int = 12345,
    enable_average_rank: bool = True,
    average_rank_start_epoch: int = 30,
    average_rank_chunk_size: int = 4096,
    validation_context_batch_size: int = 128,
    validation_num_hard_negatives: int = 30,
    validation_num_other_negatives: int = 30,
    validation_max_questions: int = 10000,
    config_additional_override: str = "",
    node_count: int = 1,
    process_count_per_node: int = 8,
    job_type: str = "distributed",
    compute_target: str = None,
    is_preemptible: bool = False,
    environment_variables: Dict[str, str] = {},
    instance_type: str = "ND96isr_MI300X_v5",
    is_singularity_job: bool = True,
    singularity_vc: str = "webxtcoresearchwus3",
    vc_arm_id: str = None,
    sla_tier: str = "Premium",
    priority: str = "Low",
    locations: List[str] = None,
    environment_name: str = None,
    output_datastore: str = "relevance_c09",
    output_path: str = None,
) -> Pipeline:
    """Mount NQ JSON files and run one DPR training step."""

    overrides = _compose_overrides(
        trainer_config=trainer_config,
        strategy_config=strategy_config,
        model_name_or_path=model_name_or_path,
        epochs=epochs,
        per_device_batch_size=per_device_batch_size,
        val_batch_size=val_batch_size,
        learning_rate=learning_rate,
        adam_eps=adam_eps,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        seq_length=seq_length,
        workers=workers,
        seed=seed,
        enable_average_rank=enable_average_rank,
        average_rank_start_epoch=average_rank_start_epoch,
        average_rank_chunk_size=average_rank_chunk_size,
        validation_context_batch_size=validation_context_batch_size,
        validation_num_hard_negatives=validation_num_hard_negatives,
        validation_num_other_negatives=validation_num_other_negatives,
        validation_max_questions=validation_max_questions,
        extra_override=config_additional_override,
    )
    train_step = smile_lightning_train(
        job_type=job_type,
        project_name="retrieval/dpr",
        config_name="train_app",
        config_additional_override=overrides,
        node_count=node_count,
        process_count_per_node=process_count_per_node,
        is_preemptible=is_preemptible,
        environment_variables={str(key): str(value) for key, value in environment_variables.items()},
        compute_target=compute_target,
        instance_type=instance_type,
        is_singularity_job=is_singularity_job,
        singularity_vc=singularity_vc,
        vc_arm_id=vc_arm_id,
        sla_tier=sla_tier,
        priority=priority,
        locations=locations,
        environment_name=environment_name,
        input_path_0=train_data,
        input_path_1=validation_data,
        output_datastore=output_datastore,
        output_path=output_path,
    )
    return {"model_output": train_step.outputs.output_path}

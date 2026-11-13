"""Fine-tune an NQ DPR checkpoint on NFCorpus using SMILE-Lightning."""

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
    max_question_length: int,
    max_context_length: int,
    context_overlap: int,
    hard_negative_count: int,
    hard_negatives_per_query: int,
    workers: int,
    seed: int,
    similarity_mode: str,
    score_scale: float,
    extra_override: str,
) -> str:
    """Build graph-time Hydra options; AML data ports resolve at runtime."""

    if not model_name_or_path.strip() or any(character.isspace() for character in model_name_or_path):
        raise ValueError("model_name_or_path must be nonempty and contain no whitespace")
    if epochs <= 0 or per_device_batch_size <= 0 or val_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if max_question_length <= 0 or max_context_length < 8:
        raise ValueError("max_question_length must be positive and max_context_length must be at least 8")
    if not 0 <= context_overlap < max_context_length - 3:
        raise ValueError("context_overlap must be between zero and max_context_length - 4")
    if hard_negative_count < 1:
        raise ValueError("hard_negative_count must be positive")
    if hard_negatives_per_query != 1:
        raise ValueError("NFCorpus fine-tuning currently supports exactly one hard negative per positive pair")
    if workers < 0 or warmup_steps < 0 or score_scale <= 0:
        raise ValueError("workers and warmup_steps cannot be negative; score_scale must be positive")
    if similarity_mode not in ("dot", "cosine"):
        raise ValueError("similarity_mode must be dot or cosine")

    parts = [
        f"trainer={trainer_config}",
        f"trainer/strategy={strategy_config}",
        "trainer.precision=32-true",
        f"trainer.max_epochs={epochs}",
        "trainer.gradient_clip_val=2.0",
        "trainer.gradient_clip_algorithm=norm",
        f"++seed={seed}",
        "datamodule.data_root=$input_path_0",
        f"datamodule.tokenizer_name_or_path={model_name_or_path}",
        "datamodule.use_fast_tokenizer=false",
        f"datamodule.batch_size={per_device_batch_size}",
        f"datamodule.val_batch_size={val_batch_size}",
        f"datamodule.num_workers={workers}",
        f"datamodule.max_question_length={max_question_length}",
        f"datamodule.max_context_length={max_context_length}",
        f"datamodule.context_overlap={context_overlap}",
        f"datamodule.hard_negative_count={hard_negative_count}",
        f"datamodule.hard_negatives_per_query={hard_negatives_per_query}",
        f"datamodule.seed={seed}",
        "datamodule.drop_last=false",
        f"module.model.question_model_name_or_path={model_name_or_path}",
        "module.model.context_model_name_or_path=null",
        "module.nq_checkpoint_path=$input_path_1",
        f"module.learning_rate={learning_rate}",
        f"module.adam_eps={adam_eps}",
        f"module.weight_decay={weight_decay}",
        f"module.warmup_steps={warmup_steps}",
        f"module.similarity_mode={similarity_mode}",
        f"module.score_scale={score_scale}",
    ]
    if extra_override and extra_override.strip():
        parts.append(extra_override.strip())
    return " ".join(parts)


@dsl.pipeline(
    name="nfcorpus_finetune",
    description="Adapt an NQ-trained DPR to NFCorpus and select by dev Recall@10",
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
        "max_question_length",
        "max_context_length",
        "context_overlap",
        "hard_negative_count",
        "hard_negatives_per_query",
        "workers",
        "seed",
        "similarity_mode",
        "score_scale",
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
def nfcorpus_finetune(
    nfcorpus_data: Input(type="uri_folder"),  # noqa: F821
    nq_checkpoint: Input(type="uri_file"),  # noqa: F821
    trainer_config: str = "gpu",
    strategy_config: str = "ddp",
    model_name_or_path: str = "bert-base-uncased",
    epochs: int = 5,
    per_device_batch_size: int = 8,
    val_batch_size: int = 8,
    learning_rate: float = 1.0e-5,
    adam_eps: float = 1.0e-8,
    weight_decay: float = 0.0,
    warmup_steps: int = 10,
    max_question_length: int = 256,
    max_context_length: int = 256,
    context_overlap: int = 64,
    hard_negative_count: int = 20,
    hard_negatives_per_query: int = 1,
    workers: int = 4,
    seed: int = 12345,
    similarity_mode: str = "dot",
    score_scale: float = 1.0,
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
    """Mount one NFCorpus data folder plus the NQ checkpoint for warm start."""

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
        max_question_length=max_question_length,
        max_context_length=max_context_length,
        context_overlap=context_overlap,
        hard_negative_count=hard_negative_count,
        hard_negatives_per_query=hard_negatives_per_query,
        workers=workers,
        seed=seed,
        similarity_mode=similarity_mode,
        score_scale=score_scale,
        extra_override=config_additional_override,
    )
    train_step = smile_lightning_train(
        job_type=job_type,
        project_name="retrieval/nfcorpus",
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
        input_path_0=nfcorpus_data,
        input_path_1=nq_checkpoint,
        output_datastore=output_datastore,
        output_path=output_path,
    )
    return {"model_output": train_step.outputs.output_path}

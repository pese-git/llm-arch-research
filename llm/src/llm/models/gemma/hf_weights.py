"""
Перенос весов Gemma из формата HuggingFace (`GemmaForCausalLM`).

Имена слоёв и перестановка строк Q/K — как у LLaMA (см. llm.models.llama.hf_weights).
Отличие одно: `GemmaRMSNorm` хранит вес w и умножает на (1 + w), а RMSNorm здесь — на сам
вес, поэтому к весам всех RMSNorm прибавляется 1.

Конфиг должен повторять HF: "num_kv_heads", "head_size" (head_dim), "intermediate_size",
"bias": False, "tie_word_embeddings": True, "scale_embeddings": True, rms_norm_eps и rope_theta.
"""

from llm.models.llama.hf_weights import convert_hf_state_dict as _convert_llama_family


def convert_hf_state_dict(hf_state_dict: dict, num_heads: int, num_kv_heads: int = None) -> dict:
    """
    Преобразует state_dict `GemmaForCausalLM` в state_dict `Gemma`.

    num_heads — config.num_attention_heads, num_kv_heads — config.num_key_value_heads
    (по умолчанию num_heads).
    """
    result = _convert_llama_family(hf_state_dict, num_heads=num_heads, num_kv_heads=num_kv_heads)
    for key in result:
        if key.endswith("._w"):  # веса RMSNorm: (1 + w) в HF → w здесь
            result[key] = result[key] + 1
    return result
